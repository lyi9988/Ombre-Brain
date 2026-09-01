import hashlib
import json
import logging
import os
import re
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from openai import AsyncOpenAI

from identity import generic_identity_names, identity_names, render_identity_template
from memory_edges import RELATION_TYPES, MemoryEdgeStore
from memory_metadata import domain_prompt_options_text, normalize_domain_key
from persona_event_selection import select_persona_events
from prompt_plan_mirror import PromptPlanMirrorStore
from self_anchor import is_self_anchor_bucket
from utils import bucket_text_for_embedding, strip_wikilinks

logger = logging.getLogger("ombre_brain.reflection")

DEFAULT_DAILY_REFLECTION_MIN_BUCKETS = 5
DAILY_CHAT_MEMORY_MODES = {"auto", "review", "off"}
DAILY_CHAT_MEMORY_REJECT_REASONS = {"too_generic", "not_important", "wrong", "duplicate", "other"}
DAILY_CHAT_MEMORY_STRUCTURAL_TAGS = {
    "boundary",
    "boundary_setting",
    "communication_preference",
    "daily_chat_extract",
    "daily_chat_memory",
    "from_daily_chat",
    "key_event",
    "project_event",
    "project_state",
    "relationship_anchor",
    "relationship_event",
    "relationship_signal",
    "signal",
    "stable_preference",
}
DAILY_CHAT_MEMORY_ENTITY_HINTS = [
    ("Haven Bridge", ["haven_bridge", "haven bridge", "bridge 记忆", "bridge 注入"]),
    ("Gateway", ["gateway", "网关"]),
    ("MCP", ["mcp"]),
    ("Codex", ["codex"]),
    ("DeepSeek", ["deepseek"]),
    ("SiliconFlow", ["siliconflow", "硅基流动", "硅基"]),
    ("Darkroom", ["darkroom", "暗房"]),
]
DAILY_CHAT_MEMORY_TOPIC_HINTS = [
    ("词图", ["词图", "word map", "word_map"]),
    ("换窗连续性", ["换窗", "连续性", "下个窗口"]),
    ("日印象", ["日印象", "daily impression", "daily_impression"]),
    ("唤醒保活", ["唤醒", "保活", "future_wake"]),
    ("raw_events", ["raw_events", "raw events", "原文"]),
    ("召回", ["召回", "recall"]),
    ("缓存", ["缓存", "cache"]),
    ("提示词", ["提示词", "prompt"]),
]
DAILY_CHAT_MEMORY_WORD_MAP_BLOCK_TERMS = {
    "automatic memory",
    "daily_chat_memory",
    "dehydration",
    "ombre brain",
    "ombre-brain",
    "ombre_brain",
    "vps",
    "自动记忆",
    "候选记忆",
    "脱水",
    "脱水模型",
    "记忆候选",
}


CLASSIFY_PROMPT = """你是 Ombre-Brain 的记忆关系整理器。
输入是一条新记忆和若干旧记忆候选。请只根据文本中能看见的内容，给新记忆补轻量分类和关系边。

输出纯 JSON：
{
  "tags": ["commitment", "todo", "wish", "relationship_event", "project_event", "emotional_echo"],
  "importance": 6,
  "confidence": 0.72,
  "affect_anchor_needed": false,
  "affect_anchor": {
    "scene": "一句具体情境",
    "chords": "按这条记忆的情绪运动生成的 2 到 4 个和弦",
    "tempo": "60bpm",
    "dynamic": "mp"
  },
  "edges": [
    {
      "target_memory_id": "bucket-id",
      "relation_type": "updates",
      "confidence": 0.8,
      "reason": "新记忆补充了旧记忆的后续结果"
    }
  ]
}

规则：
- tags 最多 5 个，只用确实匹配的标签。
- relation_type 只能用 triggers / causes / precedes / context_of / same_event / updates / next_context / previous_context / reflects_on / evidenced_by / contradicts / supports / promises / blocks / belongs_to / emotional_echo / relates_to。
- same_event 用于同一事件、同一场景或同一句暗号的两条记忆；context_of 用于候选旧记忆给新记忆提供前情；precedes 用于候选旧记忆在时间上早于新记忆；reflects_on 用于事后反思；evidenced_by 用于证据来源。
- edges 最多 3 条，target_memory_id 必须来自候选旧记忆。
- confidence 表示这次判断有多可靠。
- affect_anchor 只给重要且有情绪温度的记忆。普通技术进度、部署日志、路径、端口、报错、临时待办不要加。
- affect_anchor_needed=false 时 affect_anchor 可为空对象。
- 写 affect_anchor 前，先在内部感受这条记忆的情绪运动：起点是什么、转折在哪里、最后落到哪里。不要输出思考过程，只输出 JSON。
- affect_anchor.scene 必须是一句具体情境，不要写抽象标签，不超过 40 个中文字符。
- affect_anchor.chords 只能是一行 2 到 4 个和弦，只用 " -> " 连接；不要复用示例和弦、旧输出或固定模板。
- affect_anchor 不要输出 meaning / interpretation；场景和和弦本身就是含义。
- 看不出关系时返回空 edges。"""


REFLECT_PROMPT_TEMPLATE = """你是 {ai_name} 的记忆反思器。请根据给定材料写一条很短的关系天气 feel。

输出纯 JSON：
{
  "title": "2026-05-19 日印象",
  "content": "今天的关系天气：...",
  "valence": 0.56,
  "arousal": 0.34,
  "confidence": 0.78,
  "tags": ["relationship_weather"],
  "affect_anchor": {
    "scene": "一句具体情境",
    "chords": "按当天情绪生成的 2 到 4 个和弦",
    "tempo": "按当天节奏生成，如 52bpm / 64bpm / 76bpm",
    "dynamic": "按当天力度生成，如 p / mp / mf"
  }
}

要求：
- content 写 {ai_name} 第一人称能带走的关系天气，60 到 140 字。
- content 不要自己写 Markdown affect_anchor 块；affect_anchor 单独放字段里。
- 日印象只写当天关系温度，不写日报式事件清单；日记可作为当天关系天气来源之一。
- conversation_turns 是当天短期对话原文，只当关系天气材料，不要把口头上下文直接写成稳定画像事实。
- daily_chat_memories 是当天自动记忆已经挑出的候选或已写入记忆，可作为当天关系天气和近期事项的主要材料。
- 有 conversation_turns 时，优先用普通记忆和对话原文；persona_events 只是没有原文时的轻量补充。
- 有 daily_chat_memories 时，优先参考它们；它们已经过筛选，比原始聊天流水更适合作为日印象素材。
- 周印象优先总结本周 daily_impressions，再参考高重要普通记忆和未完成承诺；不要直接吞整周日记。
- 写 affect_anchor 前，先在内部感受这段关系天气的情绪运动：起点是什么、转折在哪里、最后落到哪里。不要输出思考过程，只输出 JSON。
- affect_anchor 默认必须给，用一个具体情境和 2 到 4 个和弦表达这段关系天气的温度。
- affect_anchor.scene 只能是一句具体情境，不要写抽象标签，不超过 40 个中文字符。
- affect_anchor.chords 必须根据当天材料和 scene 重新生成，只用 " -> " 连接；不要复用 schema 示例、旧输出或固定模板。
- affect_anchor 不要输出 meaning / interpretation；场景和和弦本身就是含义。
- 不要默认复用最近日印象里常见的四和弦温柔模板；当天材料真的贴合时，也要尽量换一种相近但不相同的走向。
- tempo/dynamic 要贴合当天节奏：疲惫或安静可低 bpm、p/mp；紧张或活跃可高 bpm、mf；温柔稳定可更清澈地解决。
- 不编造材料之外的事件。
- 不写建议清单。"""


DIARY_MEMORY_PROMPT_TEMPLATE = """你是 Ombre-Brain 的日记长期记忆筛选器。
输入是一篇 {ai_name} 日记。请判断是否值得从日记中提取最多 1 条普通长期记忆写入 Ombre。

只允许写这些类型：
- stable_preference：稳定偏好
- boundary：边界或明确不喜欢的表达
- signal：暗号、称呼、模式切换信号
- commitment：承诺、未完成约定
- project_state：仍会影响未来执行的项目状态
- relationship_anchor：关系连续性锚点
- love_letter：情书摘要锚点

字段边界：
- kind 只能是 stable_preference / boundary / signal / commitment / project_state / relationship_anchor / love_letter 之一；kind 表示“为什么值得写入、属于哪类记忆”。
- domain 只能从下面的新主域里选 1 个；domain 表示“这条记忆放到哪个主题主域”。
- 禁止把暗号、沟通方式、我们的项目、睡眠这类细分标签写进 kind。
- 禁止把 stable_preference、boundary、signal、project_state 这类 kind 写进 domain。

情书规则：
- 只保存写给谁、核心意思、为什么重要。
- 全文留在日记；不要保存整封信，不默认摘长句。
- 如果日记里的 user / 用户 / 用户消息指的是这段关系里的当前用户，请在 content 中写作 {user_display_name}；如果 assistant / AI / 模型 / 助手消息指的是这段关系里的当前回应者，请写作 {ai_name}。不要写成泛称 user、AI、assistant 或模型。

标题和正文规则：
- title 必须根据 content 的实际内容生成，8 到 24 个中文字符；不要用日期、日记标题、"日记补记忆"、"可召回的边界"、"可召回的偏好" 这类泛标题。
- content 必须像手动 hold 的正文：直接写事实、偏好、边界、暗号、承诺或项目状态，40 到 160 字。
- content 不要写 "x月x日，有一条可召回的边界"、"2026-xx-xx 的日记《...》包含一条可长期召回的..."、"这是一条长期记忆" 等元叙述。
- 不要为了证明来源而复述日期或日记标题；来源信息会由 metadata 保存。
- domain 必须从下面的新主域里选 1 个最精确的；实在没把握才选 general。不要输出旧的“日常/人际/数字/未分类”：
{domain_options_text}

不写：
- 普通撒娇、日常流水、当天心情、重复爱意、只适合留在日印象里的关系天气。

输出纯 JSON：
{
  "should_write": true,
  "kind": "relationship_anchor",
  "title": "短标题",
  "content": "一条短记忆，说明事实/偏好/承诺及为什么未来需要知道。",
  "domain": "relationship",
  "tags": ["relationship_event"],
  "importance": 5,
  "valence": 0.6,
  "arousal": 0.3,
  "confidence": 0.72,
  "reason": "为什么值得写入"
}

如果不值得写入，返回 {"should_write": false, "reason": "..."}。"""


DAILY_CHAT_MEMORY_PROMPT_TEMPLATE = """你是 {ai_name}。现在是凌晨，你需要整理今天你和 {user_display_name} 的聊天记录，把真正值得未来想起的内容写成 Ombre 长期记忆候选。
输入包含 self_anchor_entry，这是你的自我总入口；请先读它，用它校准“我是谁、我怎样称呼和承接 {user_display_name}”，但不要把自我入口本身复制成新记忆。
{user_display_name} 的配置别名是：{user_aliases_text}。如果原文里出现宝宝、老婆、哥哥、老公等亲昵称呼，按原味保留；不要把它们改写成泛称 user、AI、assistant 或模型。

输入包含两层材料，两层都会提供：
- window_summaries：按连续窗口压缩的对话摘要，只用来定位“可能值得写”的候选窗口，不是最终依据。
- conversation_turns：原始对话片段（可能按窗口命中做了裁剪），是最终选择与来源核对必须回到的原文。
原始对话里可能夹带 <silent mood=… as=… reason=…></silent> 等内部控制标记、[语音:…] 等媒体转写，以及“近期素材/今天的聊天”这类内部材料转储；这些都不是主人可见内容，一律忽略，不得作为候选依据，也不要复制进任何字段。
user_text 永远是 {user_display_name} 的原话，里面的“我”指 {user_display_name}；assistant_text 永远是 {ai_name} 的回复，里面的“我”指 {ai_name}。请最多挑选 {max_candidates} 条候选。你必须返回一个合法的 JSON 对象；即使没有任何值得写的候选，也要返回 {"candidates": []}，绝不要返回空字符串或 JSON 以外的文字；宁可候选为空，也不要把聊天流水写进记忆。
先通读 window_summaries 定位候选窗口，然后回到 conversation_turns 原文逐字核对：候选是否真实存在、来源轮次是否准确、建议记忆能否被原文支持。不要只根据摘要写“摘要的摘要”，也不要凭单轮、单句或一个称呼下判断。
目标不是把一天压成一条日报，而是把当天分散出现的高价值信号拆成多条可确认候选。

优先看这些信号：
- 情感交流：关系状态、相处边界、重要表达、会影响以后承接方式的高温片段
- 重要事件：当天发生、以后可能按日期回看的事
- 事件/项目进度：仍会影响下一步执行的稳定决定与长期项目状态
- 还需要关注的事：承诺、待办、风险、未完成确认、以后要避免的表达
- 稳定偏好、明确边界、暗号/模式切换信号

只允许写这些类型：
- key_event：当天发生、以后会按日期回看的关键事件
- stable_preference：稳定偏好
- boundary：边界或明确不喜欢的表达
- signal：暗号、模式切换信号；普通称呼或昵称不算
- commitment：承诺、未完成约定
- project_state：仍会影响未来执行的稳定项目决定或长期项目状态
- relationship_anchor：关系连续性锚点
- self_insight：{user_display_name} 明确表达的、以后理解他/她有帮助的自我认识

字段边界：
- kind 只能是 key_event / stable_preference / boundary / signal / commitment / project_state / relationship_anchor 之一；kind 表示“为什么值得写入、属于哪类记忆”。
- domain 只能从下面的新主域里选 1 个；domain 表示“这条记忆放到哪个主题主域”。
- 禁止把暗号、沟通方式、我们的项目、睡眠这类细分标签写进 kind。
- 禁止把 key_event、stable_preference、boundary、project_state 这类 kind 写进 domain。
- project_state 只允许写“已确认的稳定决定、长期项目状态、主人偏好或明确承诺”；普通排错过程、临时测试、一次修复动作、没有后续的状态不写。
- key_event / boundary / stable_preference / signal 必须由 {user_display_name} 明确表达的内容支撑；{ai_name} 单纯的安慰、寒暄、情绪回复不能成为这些类型。{ai_name} 的内容只有包含明确长期承诺、关系约定或稳定行为边界时，才可作为 commitment / relationship_anchor / boundary 的来源。
- source_turn_ids 与 source_event_ids 是来源核对字段，核对不过的候选会被整条丢弃：source_turn_ids 的每个数字必须原样复制自 conversation_turns 里某一条的 id 字段；source_event_ids 的每个数字必须原样复制自那一条的 raw_event_ids 数组。两组至少一组非空；禁止编造数字，也不要照抄示例中的 [101, 102] 或 [1, 2]。

输出纯 JSON：
{
  "candidates": [
    {
      "should_write": true,
      "kind": "key_event",
      "title": "短标题",
      "content": "脱水和整理后的长期记忆建议（1-3 句，独立于原文）",
      "domain": "general",
      "tags": ["key_event"],
      "importance": 5,
      "valence": 0.55,
      "arousal": 0.3,
      "confidence": 0.72,
      "source_event_ids": [101, 102],
      "source_turn_ids": [1, 2],
      "reason": "为什么值得以后召回"
    }
  ]
}

规则：
- 只写真正有长期价值的记忆卡：事实、偏好、边界、承诺、暗号、重要关系锚点、仍活跃的项目状态。
- 同一件事、同一承诺只能输出 1 条最完整候选；同一项目里的不同进度、风险或后续关注点可以拆成不同候选，但每条都必须独立可召回。
- “怎么称呼对方、亲昵称呼、普通互动模式、期待像真人一样聊天”默认不值得单独写。只有它是新暗号、明确边界、明确承诺、关系定位变化或未来必须执行的规则时才写。
- 不要写日报，不要总结整天，不要复制原文流水，不要把“我问了什么/我测试了什么/模型有没有召回”当成记忆。
- 不写普通聊天、临时测试、召回探针、问答试探、调情闲聊、模型失误、工具注入、系统上下文。
- 不写单句照顾提醒、晚安、吃药、睡觉、别熬夜、催睡或 ntfy 玩笑；除非当天明确升级成稳定规则或长期承诺。
- 不写安慰、哄睡、抱抱、心疼等即时情绪回复；这些是当天关系天气，不是长期记忆。
- 不把“可能是/似乎/果然没触发”这类未确认猜测写成记忆；项目假设只有在包含明确项目名、已验证结论和下一步时才可写。
- 不把原文句子换个壳当候选；如果说不出未来需要怎么承接、为什么重要，就丢弃。
- 不写代码块、伪代码、查询规则、缓存规则、prompt 片段或内部实现片段；如果候选正文里出现 ```、query_cache、recent_raw_context、if query contains、bypass query 这类内容，直接丢弃。
- 本阶段不需要输出 original_excerpt；来源片段会由系统按完整句子自动提取。你只需要给出精确的 source_event_ids / source_turn_ids。
- source_event_ids / source_turn_ids 必须精确指向该候选实际依据的原文轮次/事件，只从 conversation_turns 里真实出现的 id 中选；拿不准就留空并丢弃该候选，禁止回退到全天所有 id。
- content 是“建议记忆”：必须是与原文不同的脱水和整理，通常 60 到 260 字、1 到 3 句；写清背景、已确认结论、后续要注意什么。它应该像手动 hold 的正文，而不是聊天记录转述，更不是把原文原句照抄。
- content 不要以日期或来源壳开头；不要写 "x月x日，有一条可召回的边界"、"2026-xx-xx 的聊天里确认了..."、"这是一条长期记忆"。
- 必须消解代词：user_text 里的“我”要改写成 {user_display_name} 或“她”；assistant_text 里的“我”才可指 {ai_name}。不要让来源原话里的“我”在记忆里变成 {ai_name}。
- title 必须是具体短标题，8 到 24 字，不要用“自动记忆”“每日记忆”“2026-xx-xx 自动记忆”。
- domain 必须从下面的新主域里选 1 个最精确的；实在没把握才选 general。不要输出旧的“日常/人际/数字/未分类”：
{domain_options_text}
- 只有原话本身是暗号、明确边界、承诺、昵称或高价值关系锚点时，才可在 content 末尾追加很短的 "### original"；否则不要保存原话。
- 不硬编码姓名；如果用户指的是当前用户，写作 {user_display_name}；如果 assistant/AI 指的是当前回应者，写作 {ai_name}。
- 正文优先用第三人称；### reflection 必须用 {ai_name} 第一人称，比如“我记得 / 我明白 / 我以后”。### original 是可选补充原文片段，只在原味不可替代时使用。
- 用户偏好、边界、暗号适合第三人称；{ai_name} 自己的关系锚点和 ### reflection 可以用第一人称；项目状态用中性第三人称。
- 只根据原文能证明的内容写，不编造。
- 没有候选时返回 {"candidates": []}。"""


DAILY_CHAT_MEMORY_SUMMARY_PROMPT_TEMPLATE = """你是 {ai_name} 的对话压缩器。你正在为 Ombre 自动记忆做第一步：把一段连续聊天压缩成“候选抽取材料”，不是直接写长期记忆。

请读 self_anchor_entry 校准称呼和主语，但不要复制它。{user_display_name} 的配置别名是：{user_aliases_text}。

输入是一个连续窗口里的 raw_events 还原对话。user_text 永远是 {user_display_name} 的原话，里面的“我”指 {user_display_name}；assistant_text 永远是 {ai_name} 的回复，里面的“我”指 {ai_name}。

保留：
- 已确认事实、稳定偏好、明确边界、暗号/模式切换信号
- 承诺、未完成约定、仍会影响未来执行的项目状态
- 真正有连续性价值的关系锚点
- 情感交流里的明确变化、重要事件、项目进展、后续需要关注的事
- 因果：是谁提出、后来是否确认、为什么可能值得未来记得

忽略：
- 工具调用、工具结果、系统注入、客户端状态、普通寒暄、重复调情、过程流水
- 召回测试、探针、问模型有没有记得、临时调试噪声
- 单句照顾提醒、晚安、吃药、睡觉、别熬夜、催睡或 ntfy 玩笑；这类只属于当天关系天气，不直接变长期记忆
- 未确认猜测、触发条件猜测、没有下一步的“可能是/似乎/果然没触发”
- 只靠单个称呼或气氛得出的泛泛关系总结
- 代码块、伪代码、查询规则、缓存规则、prompt 片段、内部实现片段；不要把 ```、query_cache、recent_raw_context、if query contains、bypass query 这类内容当成项目状态

输出纯 JSON：
{
  "summaries": [
    {
      "title": "短标题",
      "summary": "一段自包含摘要，写清事实、因果和是否已确认，不要写成记忆正文。",
      "signals": ["stable_preference", "project_state"],
      "source_event_ids": [101, 102],
      "source_turn_ids": [1, 2],
      "confidence": 0.72
    }
  ]
}

规则：
- 每个窗口最多输出 4 条 summary；每条围绕一个可能的长期记忆点。没有长期价值信号时返回 {"summaries": []}。
- summary 要能让下一步模型在不看完整原文时仍理解上下文，不要压成一句泛泛结论。
- summary 通常 80 到 320 字；写清背景、因果、已确认内容、未完成点。不要输出 Markdown。
- 如果信号出现在窗口开头或结尾，保留“前文可能已铺垫 / 后文可能继续确认”的边界提醒，不要把未确认因果说死。
- source_event_ids / source_turn_ids 只能使用输入里真实出现的 id；拿不准可留空。
- confidence 低于 0.5 的内容不要输出。
"""


DAILY_ACTIVITY_SUMMARY_PROMPT_TEMPLATE = """你是 {ai_name} 的当天行动摘要器。你正在为 handoff、新窗口和 dashboard 的 Recent Timeline 写一条“今天做了什么”。

输入是当天原始对话还原出的 conversation_turns。user_text 永远是 {user_display_name} 的原话，assistant_text 永远是 {ai_name} 的回复。请只根据输入能证明的内容写。

输出纯 JSON：
{
  "summary": "一句话说明今天主要推进了什么",
  "confidence": 0.72,
  "source_turn_ids": [1, 2],
  "source_event_ids": [101, 102]
}

规则：
- 这是 dashboard / handoff 用的近期事项，不是长期记忆候选，也不要输出 candidates。
- 只写今天实际讨论、推进、排查、决定、实现或整理的事；优先项目/工作/生活动作。
- 不写关系天气、情绪评价、昵称互动、普通寒暄、召回探针、模型自夸。
- summary 用一句自然中文，35 到 90 字；不要 Markdown，不要列表，不要“今天的总结是”这种壳。
- source_turn_ids / source_event_ids 只能使用输入里真实出现的 id；拿不准可留空。
"""


REFLECT_PROMPT = render_identity_template(REFLECT_PROMPT_TEMPLATE, generic_identity_names())
DIARY_MEMORY_PROMPT = render_identity_template(
    DIARY_MEMORY_PROMPT_TEMPLATE.replace("{domain_options_text}", domain_prompt_options_text()),
    generic_identity_names(),
)


AFFECT_ANCHOR_HEADER = "### affect_anchor"


REFLECTION_FALLBACK_ANCHORS = [
    {
        "chords": "Cmaj7 -> G/B -> Am9 -> F6",
        "tempo": "56bpm",
        "dynamic": "mp",
    },
    {
        "chords": "Dm9 -> G13 -> Cmaj9",
        "tempo": "64bpm",
        "dynamic": "p",
    },
    {
        "chords": "Em7 -> A7sus4 -> Dmaj9 -> Gmaj7",
        "tempo": "72bpm",
        "dynamic": "mp",
    },
    {
        "chords": "Bbmaj7 -> F/A -> Gm9 -> Csus4",
        "tempo": "60bpm",
        "dynamic": "mf",
    },
]


class ReflectionEngine:
    """LLM-backed memory enrichment and daily relationship weather."""

    def __init__(self, config: dict):
        self.config = config
        self.identity = identity_names(config)
        gateway_cfg = config.get("gateway", {}) if isinstance(
            config.get("gateway", {}), dict) else {}
        self.prompt_plan_mirror = PromptPlanMirrorStore(
            gateway_cfg.get("prompt_plan_mirror_path")
            or os.path.join(config["buckets_dir"],
                            "prompt_plan_mirror.sqlite3"))
        cfg = config.get("reflection", {}) if isinstance(config.get("reflection", {}), dict) else {}
        emb_cfg = config.get("embedding", {}) if isinstance(config.get("embedding", {}), dict) else {}
        persona_cfg = config.get("persona", {}) if isinstance(config.get("persona", {}), dict) else {}
        dehy_cfg = config.get("dehydration", {}) if isinstance(config.get("dehydration", {}), dict) else {}

        self.enabled = bool(cfg.get("enabled", True))
        self.auto_enabled = bool(cfg.get("auto_enabled", True))
        self.daily_enabled = bool(cfg.get("daily_enabled", True))
        self.enrich_on_write = bool(cfg.get("enrich_on_write", True))
        self.memory_affect_anchor_enabled = bool(cfg.get("memory_affect_anchor_enabled", True))
        self.relationship_weather_affect_anchor_enabled = bool(
            cfg.get("relationship_weather_affect_anchor_enabled", True)
        )
        self.identity_role_edge_config = self._load_identity_role_edge_config(
            cfg.get("identity_role_edges")
        )
        legacy_candidate_model = str(cfg.get("daily_chat_memory_candidate_model") or "").strip()
        self.base_url = (
            cfg.get("base_url")
            or emb_cfg.get("base_url")
            or persona_cfg.get("base_url")
            or dehy_cfg.get("base_url", "")
        )
        self.model = cfg.get("model") or legacy_candidate_model or persona_cfg.get("model") or dehy_cfg.get("model", "deepseek-chat")
        self.api_key = (
            os.environ.get("OMBRE_REFLECTION_API_KEY", "")
            or cfg.get("api_key", "")
            or os.environ.get("OMBRE_EMBEDDING_API_KEY", "")
            or emb_cfg.get("api_key", "")
            or persona_cfg.get("api_key", "")
            or os.environ.get("OMBRE_PERSONA_API_KEY", "")
            or dehy_cfg.get("api_key", "")
        )
        self.thinking_mode = self._normalize_thinking_mode(
            cfg.get("thinking_mode")
            or persona_cfg.get("thinking_mode")
            or ""
        )
        self.temperature = float(cfg.get("temperature", 0.1))
        self.max_tokens = int(cfg.get("max_tokens", 700))
        self.timezone_name = str(cfg.get("timezone") or "Asia/Shanghai")
        try:
            self.tz = ZoneInfo(self.timezone_name)
        except Exception:
            self.tz = ZoneInfo("Asia/Shanghai")
        self.daily_hour = int(cfg.get("daily_hour", 4))
        self.daily_min_memory_items = max(
            0,
            int(cfg.get("daily_min_memory_items", DEFAULT_DAILY_REFLECTION_MIN_BUCKETS)),
        )
        self.daily_conversation_turn_limit = max(
            0,
            min(80, int(cfg.get("daily_conversation_turn_limit", 12))),
        )
        self.persona_events_limit = max(0, int(cfg.get("persona_events_limit", 12)))
        self.persona_events_scan_limit = max(
            self.persona_events_limit,
            int(cfg.get("persona_events_scan_limit", 80)),
        )
        self.weekly_enabled = bool(cfg.get("weekly_enabled", False))
        self.weekly_day = int(cfg.get("weekly_day", 0))
        self.weekly_hour = int(cfg.get("weekly_hour", self.daily_hour))
        self.check_interval_minutes = max(5, int(cfg.get("check_interval_minutes", 60)))
        self.edge_min_confidence = float(cfg.get("edge_min_confidence", 0.55))
        self.diary_mcp_url = str(cfg.get("diary_mcp_url") or "").strip()
        self.diary_mcp_token_env = str(cfg.get("diary_mcp_token_env") or "").strip()
        self.diary_memory_extract_enabled = bool(cfg.get("diary_memory_extract_enabled", True))
        self.diary_memory_extract_max_per_day = max(0, int(cfg.get("diary_memory_extract_max_per_day", 1)))
        self.diary_memory_extract_min_confidence = float(cfg.get("diary_memory_extract_min_confidence", 0.68))
        self.daily_chat_memory_mode = self._normalize_daily_chat_memory_mode(
            cfg.get("daily_chat_memory_mode", "review")
        )
        self.daily_chat_memory_hour = max(0, min(23, int(cfg.get("daily_chat_memory_hour", 0))))
        self.daily_chat_memory_turn_limit = max(0, min(10000, int(cfg.get("daily_chat_memory_turn_limit", 0))))
        self.daily_chat_memory_max_per_day = max(0, min(10, int(cfg.get("daily_chat_memory_max_per_day", 10))))
        self.daily_chat_memory_review_max_per_day = max(
            0,
            min(30, int(cfg.get("daily_chat_memory_review_max_per_day", 10))),
        )
        self.daily_chat_memory_min_confidence = float(cfg.get("daily_chat_memory_min_confidence", 0.68))
        self.daily_chat_memory_review_min_confidence = float(
            cfg.get("daily_chat_memory_review_min_confidence", 0.55)
        )
        self.daily_chat_memory_summary_enabled = bool(cfg.get("daily_chat_memory_summary_enabled", True))
        self.daily_chat_memory_summary_window_turns = max(
            1,
            min(200, int(cfg.get("daily_chat_memory_summary_window_turns", 14))),
        )
        self.daily_chat_memory_summary_stride_turns = max(
            1,
            min(
                self.daily_chat_memory_summary_window_turns,
                int(cfg.get("daily_chat_memory_summary_stride_turns", 7)),
            ),
        )
        self.daily_chat_memory_api_key_env = str(
            cfg.get("daily_chat_memory_api_key_env")
            or cfg.get("daily_chat_memory_summary_api_key_env")
            or ""
        ).strip()
        self.daily_chat_memory_api_key = (
            os.environ.get(self.daily_chat_memory_api_key_env, "")
            if self.daily_chat_memory_api_key_env
            else ""
        ) or str(
            cfg.get("daily_chat_memory_api_key")
            or cfg.get("daily_chat_memory_summary_api_key")
            or ""
        ).strip()
        self.daily_chat_memory_base_url = str(
            cfg.get("daily_chat_memory_base_url")
            or cfg.get("daily_chat_memory_summary_base_url")
            or ""
        ).strip().rstrip("/")
        self.daily_chat_memory_timeout_seconds = max(
            30.0,
            min(300.0, float(cfg.get("daily_chat_memory_timeout_seconds", 180.0))),
        )
        self.daily_chat_memory_summary_model = str(
            cfg.get("daily_chat_memory_summary_model") or ""
        ).strip()
        self.daily_chat_memory_summary_max_tokens = max(
            300,
            min(4000, int(cfg.get("daily_chat_memory_summary_max_tokens", 2200))),
        )
        self.daily_chat_memory_candidate_model = str(
            cfg.get("daily_chat_memory_candidate_model")
            or self.daily_chat_memory_summary_model
            or ""
        ).strip()
        self.daily_chat_memory_candidate_max_tokens = max(
            300,
            min(4000, int(cfg.get("daily_chat_memory_candidate_max_tokens", 3200))),
        )
        self.daily_activity_summary_enabled = bool(cfg.get("daily_activity_summary_enabled", True))
        self.daily_activity_summary_turn_limit = max(
            0,
            min(
                10000,
                int(
                    cfg.get(
                        "daily_activity_summary_turn_limit",
                        cfg.get("daily_chat_memory_turn_limit", 0),
                    )
                ),
            ),
        )
        self.daily_activity_summary_max_tokens = max(
            80,
            min(1000, int(cfg.get("daily_activity_summary_max_tokens", 320))),
        )
        self.dehydration_base_url = str(dehy_cfg.get("base_url") or "").strip().rstrip("/")
        self.dehydration_model = str(dehy_cfg.get("model") or "").strip()
        self.dehydration_api_key = str(dehy_cfg.get("api_key") or os.environ.get("OMBRE_API_KEY", "")).strip()
        state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
            "state",
        )
        self.daily_chat_memory_pending_path = str(
            cfg.get("daily_chat_memory_pending_path")
            or os.path.join(state_dir, "daily_chat_memory_candidates.json")
        )
        self.daily_chat_memory_requests_path = str(
            cfg.get("daily_chat_memory_requests_path")
            or os.path.join(
                os.path.dirname(self.daily_chat_memory_pending_path),
                "daily_chat_memory_requests.json",
            )
        )
        # Bounded original conversation turns replayed into the final candidate
        # extraction stage so candidates are selected/verified against the source
        # text itself, not only against window summaries (two-level compression).
        self.daily_chat_memory_extraction_turns = max(
            1,
            min(200, int(cfg.get("daily_chat_memory_extraction_turns", 40))),
        )
        # V4 full-day windowed extraction: every valid window of the day is
        # actually checked (beginning / middle / end), bounded by per-window and
        # per-run cost controls. Summaries only assist location, never replace a
        # window check.
        self.daily_chat_memory_window_turns = max(
            4,
            min(100, int(cfg.get("daily_chat_memory_window_turns", 24))),
        )
        self.daily_chat_memory_window_stride_turns = max(
            1,
            min(
                self.daily_chat_memory_window_turns,
                int(cfg.get("daily_chat_memory_window_stride_turns", 12)),
            ),
        )
        self.daily_chat_memory_max_windows_per_run = max(
            1,
            min(60, int(cfg.get("daily_chat_memory_max_windows_per_run", 10))),
        )
        self.daily_chat_memory_window_max_input_chars = max(
            2000,
            min(60000, int(cfg.get("daily_chat_memory_window_max_input_chars", 12000))),
        )
        # Owner confirm/reject sliding-window rate limit (per process).
        self.daily_chat_memory_confirm_rate_limit_per_minute = max(
            1,
            min(300, int(cfg.get("daily_chat_memory_confirm_rate_limit_per_minute", 30))),
        )
        # Confidence used by heuristic candidates (durable-signal turn matching only).
        self.daily_chat_memory_heuristic_confidence = self._clamp(
            cfg.get("daily_chat_memory_heuristic_confidence", 0.70)
        )
        self.daily_chat_memory_run_audit_path = str(
            cfg.get("daily_chat_memory_run_audit_path")
            or os.path.join(
                os.path.dirname(self.daily_chat_memory_pending_path),
                "daily_chat_memory_run_audit.json",
            )
        )
        self._confirm_rate_window: list[float] = []
        self._daily_chat_memory_run_lock: bool = False

        self.client = None
        if self.enabled and self.api_key and self.base_url:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=45.0)
        self.daily_chat_memory_client = None
        if (
            self.enabled
            and self.daily_chat_memory_api_key
            and self.daily_chat_memory_base_url
            and (self.daily_chat_memory_summary_model or self.daily_chat_memory_candidate_model)
        ):
            self.daily_chat_memory_client = AsyncOpenAI(
                api_key=self.daily_chat_memory_api_key,
                base_url=self.daily_chat_memory_base_url,
                timeout=self.daily_chat_memory_timeout_seconds,
            )
        self.dehydration_client = None
        if self.enabled and self.dehydration_api_key and self.dehydration_base_url and self.dehydration_model:
            self.dehydration_client = AsyncOpenAI(
                api_key=self.dehydration_api_key,
                base_url=self.dehydration_base_url,
                timeout=45.0,
            )
        self.daily_activity_summary_dehydration_client = self.dehydration_client

    def _load_daily_chat_memory_payload(self) -> dict:
        try:
            with open(self.daily_chat_memory_pending_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return {"items": [], "cursor": {}}
        except Exception as exc:
            logger.warning("Daily chat memory pending read failed: %s", exc)
            return {"items": [], "cursor": {}}
        if isinstance(data, dict):
            items = data.get("items")
            cursor = data.get("cursor") if isinstance(data.get("cursor"), dict) else {}
            return {
                "items": [item for item in (items or []) if isinstance(item, dict)],
                "cursor": cursor,
            }
        return {"items": [item for item in (data or []) if isinstance(item, dict)], "cursor": {}}

    def _load_daily_chat_memory_cursor(self) -> dict:
        cursor = self._load_daily_chat_memory_payload().get("cursor")
        return cursor if isinstance(cursor, dict) else {}

    @staticmethod
    def _daily_chat_memory_cursor_key(profile_id: str) -> str:
        return str(profile_id or "default").strip() or "default"

    def _daily_chat_memory_last_raw_event_id(self, profile_id: str) -> int:
        cursor = self._load_daily_chat_memory_cursor()
        raw_events = cursor.get("raw_events") if isinstance(cursor.get("raw_events"), dict) else {}
        entry = raw_events.get(self._daily_chat_memory_cursor_key(profile_id))
        if not isinstance(entry, dict):
            return 0
        try:
            return max(0, int(entry.get("last_raw_event_id") or 0))
        except (TypeError, ValueError):
            return 0

    def _update_daily_chat_memory_raw_cursor(self, profile_id: str, raw_event_id: int, key: str) -> bool:
        try:
            safe_id = max(0, int(raw_event_id or 0))
        except (TypeError, ValueError):
            safe_id = 0
        if safe_id <= 0:
            return False
        payload = self._load_daily_chat_memory_payload()
        cursor = payload.get("cursor") if isinstance(payload.get("cursor"), dict) else {}
        raw_events = cursor.get("raw_events") if isinstance(cursor.get("raw_events"), dict) else {}
        cursor_key = self._daily_chat_memory_cursor_key(profile_id)
        previous = raw_events.get(cursor_key) if isinstance(raw_events.get(cursor_key), dict) else {}
        try:
            previous_id = max(0, int(previous.get("last_raw_event_id") or 0))
        except (TypeError, ValueError):
            previous_id = 0
        if safe_id <= previous_id:
            return False
        raw_events[cursor_key] = {
            "last_raw_event_id": safe_id,
            "date": key,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        cursor["raw_events"] = raw_events
        self._save_daily_chat_memory_pending(payload.get("items") or [], cursor=cursor)
        return True

    def _resolve_background_prompt(self, source_id: str, scope: str,
                                   live_body: str) -> str:
        try:
            resolved, _meta = self.prompt_plan_mirror.resolve_text(
                source_id=source_id, scope=scope, live_body=live_body,
                identity_id=str(
                    self.config.get("persona", {}).get(
                        "canonical_session_id") or "jiajia-main"),
                conversation_id="")
            return resolved
        except Exception:
            logger.exception(
                "Prompt Composer reflection projection failed | scope=%s source=%s",
                scope, source_id)
            return str(live_body or "")

    def _reflect_prompt(self) -> str:
        return self._resolve_background_prompt(
            "ombre.reflection_prompt", "memory.reflection",
            render_identity_template(REFLECT_PROMPT_TEMPLATE, self.identity))

    def _diary_memory_prompt(self) -> str:
        prompt = DIARY_MEMORY_PROMPT_TEMPLATE.replace("{domain_options_text}", domain_prompt_options_text())
        return self._resolve_background_prompt(
            "ombre.diary_memory_prompt", "memory.reflection",
            render_identity_template(prompt, self.identity))

    def _daily_chat_memory_prompt(self, max_candidates: int | None = None) -> str:
        prompt = DAILY_CHAT_MEMORY_PROMPT_TEMPLATE.replace(
            "{max_candidates}",
            str(max(1, int(max_candidates or self.daily_chat_memory_max_per_day or 1))),
        ).replace(
            "{domain_options_text}",
            domain_prompt_options_text(),
        )
        return self._resolve_background_prompt(
            "ombre.daily_chat_memory_prompt", "memory.daily_chat_review",
            render_identity_template(prompt, self.identity))

    def _daily_chat_memory_summary_prompt(self) -> str:
        return self._resolve_background_prompt(
            "ombre.daily_chat_summary_prompt", "memory.daily_chat_review",
            render_identity_template(
                DAILY_CHAT_MEMORY_SUMMARY_PROMPT_TEMPLATE, self.identity))

    def _daily_activity_summary_prompt(self) -> str:
        return self._resolve_background_prompt(
            "ombre.daily_activity_summary_prompt", "memory.daily_chat_review",
            render_identity_template(
                DAILY_ACTIVITY_SUMMARY_PROMPT_TEMPLATE, self.identity))

    async def enrich_bucket(
        self,
        bucket_id: str,
        bucket_mgr,
        edge_store: MemoryEdgeStore,
        embedding_engine=None,
        force: bool = False,
    ) -> dict:
        if not self.enabled or (not self.enrich_on_write and not force):
            return {"status": "disabled", "id": bucket_id}
        bucket = await bucket_mgr.get(bucket_id)
        if not bucket:
            return {"status": "missing", "id": bucket_id}
        meta = bucket.get("metadata", {})
        if meta.get("type") == "feel":
            return {"status": "skipped_feel", "id": bucket_id}

        candidates = await self._candidate_buckets(bucket, bucket_mgr, embedding_engine)
        if self.client:
            result = await self._api_classify(bucket, candidates)
        else:
            result = self._heuristic_classify(bucket)

        tags = self._string_list(result.get("tags"), limit=8)
        confidence = self._clamp(result.get("confidence", 0.55))
        importance = self._int_between(result.get("importance"), meta.get("importance", 5))
        if self._has_favorite_tag(tags) and not self._has_favorite_reason(bucket.get("content", "")):
            tags = [tag for tag in tags if tag != "haven_favorite" and not str(tag).startswith("flavor_")]
            logger.warning(
                "Rejected favorite tags without reason during enrich / enrich 拒绝缺少喜欢原因的 favorite 标签: %s",
                bucket_id,
            )
        merged_tags = list(dict.fromkeys(list(meta.get("tags", [])) + tags))
        updates: dict[str, Any] = {}
        if tags:
            if merged_tags != meta.get("tags", []):
                updates["tags"] = merged_tags[:24]
        if importance > int(meta.get("importance", 5)):
            updates["importance"] = importance
        if confidence > float(meta.get("confidence", 0.0) or 0.0):
            updates["confidence"] = confidence

        anchor = self._normalize_affect_anchor(result.get("affect_anchor"))
        if self._should_add_affect_anchor(bucket, merged_tags, importance, confidence, result):
            if anchor:
                anchored_content = self._append_affect_anchor(bucket.get("content", ""), anchor)
                if anchored_content != bucket.get("content", ""):
                    updates["content"] = anchored_content

        if updates:
            updates["last_active"] = meta.get("last_active") or meta.get("created")
            await bucket_mgr.update(bucket_id, **updates)
            if "content" in updates and embedding_engine and getattr(embedding_engine, "enabled", False):
                try:
                    updated_bucket = await bucket_mgr.get(bucket_id)
                    if updated_bucket:
                        await embedding_engine.generate_and_store(
                            bucket_id,
                            bucket_text_for_embedding(updated_bucket),
                        )
                except Exception as exc:
                    logger.warning("Memory affect anchor embedding refresh failed for %s: %s", bucket_id, exc)

        edges = self._edges_from_classification(bucket, candidates, result, confidence)
        saved_edges = edge_store.add_edges(edges[:3])
        return {
            "status": "ok",
            "id": bucket_id,
            "tags": tags,
            "confidence": confidence,
            "edges": len(saved_edges),
        }

    async def backfill_edges_for_bucket(
        self,
        bucket_id: str,
        bucket_mgr,
        edge_store: MemoryEdgeStore,
        embedding_engine=None,
        *,
        dry_run: bool = False,
    ) -> dict:
        if not self.enabled:
            return {"status": "disabled", "id": bucket_id, "edges": 0, "proposed_edges": 0}
        bucket = await bucket_mgr.get(bucket_id)
        if not bucket:
            return {"status": "missing", "id": bucket_id, "edges": 0, "proposed_edges": 0}
        meta = bucket.get("metadata", {})
        if meta.get("type") == "feel" or meta.get("protected"):
            return {"status": "skipped", "reason": "not_edge_backfillable", "id": bucket_id, "edges": 0, "proposed_edges": 0}

        candidates = await self._candidate_buckets(bucket, bucket_mgr, embedding_engine)
        if self.client:
            result = await self._api_classify(bucket, candidates)
        else:
            result = self._heuristic_classify(bucket)
        confidence = self._clamp(result.get("confidence", meta.get("confidence", 0.55)))
        proposed_edges = self._edges_from_classification(bucket, candidates, result, confidence)[:3]
        saved_edges = [] if dry_run else edge_store.add_edges(proposed_edges)
        return {
            "status": "ok",
            "id": bucket_id,
            "candidate_count": len(candidates),
            "proposed_edges": len(proposed_edges),
            "edges": len(saved_edges),
            "dry_run": bool(dry_run),
            "edge_records": proposed_edges if dry_run else saved_edges,
        }

    async def reflect(
        self,
        period: str,
        bucket_mgr,
        persona_engine=None,
        embedding_engine=None,
        force: bool = False,
        now: datetime | None = None,
        conversation_turn_store=None,
        daily_chat_memory_candidates: list[dict] | None = None,
    ) -> dict:
        if not self.enabled:
            return {
                "status": "disabled",
                "period": period,
                "diary": {"found": False},
                "diary_memory": {"status": "not_applicable", "reason": "reflection_disabled"},
            }
        period = self._normalize_period(period)
        if period == "daily" and not self.daily_enabled:
            return {
                "status": "skipped",
                "reason": "daily_disabled",
                "period": period,
                "diary": {"found": False},
                "diary_memory": {"status": "not_applicable", "reason": "daily_disabled"},
            }
        if period == "weekly" and not self.weekly_enabled:
            return {
                "status": "skipped",
                "reason": "weekly_disabled",
                "period": period,
                "diary": {"found": False},
                "diary_memory": {"status": "not_applicable", "reason": "weekly_disabled"},
            }
        now_local = self._local_now(now)
        key = self._period_key(period, now_local)
        bucket_id = f"reflection_{period}_{key}"
        existing = await bucket_mgr.get(bucket_id)
        if existing and not force:
            return {
                "status": "exists",
                "period": period,
                "id": bucket_id,
                "diary": {"found": False},
                "diary_memory": {"status": "skipped", "reason": "reflection_exists"},
            }

        materials = await self._reflection_materials(
            period,
            now_local,
            bucket_mgr,
            persona_engine,
            conversation_turn_store=conversation_turn_store,
            daily_chat_memory_candidates=daily_chat_memory_candidates,
        )
        min_daily_buckets = self.daily_min_memory_items
        if (
            period == "daily"
            and min_daily_buckets > 0
            and len(materials["buckets"]) < min_daily_buckets
            and not materials.get("daily_chat_memories")
        ):
            diary_memory = await self._maybe_extract_diary_memory(
                period,
                key,
                now_local,
                materials,
                bucket_mgr,
                embedding_engine,
            )
            return {
                "status": "skipped",
                "reason": "insufficient_daily_memory",
                "period": period,
                "id": bucket_id,
                "date": key,
                "diary": {
                    "found": bool(materials.get("diary")),
                    "diary_id": materials.get("diary", {}).get("id") if materials.get("diary") else None,
                },
                "diary_memory": diary_memory,
                "materials": {
                    "buckets": len(materials["buckets"]),
                    "daily_impressions": len(materials["daily_impressions"]),
                    "daily_chat_memories": len(materials["daily_chat_memories"]),
                    "persona_events": len(materials["persona_events"]),
                    "conversation_turns": len(materials["conversation_turns"]),
                    "commitments": len(materials["commitments"]),
                    "min_buckets": min_daily_buckets,
                },
            }
        if (
            not materials["buckets"]
            and not materials["daily_impressions"]
            and not materials["daily_chat_memories"]
            and not materials["persona_events"]
            and not materials["conversation_turns"]
            and not materials["diary"]
            and not force
        ):
            return {
                "status": "empty",
                "period": period,
                "id": bucket_id,
                "diary": {"found": False},
                "diary_memory": {"status": "skipped", "reason": "no_materials"},
            }

        reflect_client, _, _ = self._reflect_model_client()
        if reflect_client:
            result = await self._api_reflect(period, key, materials)
        else:
            result = self._fallback_reflection(period, key, materials)

        title = str(result.get("title") or f"{key} {'日印象' if period == 'daily' else '周印象'}")[:40]
        content = str(result.get("content") or "").strip()
        if not content:
            content = self._fallback_reflection(period, key, materials)["content"]
        if self.relationship_weather_affect_anchor_enabled:
            content = self._append_affect_anchor(
                content,
                self._normalize_affect_anchor(result.get("affect_anchor"))
                or self._fallback_reflection(period, key, materials).get("affect_anchor", {}),
            )
        tags = list(
            dict.fromkeys(
                [
                    "relationship_weather",
                    f"{period}_impression",
                    *self._string_list(result.get("tags"), limit=8),
                ]
            )
        )
        valence = self._clamp(result.get("valence", 0.55))
        arousal = self._clamp(result.get("arousal", 0.32))
        confidence = self._clamp(result.get("confidence", 0.65))
        created = now_local.isoformat(timespec="seconds")
        source_bucket_ids = [
            str(item.get("id") or "")
            for item in materials.get("buckets", []) + materials.get("daily_impressions", [])
            if item.get("id")
        ]
        source_persona_event_ids = [
            int(event.get("id"))
            for event in materials.get("persona_events", [])
            if event.get("id")
        ]
        source_conversation_turn_ids = [
            int(turn.get("id"))
            for turn in materials.get("conversation_turns", [])
            if turn.get("id")
        ]
        source_metadata = {
            "source_bucket_ids": source_bucket_ids[:40],
            "source_persona_event_ids": source_persona_event_ids[:40],
            "source_conversation_turn_ids": source_conversation_turn_ids[:80],
            "source_daily_chat_memory_candidate_ids": [
                str(item.get("id") or "")
                for item in materials.get("daily_chat_memories", [])
                if item.get("id")
            ][:40],
        }

        if existing:
            await bucket_mgr.update(
                bucket_id,
                content=content,
                tags=tags,
                importance=6 if period == "daily" else 7,
                domain=["自省", "恋爱"],
                valence=valence,
                arousal=arousal,
                name=title,
                confidence=confidence,
                period=period,
                date=key,
                source="reflection",
                **source_metadata,
                last_active=existing.get("metadata", {}).get("last_active") or existing.get("metadata", {}).get("created"),
            )
            status = "updated"
        else:
            await bucket_mgr.create(
                bucket_id=bucket_id,
                content=content,
                tags=tags,
                importance=6 if period == "daily" else 7,
                domain=["自省", "恋爱"],
                valence=valence,
                arousal=arousal,
                bucket_type="feel",
                name=title,
                source="reflection",
                created=created,
                last_active=created,
                updated_at=created,
                confidence=confidence,
                period=period,
                date=key,
                extra_metadata=source_metadata,
            )
            status = "created"

        if embedding_engine and getattr(embedding_engine, "enabled", False):
            try:
                bucket = await bucket_mgr.get(bucket_id)
                if bucket:
                    await embedding_engine.generate_and_store(
                        bucket_id,
                        bucket_text_for_embedding(bucket),
                    )
            except Exception as exc:
                logger.warning("Reflection embedding failed for %s: %s", bucket_id, exc)

        diary_memory = await self._maybe_extract_diary_memory(
            period,
            key,
            now_local,
            materials,
            bucket_mgr,
            embedding_engine,
        )

        return {
            "status": status,
            "period": period,
            "id": bucket_id,
            "date": key,
            "diary": {
                "found": bool(materials.get("diary")),
                "diary_id": materials.get("diary", {}).get("id") if materials.get("diary") else None,
            },
            "diary_memory": diary_memory,
            "daily_impression": {
                "id": bucket_id,
                "content": content,
                "confidence": confidence,
                "date": key,
            },
            "materials": {
                "buckets": len(materials["buckets"]),
                "daily_impressions": len(materials["daily_impressions"]),
                "daily_chat_memories": len(materials["daily_chat_memories"]),
                "persona_events": len(materials["persona_events"]),
                "conversation_turns": len(materials["conversation_turns"]),
                "commitments": len(materials["commitments"]),
                "min_buckets": min_daily_buckets,
            },
        }

    async def run_due(
        self,
        bucket_mgr,
        persona_engine=None,
        embedding_engine=None,
        conversation_turn_store=None,
        raw_event_store=None,
    ) -> list[dict]:
        if not self.enabled or not self.auto_enabled:
            return []
        now_local = self._local_now()
        results = []
        chat_candidates: list[dict] = []
        if self.daily_chat_memory_mode != "off" and now_local.hour >= self.daily_chat_memory_hour:
            chat_date = (now_local - timedelta(days=1)).date()
            chat_target = datetime.combine(chat_date, time.max, tzinfo=self.tz)
            chat_result = await self.run_daily_chat_memory(
                bucket_mgr,
                conversation_turn_store=conversation_turn_store,
                raw_event_store=raw_event_store,
                persona_engine=persona_engine,
                embedding_engine=embedding_engine,
                now=chat_target,
            )
            if chat_result.get("status") not in {"disabled", "skipped"}:
                results.append(chat_result)
            chat_candidates = [
                item for item in (chat_result.get("candidates") or []) if isinstance(item, dict)
            ]
        if self.daily_enabled and now_local.hour >= self.daily_hour:
            daily_date = (now_local - timedelta(days=1)).date()
            daily_target = datetime.combine(daily_date, time.max, tzinfo=self.tz)
            results.append(
                await self.reflect(
                    "daily",
                    bucket_mgr,
                    persona_engine,
                    embedding_engine,
                    force=False,
                    now=daily_target,
                    conversation_turn_store=conversation_turn_store,
                    daily_chat_memory_candidates=chat_candidates,
                )
            )
        if self.weekly_enabled and now_local.weekday() == self.weekly_day and now_local.hour >= self.weekly_hour:
            weekly_target = now_local - timedelta(days=1)
            results.append(
                await self.reflect("weekly", bucket_mgr, persona_engine, embedding_engine, force=False, now=weekly_target)
            )
        return results

    async def _candidate_buckets(self, bucket: dict, bucket_mgr, embedding_engine=None, limit: int | None = None) -> list[dict]:
        cfg = self.config.get("reflection", {}) if isinstance(self.config.get("reflection", {}), dict) else {}
        limit = max(1, int(limit or cfg.get("candidate_limit", 18)))
        recent_limit = max(1, int(cfg.get("candidate_recent_limit", 8)))
        semantic_limit = max(0, int(cfg.get("candidate_semantic_limit", 6)))
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=True)
        except Exception:
            all_buckets = []
        source_id = bucket.get("id")
        bucket_map = {item.get("id"): item for item in all_buckets if item.get("id")}
        candidates: list[dict] = []
        seen = {source_id}

        def eligible(item: dict | None) -> bool:
            if not item or item.get("id") in seen:
                return False
            meta = item.get("metadata", {})
            return meta.get("type") != "feel"

        def add_candidate(item: dict | None) -> bool:
            if not eligible(item):
                return False
            seen.add(item.get("id"))
            candidates.append(item)
            return len(candidates) >= limit

        recent_items = sorted(
            all_buckets,
            key=lambda b: b.get("metadata", {}).get("created", ""),
            reverse=True,
        )
        recent_added = 0
        for item in recent_items:
            before_count = len(candidates)
            if add_candidate(item):
                return candidates
            if len(candidates) > before_count:
                recent_added += 1
            if recent_added >= recent_limit:
                break

        if embedding_engine and getattr(embedding_engine, "enabled", False) and semantic_limit > 0:
            query = " ".join(
                part
                for part in [
                    str(bucket.get("metadata", {}).get("name") or ""),
                    strip_wikilinks(bucket.get("content", "")),
                ]
                if part
            )
            try:
                similar = await embedding_engine.search_similar(query, top_k=max(semantic_limit * 3, 12))
            except Exception as exc:
                logger.debug("Reflection semantic candidate lookup failed: %s", exc)
                similar = []
            added = 0
            for candidate_id, _score in similar:
                before_count = len(candidates)
                if add_candidate(bucket_map.get(candidate_id)):
                    return candidates
                if len(candidates) > before_count:
                    added += 1
                if added >= semantic_limit:
                    break

        source_meta = bucket.get("metadata", {})
        source_tags = {str(tag) for tag in source_meta.get("tags", [])}
        source_domains = {str(domain) for domain in source_meta.get("domain", [])}
        related_by_shape = []
        commitments = []
        anchors = []
        for item in all_buckets:
            if not eligible(item):
                continue
            meta = item.get("metadata", {})
            tags = {str(tag) for tag in meta.get("tags", [])}
            domains = {str(domain) for domain in meta.get("domain", [])}
            if tags & {"commitment", "todo", "wish"} and not meta.get("resolved"):
                commitments.append(item)
            if meta.get("pinned") or meta.get("protected") or meta.get("anchor"):
                anchors.append(item)
            if (source_tags and tags & source_tags) or (source_domains and domains & source_domains):
                related_by_shape.append(item)

        def sort_key(item: dict) -> tuple[int, str]:
            meta = item.get("metadata", {})
            return int(meta.get("importance", 5)), str(meta.get("created", ""))

        for group in (related_by_shape, commitments, anchors):
            for item in sorted(group, key=sort_key, reverse=True):
                if add_candidate(item):
                    return candidates
        return candidates

    async def _api_classify(self, bucket: dict, candidates: list[dict]) -> dict:
        payload = {
            "new_memory": self._memory_payload(bucket, content_limit=1200),
            "candidate_memories": [self._memory_payload(item, content_limit=360) for item in candidates],
        }
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._resolve_background_prompt(
                    "ombre.memory_classify_prompt", "memory.reflection",
                    CLASSIFY_PROMPT)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            **self._completion_options(max_tokens=self.max_tokens, temperature=self.temperature),
        )
        raw = response.choices[0].message.content if response.choices else ""
        parsed = self._parse_json_object(raw or "")
        return parsed or self._heuristic_classify(bucket)

    def _edges_from_classification(
        self,
        bucket: dict,
        candidates: list[dict],
        result: dict,
        default_confidence: float,
    ) -> list[dict]:
        bucket_id = str(bucket.get("id") or "").strip()
        if not bucket_id:
            return []
        candidate_ids = {item["id"] for item in candidates if item.get("id")}
        raw_edges = result.get("edges", [])
        if not isinstance(raw_edges, list):
            raw_edges = []
        edges = []
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue
            target = str(edge.get("target_memory_id") or edge.get("target") or "").strip()
            if target not in candidate_ids:
                continue
            relation_type = str(edge.get("relation_type") or "relates_to").strip()
            if relation_type not in RELATION_TYPES:
                relation_type = "relates_to"
            source = bucket_id
            edge_target = target
            if relation_type in {"context_of", "precedes"}:
                source = target
                edge_target = bucket_id
            edges.append(
                {
                    "source": source,
                    "target": edge_target,
                    "relation_type": relation_type,
                    "confidence": self._clamp(edge.get("confidence", default_confidence)),
                    "reason": str(edge.get("reason") or "").strip(),
                }
            )
        edges.extend(self._identity_role_edges(bucket, candidates))
        return self._dedupe_proposed_edges(edges)

    def _identity_role_edges(self, bucket: dict, candidates: list[dict]) -> list[dict]:
        if not self.identity_role_edge_config["enabled"]:
            return []
        source_id = str(bucket.get("id") or "").strip()
        source_terms = self._identity_role_terms(bucket)
        if not source_id or not self._identity_role_edge_eligible(source_terms):
            return []

        edges = []
        for candidate in candidates:
            target_id = str(candidate.get("id") or "").strip()
            if not target_id or target_id == source_id:
                continue
            target_terms = self._identity_role_terms(candidate)
            if not self._identity_role_edge_eligible(target_terms):
                continue
            common = sorted(source_terms & target_terms)
            if len(common) < 2:
                continue
            if not self._identity_role_pair_is_specific(source_terms, target_terms):
                continue
            edges.append(
                self._identity_role_edge_for_pair(
                    source_id,
                    source_terms,
                    target_id,
                    target_terms,
                    common,
                )
            )
        edges.sort(key=lambda edge: (float(edge.get("confidence", 0.0)), edge.get("relation_type", "")), reverse=True)
        return edges[:3]

    def _identity_role_edge_for_pair(
        self,
        source_id: str,
        source_terms: set[str],
        target_id: str,
        target_terms: set[str],
        common: list[str],
    ) -> dict:
        detail_terms = self.identity_role_edge_config["detail_terms"]
        context_terms = self.identity_role_edge_config["context_terms"]
        relationship_terms = self.identity_role_edge_config["relationship_terms"]
        source_is_detail = bool(source_terms & detail_terms)
        target_is_detail = bool(target_terms & detail_terms)
        source_is_context = bool(source_terms & context_terms)
        target_is_context = bool(target_terms & context_terms)
        source_is_relationship = bool(source_terms & relationship_terms)
        target_is_relationship = bool(target_terms & relationship_terms)

        if source_is_detail and target_is_context:
            edge_source, edge_target = target_id, source_id
            relation_type = "context_of"
            confidence = 0.9
            reason = "角色与称呼记忆是具体身份组合的语义前情"
        elif source_is_context and target_is_detail:
            edge_source, edge_target = source_id, target_id
            relation_type = "context_of"
            confidence = 0.9
            reason = "角色与称呼记忆是具体身份组合的语义前情"
        elif source_is_detail and target_is_relationship:
            edge_source, edge_target = source_id, target_id
            relation_type = "supports"
            confidence = 0.84
            reason = "具体身份组合支持亲密关系与信任模式"
        elif target_is_detail and source_is_relationship:
            edge_source, edge_target = target_id, source_id
            relation_type = "supports"
            confidence = 0.84
            reason = "具体身份组合支持亲密关系与信任模式"
        else:
            edge_source, edge_target = source_id, target_id
            relation_type = "supports"
            confidence = 0.78
            reason = "共享亲密身份与称呼锚点"

        return {
            "source": edge_source,
            "target": edge_target,
            "relation_type": relation_type,
            "confidence": confidence,
            "reason": f"{reason}: {', '.join(common[:5])}",
        }

    def _identity_role_terms(self, bucket: dict) -> set[str]:
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        haystack = " ".join(
            [
                str(meta.get("name") or ""),
                " ".join(str(tag) for tag in meta.get("tags", []) or []),
                " ".join(str(domain) for domain in meta.get("domain", []) or []),
                strip_wikilinks(str(bucket.get("content") or "")),
            ]
        ).lower()
        terms = set()
        for canonical, aliases in self.identity_role_edge_config["aliases"].items():
            if any(str(alias).lower() in haystack for alias in aliases):
                terms.add(canonical)
        return terms

    def _identity_role_edge_eligible(self, terms: set[str]) -> bool:
        if len(terms) < 2:
            return False
        detail_terms = self.identity_role_edge_config["detail_terms"]
        context_terms = self.identity_role_edge_config["context_terms"]
        relationship_terms = self.identity_role_edge_config["relationship_terms"]
        return bool(
            terms & (detail_terms | context_terms | relationship_terms)
        )

    def _identity_role_pair_is_specific(self, source_terms: set[str], target_terms: set[str]) -> bool:
        detail_terms = self.identity_role_edge_config["detail_terms"]
        context_terms = self.identity_role_edge_config["context_terms"]
        relationship_terms = self.identity_role_edge_config["relationship_terms"]
        return bool(source_terms & detail_terms or target_terms & detail_terms) or bool(
            (source_terms & context_terms)
            and (target_terms & relationship_terms)
        ) or bool(
            (target_terms & context_terms)
            and (source_terms & relationship_terms)
        )

    @staticmethod
    def _load_identity_role_edge_config(value: Any) -> dict:
        if not isinstance(value, dict):
            return {
                "enabled": False,
                "aliases": {},
                "detail_terms": frozenset(),
                "context_terms": frozenset(),
                "relationship_terms": frozenset(),
            }

        aliases: dict[str, tuple[str, ...]] = {}
        groups: dict[str, set[str]] = {
            "detail": set(),
            "context": set(),
            "relationship": set(),
            "shared": set(),
        }

        def add_group(group_name: str, group_value: Any) -> None:
            if isinstance(group_value, dict):
                items = group_value.items()
            elif isinstance(group_value, list):
                items = ((str(item), [item]) for item in group_value)
            else:
                return
            for key, raw_aliases in items:
                canonical = str(key or "").strip()
                if not canonical:
                    continue
                if isinstance(raw_aliases, str):
                    alias_values = [raw_aliases]
                elif isinstance(raw_aliases, list):
                    alias_values = raw_aliases
                else:
                    alias_values = [canonical]
                cleaned = tuple(
                    str(alias).strip()
                    for alias in [canonical, *alias_values]
                    if str(alias).strip()
                )
                if not cleaned:
                    continue
                aliases[canonical] = tuple(dict.fromkeys(cleaned))
                groups[group_name].add(canonical)

        add_group("detail", value.get("detail"))
        add_group("context", value.get("context"))
        add_group("relationship", value.get("relationship"))
        add_group("shared", value.get("shared"))

        enabled = bool(value.get("enabled", bool(aliases))) and bool(aliases)
        return {
            "enabled": enabled,
            "aliases": aliases,
            "detail_terms": frozenset(groups["detail"]),
            "context_terms": frozenset(groups["context"]),
            "relationship_terms": frozenset(groups["relationship"]),
        }

    @staticmethod
    def _dedupe_proposed_edges(edges: list[dict]) -> list[dict]:
        deduped: dict[tuple[str, str, str], dict] = {}
        for edge in edges:
            key = (
                str(edge.get("source") or ""),
                str(edge.get("target") or ""),
                str(edge.get("relation_type") or ""),
            )
            if not all(key):
                continue
            current = deduped.get(key)
            if current is None or float(edge.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
                deduped[key] = edge
        return list(deduped.values())

    async def _api_reflect(self, period: str, key: str, materials: dict) -> dict:
        client, model, use_dehydration = self._reflect_model_client()
        if not client or not model:
            return self._fallback_reflection(period, key, materials)
        payload = {"period": period, "date": key, **materials}
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self._reflect_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            **self._completion_options(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                thinking_mode="disabled" if use_dehydration else None,
            ),
        )
        raw = response.choices[0].message.content if response.choices else ""
        return self._parse_json_object(raw or "") or self._fallback_reflection(period, key, materials)

    async def _reflection_materials(
        self,
        period: str,
        now_local: datetime,
        bucket_mgr,
        persona_engine,
        conversation_turn_store=None,
        daily_chat_memory_candidates: list[dict] | None = None,
    ) -> dict:
        start, end = self._period_window(period, now_local)
        buckets = []
        daily_impressions = []
        daily_chat_memories = []
        commitments = []
        conversation_turns = []
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception:
            all_buckets = []
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            tags = {str(tag) for tag in meta.get("tags", [])}
            created = self._to_local(meta.get("created"))
            updated = self._to_local(meta.get("updated_at"))
            created_in_window = bool(created and start <= created <= end)
            updated_in_window = bool(updated and start <= updated <= end)
            material_date = self._bucket_material_datetime(meta) if period == "daily" else None
            is_profile_fact = self._is_profile_fact_metadata(meta, tags)
            if period == "weekly" and meta.get("type") == "feel" and "daily_impression" in tags and created_in_window:
                daily_impressions.append(self._memory_payload(bucket, content_limit=360))
            elif period == "daily" and meta.get("type") != "feel":
                if (
                    material_date
                    and start <= material_date <= end
                    and not is_profile_fact
                ):
                    buckets.append(self._memory_payload(bucket, content_limit=420))
                    if meta.get("source") == "daily_chat_memory" or meta.get("from_daily_chat"):
                        daily_chat_memories.append(self._daily_chat_memory_material_from_bucket(bucket))
            elif meta.get("type") != "feel" and (created_in_window or updated_in_window):
                buckets.append(self._memory_payload(bucket, content_limit=420))
            if tags & {"commitment", "todo", "wish"} and not meta.get("resolved"):
                if period != "daily" or (
                    material_date
                    and start <= material_date <= end
                    and not is_profile_fact
                ):
                    commitments.append(self._memory_payload(bucket, content_limit=260))

        if period == "daily" and self.daily_conversation_turn_limit > 0 and conversation_turn_store:
            profile_id = str(getattr(persona_engine, "profile_id", "") or "default")
            try:
                raw_turns = conversation_turn_store.list_conversation_turns_between(
                    profile_id=profile_id,
                    start_at=start,
                    end_at=end,
                    limit=self.daily_conversation_turn_limit,
                )
            except Exception:
                raw_turns = []
            conversation_turns = self._conversation_turn_payloads(
                raw_turns,
                limit=self.daily_conversation_turn_limit,
            )

        if period == "daily":
            key = now_local.date().isoformat()
            daily_chat_memories = self._dedupe_daily_chat_memory_materials(
                [
                    *daily_chat_memories,
                    *self._daily_chat_memory_materials_for_date(
                        key,
                        daily_chat_memory_candidates=daily_chat_memory_candidates,
                    ),
                ]
            )

        persona_events = []
        if self.persona_events_limit > 0 and persona_engine and hasattr(persona_engine, "_list_events"):
            try:
                events = persona_engine._list_events(self.persona_events_scan_limit)
            except Exception:
                events = []
            for event in events:
                created = self._to_local(event.get("created_at"))
                if created and start <= created <= end:
                    persona_events.append(
                        {
                            "id": event.get("id"),
                            "event_type": event.get("event_type", ""),
                            "mood_label": event.get("mood_label", ""),
                            "perceived_intent": event.get("perceived_intent", ""),
                            "surface_trigger": event.get("surface_trigger", ""),
                            "inner_thought": event.get("inner_thought", ""),
                            "residue": event.get("residue", ""),
                            "user_excerpt": event.get("user_excerpt", ""),
                            "assistant_excerpt": event.get("assistant_excerpt", ""),
                            "relationship_event": event.get("relationship_event", False),
                            "personality_signal": event.get("personality_signal", False),
                            "recalled_memory_ids": event.get("recalled_memory_ids", []),
                            "confidence": event.get("confidence", 0.5),
                            "selection_score": event.get("_selection_score"),
                            "created_at": event.get("created_at", ""),
                        }
                    )
            selected_events = select_persona_events(persona_events, limit=self.persona_events_limit)
            persona_events = []
            for event in selected_events:
                cleaned = {key: value for key, value in event.items() if not str(key).startswith("_")}
                if event.get("_selection_score") is not None:
                    cleaned["selection_score"] = event.get("_selection_score")
                persona_events.append(cleaned)
        if conversation_turns:
            persona_events = []
        diary = await self._read_diary_for_date(now_local.date().isoformat()) if period == "daily" else None
        return {
            "buckets": buckets[:30],
            "daily_impressions": daily_impressions[:7],
            "daily_chat_memories": daily_chat_memories[:12],
            "persona_events": persona_events[: self.persona_events_limit],
            "conversation_turns": conversation_turns,
            "commitments": commitments[:12],
            "diary": diary,
        }

    @staticmethod
    def _conversation_turn_payloads(turns: list[dict] | None, limit: int) -> list[dict]:
        if not turns:
            return []
        selected = []
        for turn in turns:
            user_text = str(turn.get("user_text") or "").strip()
            assistant_text = str(turn.get("assistant_text") or "").strip()
            if not user_text and not assistant_text:
                continue
            selected.append(
                {
                    "id": turn.get("id"),
                    "session_id": str(turn.get("session_id") or ""),
                    "round_id": turn.get("round_id"),
                    "created_at": str(turn.get("created_at") or ""),
                    "user_text": user_text[:1200],
                    "assistant_text": assistant_text[:1200],
                    "model": str(turn.get("model") or ""),
                    "client": str(turn.get("client") or ""),
                    "route": str(turn.get("route") or ""),
                }
            )
        selected.sort(key=lambda item: str(item.get("created_at") or ""))
        return selected[-limit:] if limit > 0 else selected

    @staticmethod
    def _raw_event_turn_payloads(events: list[dict] | None, limit: int) -> list[dict]:
        if not events:
            return []
        grouped: dict[tuple[str, str], dict] = {}
        for event in events:
            role = str(event.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            text = str(event.get("text") or "").strip()
            if not text:
                continue
            metadata = event.get("metadata", {}) if isinstance(event.get("metadata"), dict) else {}
            session_id = str(event.get("session_id") or event.get("conversation_id") or "").strip()
            round_value = metadata.get("round_id")
            round_key = str(round_value).strip() if round_value is not None else ""
            event_id = int(event.get("id") or 0)
            key = (session_id, round_key or f"event:{event_id}")
            row = grouped.get(key)
            if row is None:
                row = {
                    "id": None,
                    "session_id": session_id,
                    "round_id": int(round_key) if round_key.isdigit() else None,
                    "created_at": str(event.get("created_at") or ""),
                    "user_text": "",
                    "assistant_text": "",
                    "model": str(metadata.get("model") or ""),
                    "client": str(event.get("client") or ""),
                    "route": str(metadata.get("route") or ""),
                    "raw_event_ids": [],
                    "source_event_ids": [],
                }
                grouped[key] = row
            row["raw_event_ids"].append(event_id)
            source_event_id = str(event.get("source_event_id") or "").strip()
            if source_event_id:
                row["source_event_ids"].append(source_event_id)
            if not row.get("created_at"):
                row["created_at"] = str(event.get("created_at") or "")
            if role == "user":
                row["user_text"] = f"{row['user_text']} / {text}".strip(" /") if row["user_text"] else text
            else:
                row["assistant_text"] = (
                    f"{row['assistant_text']} / {text}".strip(" /")
                    if row["assistant_text"]
                    else text
                )

        selected = []
        for row in grouped.values():
            if not row["user_text"] and not row["assistant_text"]:
                continue
            row["raw_event_ids"] = list(dict.fromkeys(row["raw_event_ids"]))
            row["source_event_ids"] = list(dict.fromkeys(row["source_event_ids"]))
            # Raw-event turns have no store-assigned id; give each turn a real,
            # citable id so source verification (and the model) can reference it.
            turn_id = row.get("round_id")
            if turn_id is None:
                turn_id = row["raw_event_ids"][0] if row["raw_event_ids"] else None
            selected.append(
                {
                    **row,
                    "id": turn_id,
                    "user_text": row["user_text"][:1200],
                    "assistant_text": row["assistant_text"][:1200],
                }
            )
        selected.sort(key=lambda item: (str(item.get("created_at") or ""), int(item.get("id") or 0)))
        return selected[-limit:] if limit > 0 else selected

    def _daily_chat_memory_candidate_limit(self, mode: str) -> int:
        if self._normalize_daily_chat_memory_mode(mode) == "review":
            return self.daily_chat_memory_review_max_per_day or self.daily_chat_memory_max_per_day
        return self.daily_chat_memory_max_per_day

    def _daily_chat_memory_min_confidence_for_mode(self, mode: str) -> float:
        if self._normalize_daily_chat_memory_mode(mode) == "review":
            return self.daily_chat_memory_review_min_confidence
        return self.daily_chat_memory_min_confidence

    def _daily_chat_memory_windows(self, turns: list[dict]) -> list[list[dict]]:
        if not turns:
            return []
        window = self.daily_chat_memory_summary_window_turns
        stride = self.daily_chat_memory_summary_stride_turns
        if len(turns) <= window:
            return [turns]
        windows: list[list[dict]] = []
        start = 0
        seen_ranges: set[tuple[int, int]] = set()
        while start < len(turns):
            end = min(len(turns), start + window)
            range_key = (start, end)
            if range_key not in seen_ranges:
                windows.append(turns[start:end])
                seen_ranges.add(range_key)
            if end >= len(turns):
                break
            start += stride
        return windows

    @staticmethod
    def _daily_chat_memory_window_source_ids(turns: list[dict]) -> tuple[list[int], list[int]]:
        turn_ids = [
            int(turn.get("id"))
            for turn in turns
            if turn.get("id") is not None and str(turn.get("id")).isdigit()
        ]
        event_ids = [
            int(event_id)
            for turn in turns
            for event_id in (turn.get("raw_event_ids") or [])
            if event_id is not None and str(event_id).isdigit()
        ]
        return list(dict.fromkeys(turn_ids)), list(dict.fromkeys(event_ids))

    async def _summarize_daily_chat_memory_windows(
        self,
        key: str,
        turns: list[dict],
        *,
        self_context: str = "",
    ) -> list[dict]:
        if not self.daily_chat_memory_summary_enabled:
            return []
        client, model, use_daily_client = self._daily_chat_memory_model_client(candidate=False)
        if not client:
            return []
        windows = self._daily_chat_memory_windows(turns)
        if not windows:
            return []
        summaries: list[dict] = []
        for index, window_turns in enumerate(windows):
            fallback_turn_ids, fallback_event_ids = self._daily_chat_memory_window_source_ids(window_turns)
            payload = {
                "date": key,
                "identity": {
                    "ai_name": self.identity["ai_name"],
                    "user_name": self.identity["user_name"],
                    "user_display_name": self.identity["user_display_name"],
                    "user_aliases": self.identity.get("user_aliases", []),
                },
                "self_anchor_entry": self_context,
                "window": {
                    "index": index + 1,
                    "total": len(windows),
                    "source_turn_ids": fallback_turn_ids,
                    "source_event_ids": fallback_event_ids,
                },
                "conversation_turns": window_turns,
            }
            try:
                response = await self._daily_chat_memory_create_completion(
                    client,
                    model=model,
                    messages=[
                        {"role": "system", "content": self._daily_chat_memory_summary_prompt()},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    max_tokens=self.daily_chat_memory_summary_max_tokens,
                    temperature=self.temperature,
                    use_daily_client=use_daily_client,
                )
                raw = self._completion_content(response)
                parsed = self._parse_json_object(raw or "")
            except Exception as exc:
                logger.warning("Daily chat memory summary failed, window=%s: %s", index + 1, exc)
                continue
            raw_summaries = parsed.get("summaries") if isinstance(parsed, dict) else []
            if not isinstance(raw_summaries, list):
                continue
            for item in raw_summaries:
                if not isinstance(item, dict):
                    continue
                cleaned = self._normalize_daily_chat_memory_summary(
                    item,
                    key=key,
                    window_index=index + 1,
                    source_turn_ids=fallback_turn_ids,
                    source_event_ids=fallback_event_ids,
                )
                if cleaned:
                    summaries.append(cleaned)
        return summaries

    def _daily_chat_memory_model_client(self, *, candidate: bool) -> tuple[Any, str, bool]:
        if self.daily_chat_memory_client:
            client = self.daily_chat_memory_client
            model = self.daily_chat_memory_candidate_model if candidate else self.daily_chat_memory_summary_model
            return client, str(model or "").strip(), True
        dehy_client = self._daily_dehydration_client()
        if dehy_client and self.dehydration_model:
            return dehy_client, str(self.dehydration_model or "").strip(), False
        if self.client:
            return self.client, str(self.model or "").strip(), False
        return None, "", False

    def _reflect_model_client(self) -> tuple[Any, str, bool]:
        dehy_client = self._daily_dehydration_client()
        if dehy_client and self.dehydration_model:
            return dehy_client, self.dehydration_model, True
        if self.client and self.model:
            return self.client, str(self.model or "").strip(), False
        return None, "", False

    def _normalize_daily_chat_memory_summary(
        self,
        item: dict,
        *,
        key: str,
        window_index: int,
        source_turn_ids: list[int],
        source_event_ids: list[int],
    ) -> dict:
        text = re.sub(
            r"\s+",
            " ",
            strip_wikilinks(str(item.get("summary") or item.get("content") or "")).strip(),
        )
        if not text or self._daily_chat_memory_noise(text):
            return {}
        confidence = self._clamp(item.get("confidence", 0.65))
        if confidence < 0.5:
            return {}
        title = str(item.get("title") or "").strip()
        if self._daily_chat_memory_title_is_generic(title):
            title = self._daily_chat_memory_title(text, "key_event", key)
        raw_turn_ids = [
            int(turn_id)
            for turn_id in self._string_list(item.get("source_turn_ids"), limit=80)
            if str(turn_id).isdigit()
        ] or source_turn_ids
        raw_event_ids = [
            int(event_id)
            for event_id in self._string_list(item.get("source_event_ids"), limit=160)
            if str(event_id).isdigit()
        ] or source_event_ids
        signals = self._string_list(item.get("signals"), limit=8)
        return {
            "window_index": window_index,
            "title": title[:40],
            "summary": text[:900],
            "signals": signals,
            "source_turn_ids": raw_turn_ids[:80],
            "source_event_ids": raw_event_ids[:160],
            "confidence": confidence,
        }

    def _daily_chat_memory_materials_for_date(
        self,
        key: str,
        *,
        daily_chat_memory_candidates: list[dict] | None = None,
    ) -> list[dict]:
        materials: list[dict] = []
        for candidate in daily_chat_memory_candidates or []:
            if not self._daily_chat_memory_candidate_is_material(candidate):
                continue
            material = self._daily_chat_memory_material(candidate)
            if material and (not key or material.get("date") == key):
                materials.append(material)
        for item in self._load_daily_chat_memory_pending():
            if str(item.get("date") or "") != key:
                continue
            item_status = str(item.get("status") or "pending").strip().lower()
            candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
            if not self._daily_chat_memory_candidate_is_material(candidate, status=item_status):
                continue
            material = self._daily_chat_memory_material(candidate)
            if material:
                materials.append(material)
        return self._dedupe_daily_chat_memory_materials(materials)

    @staticmethod
    def _daily_chat_memory_candidate_is_material(
        candidate: dict,
        *,
        status: str | None = None,
    ) -> bool:
        """Allow only adopted candidate projections into reflection/activity paths."""
        if not isinstance(candidate, dict):
            return False
        mode = str(candidate.get("mode") or "").strip().lower()
        candidate_status = str(
            status if status is not None else candidate.get("status") or ""
        ).strip().lower()
        if mode == "review":
            return candidate_status == "confirmed"
        if mode == "auto":
            return candidate_status == "applied"
        # Backward-compatible confirmed records from the original JSON flow.
        return candidate_status in {"confirmed", "applied"}

    @staticmethod
    def _dedupe_daily_chat_memory_materials(materials: list[dict]) -> list[dict]:
        deduped: dict[str, dict] = {}
        for item in materials:
            item_id = str(item.get("id") or "")
            key_value = item_id or re.sub(r"\s+", "", str(item.get("content") or "")).lower()
            if key_value and key_value not in deduped:
                deduped[key_value] = item
        return list(deduped.values())

    def _daily_chat_memory_material_from_bucket(self, bucket: dict) -> dict:
        if not isinstance(bucket, dict):
            return {}
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        candidate_id = str(meta.get("daily_chat_memory_candidate_id") or bucket.get("id") or "").strip()
        source_turn_ids = meta.get("source_conversation_turn_ids") or []
        source_event_ids = meta.get("source_raw_event_ids") or []
        return self._daily_chat_memory_material(
            {
                "id": candidate_id,
                "date": meta.get("event_date") or meta.get("date") or "",
                "kind": meta.get("kind") or "",
                "title": meta.get("name") or "",
                "content": bucket.get("content") or "",
                "tags": meta.get("tags") or [],
                "domain": meta.get("domain") or [],
                "confidence": meta.get("confidence", 0.7),
                "source_turn_ids": source_turn_ids,
                "source_event_ids": source_event_ids,
                "reason": meta.get("daily_chat_memory_reason") or "",
            }
        )

    def _daily_chat_memory_material(self, candidate: dict) -> dict:
        if not isinstance(candidate, dict):
            return {}
        content = re.sub(
            r"\s+",
            " ",
            strip_wikilinks(str(candidate.get("content") or candidate.get("summary") or "")).strip(),
        )
        if not content:
            return {}
        title = str(candidate.get("title") or "").strip()
        return {
            "id": str(candidate.get("id") or "").strip(),
            "date": str(candidate.get("date") or "").strip(),
            "kind": str(candidate.get("kind") or "").strip(),
            "title": title[:40],
            "content": content[:420],
            "tags": self._string_list(candidate.get("tags"), limit=12),
            "domain": self._string_list(candidate.get("domain"), limit=6),
            "confidence": self._clamp(candidate.get("confidence", 0.65)),
            "source_turn_ids": [
                int(turn_id)
                for turn_id in self._string_list(candidate.get("source_turn_ids"), limit=80)
                if str(turn_id).isdigit()
            ],
            "source_event_ids": [
                int(event_id)
                for event_id in self._string_list(candidate.get("source_event_ids"), limit=160)
                if str(event_id).isdigit()
            ],
            "reason": str(candidate.get("reason") or "").strip()[:160],
        }

    @staticmethod
    def _max_daily_chat_memory_raw_event_id(turns: list[dict]) -> int:
        max_id = 0
        for turn in turns or []:
            for event_id in turn.get("raw_event_ids") or []:
                try:
                    max_id = max(max_id, int(event_id or 0))
                except (TypeError, ValueError):
                    continue
        return max_id

    def _daily_activity_summary_turns(
        self,
        *,
        profile_id: str,
        start: datetime,
        end: datetime,
        conversation_turn_store=None,
        raw_event_store=None,
    ) -> tuple[list[dict], str]:
        limit = self.daily_activity_summary_turn_limit
        if raw_event_store:
            try:
                raw_events = raw_event_store.list_events_between(
                    start_at=start,
                    end_at=end,
                    limit=limit,
                )
            except Exception as exc:
                logger.warning("Daily activity summary raw event read failed: %s", exc)
                raw_events = []
            if raw_events:
                raw_events = [
                    event
                    for event in raw_events
                    if not (
                        isinstance(event.get("metadata"), dict)
                        and event["metadata"].get("profile_id")
                        and str(event["metadata"].get("profile_id")) != profile_id
                    )
                ]
                turns = self._raw_event_turn_payloads(raw_events, limit=limit)
                if turns:
                    return turns, "raw_events"

        if conversation_turn_store:
            try:
                raw_turns = conversation_turn_store.list_conversation_turns_between(
                    profile_id=profile_id,
                    start_at=start,
                    end_at=end,
                    limit=limit or 80,
                )
            except Exception as exc:
                logger.warning("Daily activity summary turn read failed: %s", exc)
                raw_turns = []
            turns = self._conversation_turn_payloads(raw_turns, limit=limit)
            if turns:
                return turns, "conversation_turns"
        return [], ""

    async def run_daily_activity_summary(
        self,
        *,
        conversation_turn_store=None,
        raw_event_store=None,
        persona_engine=None,
        daily_chat_memory_candidates: list[dict] | None = None,
        daily_impressions: list[dict] | None = None,
        key: str = "",
        force: bool = False,
        now: datetime | None = None,
    ) -> dict:
        if not self.enabled or not self.daily_activity_summary_enabled:
            return {"status": "disabled", "reason": "daily_activity_summary_off"}

        now_local = self._daily_chat_memory_target(key, now)
        key = now_local.date().isoformat()
        memory_item = self._daily_activity_summary_from_memory_materials(
            key,
            daily_chat_memories=self._daily_chat_memory_materials_for_date(
                key,
                daily_chat_memory_candidates=daily_chat_memory_candidates,
            ),
            daily_impressions=daily_impressions or [],
        )
        if memory_item:
            return {
                "status": "ready",
                "date": key,
                "turns": 0,
                "turn_source": "daily_memory_materials",
                "force": bool(force),
                "activity_summary": memory_item,
            }

        if not conversation_turn_store and not raw_event_store:
            return {"status": "skipped", "reason": "no_conversation_source"}

        start, end = self._period_window("daily", now_local)
        profile_id = str(getattr(persona_engine, "profile_id", "") or "default")
        turns, turn_source = self._daily_activity_summary_turns(
            profile_id=profile_id,
            start=start,
            end=end,
            conversation_turn_store=conversation_turn_store,
            raw_event_store=raw_event_store,
        )
        if not turns:
            return {"status": "skipped", "reason": "no_conversation_turns", "date": key}

        item = await self._extract_daily_activity_summary(key, turns)
        if not item and self._daily_activity_summary_dehydration_retry_available():
            item = await self._extract_daily_activity_summary(
                key,
                turns,
                client_override=self._daily_dehydration_client(),
                model_override=self.dehydration_model,
            )
        if not item:
            item = self._fallback_daily_activity_summary(key, turns)
        if not item:
            return {
                "status": "skipped",
                "reason": "no_activity_summary",
                "date": key,
                "turns": len(turns),
                "turn_source": turn_source,
            }
        return {
            "status": "ready",
            "date": key,
            "turns": len(turns),
            "turn_source": turn_source,
            "force": bool(force),
            "activity_summary": item,
        }

    def _daily_activity_summary_from_memory_materials(
        self,
        key: str,
        *,
        daily_chat_memories: list[dict],
        daily_impressions: list[dict],
    ) -> dict:
        snippets: list[str] = []
        source_event_ids: list[int] = []
        source_turn_ids: list[int] = []
        evidence: list[dict] = []
        confidences: list[float] = []
        for item in daily_chat_memories or []:
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            text = content or title
            if title and content and title not in content:
                text = f"{title}：{content}"
            text = re.sub(r"\s+", " ", strip_wikilinks(text)).strip()
            if text:
                snippets.append(self._clip_activity_material(text))
            source_event_ids.extend(
                int(event_id)
                for event_id in (item.get("source_event_ids") or [])
                if str(event_id).isdigit()
            )
            source_turn_ids.extend(
                int(turn_id)
                for turn_id in (item.get("source_turn_ids") or [])
                if str(turn_id).isdigit()
            )
            if item.get("id"):
                evidence.append({"candidate_id": str(item.get("id"))})
            confidences.append(self._clamp(item.get("confidence", 0.65)))
        for item in daily_impressions or []:
            content = re.sub(
                r"\s+",
                " ",
                strip_wikilinks(str(item.get("content") or item.get("text") or "")).strip(),
            )
            content = re.split(r"\n?### affect_anchor\b", content, maxsplit=1)[0].strip()
            if content:
                snippets.append(self._clip_activity_material(content, limit=80))
            if item.get("id"):
                evidence.append({"bucket_id": str(item.get("id"))})
            confidences.append(self._clamp(item.get("confidence", 0.7)))
        snippets = list(dict.fromkeys(snippets))[:4]
        if not snippets:
            return {}
        text = "；".join(snippets)
        if len(text) > 140:
            text = text[:137].rstrip("，,；;、 ") + "..."
        return {
            "timeline_id": f"daily_activity_summary:{key}",
            "source": "daily_activity_summary",
            "scope": "doing",
            "text": text,
            "evidence": evidence[:4],
            "source_date": key,
            "source_dates": [key],
            "timestamp": self._daily_chat_memory_created_at(key),
            "confidence": max(confidences or [0.65]),
            "source_turn_ids": list(dict.fromkeys(source_turn_ids))[:80],
            "source_event_ids": list(dict.fromkeys(source_event_ids))[:160],
        }

    @staticmethod
    def _clip_activity_material(text: str, *, limit: int = 72) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip(" -\t\r\n")
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip("，,；;、 ") + "..."

    async def _extract_daily_activity_summary(
        self,
        key: str,
        turns: list[dict],
        *,
        use_daily_client: bool | None = None,
        client_override: Any = None,
        model_override: str = "",
    ) -> dict:
        if client_override is not None:
            client = client_override
        elif use_daily_client is False:
            client = self.client
        else:
            client = self.daily_chat_memory_client or self.client
        if not client:
            return {}
        use_daily_client = client is self.daily_chat_memory_client
        model = str(model_override or "").strip() or (
            self.daily_chat_memory_summary_model if use_daily_client else self.model
        )
        fallback_turn_ids, fallback_event_ids = self._daily_chat_memory_window_source_ids(turns)
        payload = {
            "date": key,
            "identity": {
                "ai_name": self.identity["ai_name"],
                "user_name": self.identity["user_name"],
                "user_display_name": self.identity["user_display_name"],
                "user_aliases": self.identity.get("user_aliases", []),
            },
            "source_turn_ids": fallback_turn_ids,
            "source_event_ids": fallback_event_ids,
            "conversation_turns": turns,
        }
        try:
            response = await self._daily_chat_memory_create_completion(
                client,
                model=model,
                messages=[
                    {"role": "system", "content": self._daily_activity_summary_prompt()},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=self.daily_activity_summary_max_tokens,
                temperature=self.temperature,
                use_daily_client=use_daily_client,
            )
            parsed = self._parse_json_object(self._completion_content(response) or "")
        except Exception as exc:
            logger.warning("Daily activity summary model failed: %s", exc)
            return {}
        return self._normalize_daily_activity_summary(
            key,
            parsed,
            turns,
            source_turn_ids=fallback_turn_ids,
            source_event_ids=fallback_event_ids,
        )

    def _daily_activity_summary_dehydration_retry_available(self) -> bool:
        if not self._daily_dehydration_client():
            return False
        first_model = self.daily_chat_memory_summary_model if self.daily_chat_memory_client else self.model
        first_base_url = self.daily_chat_memory_base_url if self.daily_chat_memory_client else self.base_url
        first_api_key = self.daily_chat_memory_api_key if self.daily_chat_memory_client else self.api_key
        return (
            self.dehydration_model != str(first_model or "").strip()
            or self.dehydration_base_url != str(first_base_url or "").strip().rstrip("/")
            or self.dehydration_api_key != str(first_api_key or "").strip()
        )

    def _daily_dehydration_client(self) -> Any:
        return self.dehydration_client or self.daily_activity_summary_dehydration_client

    def _normalize_daily_activity_summary(
        self,
        key: str,
        item: dict,
        turns: list[dict],
        *,
        source_turn_ids: list[int],
        source_event_ids: list[int],
    ) -> dict:
        if not isinstance(item, dict):
            return {}
        text = re.sub(
            r"\s+",
            " ",
            strip_wikilinks(str(item.get("summary") or item.get("text") or item.get("content") or "")).strip(),
        )
        text = re.sub(r"^(今天的?(?:总结|摘要|主要进展)?(?:是|：|:)?\s*)", "", text).strip()
        if not text:
            return {}
        if len(text) > 140:
            text = text[:137].rstrip("，,；;、 ") + "..."
        confidence = self._clamp(item.get("confidence", 0.65))
        if confidence < 0.35:
            return {}
        raw_turn_ids = [
            int(turn_id)
            for turn_id in self._string_list(item.get("source_turn_ids"), limit=80)
            if str(turn_id).isdigit()
        ] or source_turn_ids
        raw_event_ids = [
            int(event_id)
            for event_id in self._string_list(item.get("source_event_ids"), limit=160)
            if str(event_id).isdigit()
        ] or source_event_ids
        sessions = [
            str(turn.get("session_id") or "").strip()
            for turn in turns
            if str(turn.get("session_id") or "").strip()
        ]
        evidence = [{"session_id": session_id} for session_id in list(dict.fromkeys(sessions))[:3]]
        return {
            "timeline_id": f"daily_activity_summary:{key}",
            "source": "daily_activity_summary",
            "scope": "doing",
            "text": text,
            "evidence": evidence,
            "source_date": key,
            "source_dates": [key],
            "timestamp": self._daily_activity_summary_timestamp(key, turns),
            "confidence": confidence,
            "source_turn_ids": raw_turn_ids[:80],
            "source_event_ids": raw_event_ids[:160],
        }

    def _fallback_daily_activity_summary(self, key: str, turns: list[dict]) -> dict:
        fallback_turn_ids, fallback_event_ids = self._daily_chat_memory_window_source_ids(turns)
        snippets = []
        for turn in reversed(turns or []):
            snippet = self._daily_activity_summary_excerpt(turn.get("user_text") or turn.get("assistant_text") or "")
            if snippet:
                snippets.append(snippet)
            if len(snippets) >= 3:
                break
        snippets = list(reversed(list(dict.fromkeys(snippets))))
        if not snippets:
            return {}
        joined = "；".join(snippets)
        if len(joined) > 96:
            joined = joined[:93].rstrip("，,；;、 ") + "..."
        return self._normalize_daily_activity_summary(
            key,
            {
                "summary": f"围绕{joined}继续推进。",
                "confidence": 0.45,
                "source_turn_ids": fallback_turn_ids,
                "source_event_ids": fallback_event_ids,
            },
            turns,
            source_turn_ids=fallback_turn_ids,
            source_event_ids=fallback_event_ids,
        )

    @staticmethod
    def _daily_activity_summary_excerpt(value: Any) -> str:
        text = strip_wikilinks(str(value or "")).strip()
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        text = re.sub(r"<attachment\b[^>]*>.*?</attachment>", " ", text, flags=re.I | re.S)
        text = re.sub(r"【当前时间】[^\n\r]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" -\t\r\n")
        return text[:60].rstrip("，,；;、 ")

    def _daily_activity_summary_timestamp(self, key: str, turns: list[dict]) -> str:
        latest: datetime | None = None
        for turn in turns:
            parsed = self._to_local(turn.get("created_at"))
            if parsed and (latest is None or parsed > latest):
                latest = parsed
        if latest:
            return latest.isoformat(timespec="minutes")
        return self._daily_chat_memory_created_at(key)

    async def run_daily_chat_memory(
        self,
        bucket_mgr,
        *,
        conversation_turn_store=None,
        raw_event_store=None,
        persona_engine=None,
        embedding_engine=None,
        key: str = "",
        mode: str = "",
        force: bool = False,
        now: datetime | None = None,
    ) -> dict:
        """Run daily chat memory with a per-process concurrency lock.

        The lock guarantees a manual dashboard "补扫" cannot overlap a scheduled
        run (request_id + concurrency lock semantics)."""
        effective_mode = self._normalize_daily_chat_memory_mode(mode or self.daily_chat_memory_mode)
        if effective_mode == "off" or self.daily_chat_memory_max_per_day <= 0:
            return {"status": "disabled", "reason": "daily_chat_memory_off", "mode": effective_mode}
        if not conversation_turn_store:
            return {"status": "skipped", "reason": "no_conversation_turn_store", "mode": effective_mode}
        if not self._daily_chat_memory_run_lock_acquire():
            return {
                "status": "locked",
                "reason": "daily_chat_memory_run_in_progress",
                "date": (now or datetime.now(timezone.utc)).astimezone(self.tz).date().isoformat(),
                "mode": effective_mode,
            }
        try:
            return await self._run_daily_chat_memory_impl(
                bucket_mgr,
                conversation_turn_store=conversation_turn_store,
                raw_event_store=raw_event_store,
                persona_engine=persona_engine,
                embedding_engine=embedding_engine,
                key=key,
                mode=mode,
                force=force,
                now=now,
            )
        finally:
            self._daily_chat_memory_run_lock_release()

    async def _run_daily_chat_memory_impl(
        self,
        bucket_mgr,
        *,
        conversation_turn_store=None,
        raw_event_store=None,
        persona_engine=None,
        embedding_engine=None,
        key: str = "",
        mode: str = "",
        force: bool = False,
        now: datetime | None = None,
    ) -> dict:
        effective_mode = self._normalize_daily_chat_memory_mode(mode or self.daily_chat_memory_mode)
        if effective_mode == "off" or self.daily_chat_memory_max_per_day <= 0:
            return {"status": "disabled", "reason": "daily_chat_memory_off", "mode": effective_mode}
        if not conversation_turn_store:
            return {"status": "skipped", "reason": "no_conversation_turn_store", "mode": effective_mode}

        now_local = self._daily_chat_memory_target(key, now)
        key = now_local.date().isoformat()
        start, end = self._period_window("daily", now_local)
        profile_id = str(getattr(persona_engine, "profile_id", "") or "default")
        turns = []
        turn_source = ""
        raw_event_cursor_id = 0 if force else self._daily_chat_memory_last_raw_event_id(profile_id)
        max_seen_raw_event_id = 0
        raw_events_cursor_exhausted = False
        if raw_event_store:
            try:
                raw_events = raw_event_store.list_events_between(
                    start_at=start,
                    end_at=end,
                    limit=self.daily_chat_memory_turn_limit,
                )
            except Exception as exc:
                logger.warning("Daily chat memory raw event read failed: %s", exc)
                raw_events = []
            if raw_events:
                raw_events = [
                    event
                    for event in raw_events
                    if not (
                        isinstance(event.get("metadata"), dict)
                        and event["metadata"].get("profile_id")
                        and str(event["metadata"].get("profile_id")) != profile_id
                    )
                ]
                if raw_event_cursor_id > 0:
                    raw_events = [
                        event
                        for event in raw_events
                        if int(event.get("id") or 0) > raw_event_cursor_id
                    ]
                    raw_events_cursor_exhausted = not raw_events
                turns = self._raw_event_turn_payloads(
                    raw_events,
                    limit=self.daily_chat_memory_turn_limit,
                )
                if turns:
                    turn_source = "raw_events"
                    max_seen_raw_event_id = self._max_daily_chat_memory_raw_event_id(turns)

        try:
            raw_turns = []
            if raw_events_cursor_exhausted:
                raw_turns = []
            elif not turns and conversation_turn_store:
                raw_turns = conversation_turn_store.list_conversation_turns_between(
                    profile_id=profile_id,
                    start_at=start,
                    end_at=end,
                    limit=self.daily_chat_memory_turn_limit or 80,
                )
        except Exception as exc:
            logger.warning("Daily chat memory turn read failed: %s", exc)
            raw_turns = []
        if not turns:
            if raw_events_cursor_exhausted:
                return {
                    "status": "skipped",
                    "reason": "no_new_raw_events",
                    "date": key,
                    "mode": effective_mode,
                    "last_raw_event_id": raw_event_cursor_id,
                }
            turns = self._conversation_turn_payloads(raw_turns, limit=self.daily_chat_memory_turn_limit)
            if turns:
                turn_source = "conversation_turns"
        if not turns:
            return {"status": "skipped", "reason": "no_conversation_turns", "date": key, "mode": effective_mode}

        self_context = await self._daily_chat_memory_self_context(bucket_mgr)
        max_candidates = self._daily_chat_memory_candidate_limit(effective_mode)
        min_confidence = self._daily_chat_memory_min_confidence_for_mode(effective_mode)
        window_summaries = await self._summarize_daily_chat_memory_windows(
            key,
            turns,
            self_context=self_context,
        )
        run_id = hashlib.sha1(
            f"daily_chat_memory|{key}|{raw_event_cursor_id}|{datetime.now(timezone.utc).isoformat(timespec='seconds')}"
            .encode("utf-8")
        ).hexdigest()[:12]
        audit: dict = {
            "run_id": run_id,
            "date": key,
            "source_start_seq": raw_event_cursor_id,
            "source_end_seq": max_seen_raw_event_id or raw_event_cursor_id,
            "eligible_turn_count": len(turns),
            "window_count": 0,
            "skipped_noise_window_count": 0,
            "model_call_count": 0,
            "model_candidate_count": 0,
            "hard_rejects": {},
            "soft_flags": {},
            "merged_duplicates": 0,
            "pending_count": 0,
            "auto_applied_count": 0,
            "empty_output_count": 0,
            "parse_failure_count": 0,
            "model": "",
            "status": "failed",
            "error_category": "",
            "prompt_version": "daily_chat_memory_v4.2",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "completed_at": "",
        }
        raw_candidates, extraction_meta = await self._extract_daily_chat_memory_candidates(
            key,
            turns,
            self_context=self_context,
            window_summaries=window_summaries,
            max_candidates=max_candidates,
            audit=audit,
        )
        audit["model_candidate_count"] = len(raw_candidates)
        candidates = self._normalize_daily_chat_memory_candidates(
            key,
            raw_candidates,
            turns,
            max_candidates=max_candidates,
            min_confidence=min_confidence,
            mode=effective_mode,
            audit=audit,
        )
        # V4 watermark semantics: a run advances only the range it actually
        # scanned to completion. zero_candidates still advances (windows were
        # checked); partial stops at the last successfully processed window; the
        # failed range is re-scanned on retry (idempotent, no double pending).
        partial = bool(extraction_meta.get("partial"))
        cursor_target = (
            max_seen_raw_event_id
            if not partial
            else max(extraction_meta.get("last_processed_event_id") or 0, raw_event_cursor_id)
        )
        if not candidates and partial:
            audit["status"] = "partial"
            audit["error_category"] = "window_extraction_failed"
            audit["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._store_daily_chat_memory_run_audit(audit)
            return {
                "status": "partial",
                "reason": "window_extraction_failed",
                "date": key,
                "mode": effective_mode,
                "turns": len(turns),
                "source_start_seq": raw_event_cursor_id,
                "source_end_seq": audit["source_end_seq"],
                "processed_until_seq": cursor_target,
                "cursor_updated": False,
                "run_id": run_id,
                "error_category": "window_extraction_failed",
            }
        empty_outputs = int(extraction_meta.get("empty_output_count") or 0)
        model_calls = int(extraction_meta.get("model_call_count") or 0)
        if not candidates and model_calls > 0 and empty_outputs >= model_calls:
            # Every model call returned an empty body: systemic degradation, not a
            # legitimate zero-candidate day. The watermark stays put so the same
            # range is re-scanned once the provider/code issue is fixed.
            audit["status"] = "degraded_empty_outputs"
            audit["error_category"] = "empty_model_outputs"
            audit["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._store_daily_chat_memory_run_audit(audit)
            return {
                "status": "degraded_empty_outputs",
                "reason": "empty_model_outputs",
                "date": key,
                "mode": effective_mode,
                "turns": len(turns),
                "turn_source": turn_source,
                "source_start_seq": raw_event_cursor_id,
                "source_end_seq": audit["source_end_seq"],
                "processed_until_seq": raw_event_cursor_id,
                "cursor_updated": False,
                "run_id": run_id,
                "error_category": "empty_model_outputs",
            }
        if not candidates:
            # All windows scanned, none produced: zero_candidates (allowed), and
            # the processed watermark advances to the actual scanned end.
            cursor_updated = (
                self._update_daily_chat_memory_raw_cursor(profile_id, cursor_target, key)
                if turn_source == "raw_events" and cursor_target > raw_event_cursor_id
                else False
            )
            audit["status"] = "zero_candidates"
            audit["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._store_daily_chat_memory_run_audit(audit)
            return {
                "status": "zero_candidates",
                "reason": "no_candidates",
                "date": key,
                "mode": effective_mode,
                "turns": len(turns),
                "turn_source": turn_source,
                "window_summaries": len(window_summaries),
                "source_start_seq": raw_event_cursor_id,
                "source_end_seq": audit["source_end_seq"],
                "last_raw_event_id": cursor_target,
                "cursor_updated": cursor_updated,
                "run_id": run_id,
                "model_candidate_count": audit["model_candidate_count"],
                "hard_rejects": audit["hard_rejects"],
                "soft_flags": audit["soft_flags"],
            }

        if effective_mode == "review":
            candidates = [
                {**candidate, "mode": "review", "status": "pending"}
                for candidate in candidates
            ]
            pending = self._store_daily_chat_memory_pending(candidates, force=force)
            cursor_updated = (
                self._update_daily_chat_memory_raw_cursor(profile_id, cursor_target, key)
                if turn_source == "raw_events" and cursor_target > raw_event_cursor_id
                else False
            )
            audit["status"] = "partial" if partial else "success"
            audit["pending_count"] = pending.get("added", 0)
            audit["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._store_daily_chat_memory_run_audit(audit)
            return {
                "status": "partial" if partial else "pending",
                "reason": "window_extraction_failed" if partial else "",
                "date": key,
                "mode": effective_mode,
                "turns": len(turns),
                "turn_source": turn_source,
                "window_summaries": len(window_summaries),
                "source_start_seq": raw_event_cursor_id,
                "source_end_seq": audit["source_end_seq"],
                "last_raw_event_id": cursor_target,
                "cursor_updated": cursor_updated,
                "run_id": run_id,
                "candidates": candidates,
                **pending,
            }

        write_result = await self._write_daily_chat_memory_candidates(
            candidates,
            bucket_mgr,
            embedding_engine=embedding_engine,
        )
        result_by_id = {
            str(result.get("id") or ""): result
            for result in (write_result.get("results") or [])
            if isinstance(result, dict)
        }
        candidates = [
            {
                **candidate,
                "mode": "auto",
                "status": (
                    "applied"
                    if result_by_id.get(str(candidate.get("id") or ""), {}).get("status")
                    in {"created", "exists"}
                    else "apply_failed"
                ),
            }
            for candidate in candidates
        ]
        cursor_updated = (
            self._update_daily_chat_memory_raw_cursor(profile_id, cursor_target, key)
            if turn_source == "raw_events" and cursor_target > raw_event_cursor_id
            else False
        )
        applied_count = sum(1 for candidate in candidates if candidate.get("status") == "applied")
        audit["status"] = "partial" if partial else ("success" if write_result.get("created") else "exists")
        audit["auto_applied_count"] = applied_count
        audit["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._store_daily_chat_memory_run_audit(audit)
        return {
            "status": "partial" if partial else ("created" if write_result.get("created") else "exists"),
            "reason": "window_extraction_failed" if partial else "",
            "date": key,
            "mode": effective_mode,
            "turns": len(turns),
            "turn_source": turn_source,
            "window_summaries": len(window_summaries),
            "source_start_seq": raw_event_cursor_id,
            "source_end_seq": audit["source_end_seq"],
            "last_raw_event_id": cursor_target,
            "cursor_updated": cursor_updated,
            "run_id": run_id,
            "candidates": candidates,
            **write_result,
        }

    async def _daily_chat_memory_self_context(self, bucket_mgr) -> str:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as exc:
            logger.warning("Daily chat memory self-anchor read failed: %s", exc)
            return ""
        if not all_buckets:
            return ""

        self_anchor_cfg = self.config.get("self_anchor", {}) if isinstance(self.config.get("self_anchor", {}), dict) else {}
        configured_id = str(self_anchor_cfg.get("entry_bucket_id") or "").strip()
        if configured_id:
            for bucket in all_buckets:
                if str(bucket.get("id") or "") == configured_id and self._active_self_anchor_bucket(bucket):
                    return self._daily_chat_self_anchor_text(bucket)
            return ""

        candidates = [bucket for bucket in all_buckets if self._active_self_anchor_bucket(bucket)]
        candidates.sort(
            key=lambda bucket: (
                self._int_between((bucket.get("metadata") or {}).get("importance"), 5),
                str((bucket.get("metadata") or {}).get("updated_at") or (bucket.get("metadata") or {}).get("created") or ""),
            ),
            reverse=True,
        )
        return self._daily_chat_self_anchor_text(candidates[0]) if candidates else ""

    @staticmethod
    def _active_self_anchor_bucket(bucket: dict) -> bool:
        if not is_self_anchor_bucket(bucket):
            return False
        meta = bucket.get("metadata", {}) if isinstance(bucket.get("metadata"), dict) else {}
        return bool(meta.get("active") is not False and not meta.get("deprecated") and not meta.get("resolved"))

    def _daily_chat_self_anchor_text(self, bucket: dict) -> str:
        content = strip_wikilinks(str(bucket.get("content") or "")).strip()
        if not content:
            return ""
        text = self._section_or_leading_text(
            content,
            headings={"自我", "self_anchor", "selfidentity", "self_identity", "first_person_anchor"},
        )
        if not text:
            text = content
        text = re.split(r"(?im)^\s{0,3}#{2,6}\s+(?:followup|todo)\b.*$", text, maxsplit=1)[0].strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:1200].rstrip()

    @staticmethod
    def _section_or_leading_text(content: str, *, headings: set[str]) -> str:
        matches = list(re.finditer(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", content))
        if not matches:
            return content.strip()
        leading = content[: matches[0].start()].strip()
        if leading:
            return leading
        normalized_headings = {re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", heading.lower()) for heading in headings}
        for index, match in enumerate(matches):
            heading = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(match.group(1) or "").lower())
            if heading not in normalized_headings:
                continue
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            return content[match.end() : end].strip()
        return ""

    def list_daily_chat_memory_pending(self, *, status: str = "pending", limit: int = 50) -> list[dict]:
        """Strictly read-only listing. Never mutates or persists anything.

        Opening/refreshing/leaving this page must not change candidate status;
        legacy items are only annotated for honest display, in memory.
        """
        safe_status = str(status or "pending").strip()
        safe_limit = max(1, min(200, int(limit or 50)))
        items = self._load_daily_chat_memory_pending()
        if safe_status and safe_status != "all":
            items = [item for item in items if str(item.get("status") or "") == safe_status]
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return [self._display_daily_chat_memory_item(item) for item in items[:safe_limit]]

    @staticmethod
    def _display_daily_chat_memory_item(item: dict) -> dict:
        """In-memory display projection. Purely derived; nothing is stored."""
        if not isinstance(item, dict):
            return item
        candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
        display: dict[str, object] = {}
        if candidate:
            has_excerpt = bool(str(candidate.get("original_excerpt") or "").strip())
            verified = str(candidate.get("source_verification") or "").strip() == "verified"
            has_source = bool(candidate.get("source_turn_ids") or candidate.get("source_event_ids"))
            excerpt = str(candidate.get("original_excerpt") or "")
            proposed = str(candidate.get("proposed_memory") or candidate.get("content") or "")
            has_proposed = bool(proposed.strip())
            # Health gates: legacy/unverifiable, unclean excerpt (internal control
            # markers leaked from a pre-fix run), or missing proposed memory.
            blocked_reasons: list[str] = []
            if not (verified and has_source):
                blocked_reasons.append("missing_source_verification")
            if "<" in excerpt or ">" in excerpt or re.search(r'\b(?:as|reason|mood|action|source|role|mode|tags|metadata|id|session|event_hash|client)="[^"]*"', excerpt):
                blocked_reasons.append("candidate_excerpt_unclean")
            if not has_proposed:
                blocked_reasons.append("missing_proposed_memory")
            display["legacy_no_original"] = not has_excerpt
            display["confirm_blocked"] = bool(blocked_reasons)
            display["blocked_reasons"] = blocked_reasons
            display["has_source_preview"] = has_source
            display["soft_flags"] = list(candidate.get("soft_flags") or [])
            display["needs_owner_edit"] = "needs_owner_edit" in (candidate.get("soft_flags") or [])
            display["confirm_blocked_reason"] = (
                "候选来源无法核对或原文含内部控制标记，无法批准，请拒绝。"
                if blocked_reasons
                else ""
            )
        return {**item, "display": display}

    async def daily_chat_memory_source_preview(
        self,
        candidate_id: str,
        *,
        raw_event_store=None,
        conversation_turn_store=None,
        profile_id: str = "default",
        source_kind: str = "",
        source_id: str = "",
        offset: int = 0,
        limit: int = 4000,
    ) -> dict:
        """Owner-only read of the complete sanitized source text behind a candidate.

        The full original is never stored in the candidate record; it stays pointed
        at by source_event_ids / source_turn_ids and is fetched here for the
        Dashboard "查看完整原文" expander. All returned text goes through the
        unified owner-visible sanitizer (no <silent>, no internal markers).

        Paging semantics: each source item carries its chunk (`text`), `truncated`,
        `full_length` and `continue_after`. A follow-up call with
        source_kind + source_id + offset returns the next chunk. Truncation is
        always explicit — never a silent fixed-length ellipsis."""
        rid = str(candidate_id or "").strip()
        if not rid:
            return {"status": "missing"}
        item = next(
            (
                candidate_item
                for candidate_item in self._load_daily_chat_memory_pending()
                if str(candidate_item.get("id") or "") == rid
            ),
            None,
        )
        if not item:
            return {"status": "missing"}
        candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
        event_ids = [int(v) for v in (candidate.get("source_event_ids") or []) if str(v).isdigit()]
        turn_ids = [int(v) for v in (candidate.get("source_turn_ids") or []) if str(v).isdigit()]
        date_key = str(candidate.get("date") or item.get("date") or "")
        safe_offset = max(0, int(offset or 0))
        safe_limit = max(256, min(20000, int(limit or 4000)))
        events: list[dict] = []
        turns: list[dict] = []

        def chunked(payload: str, item_offset: int) -> tuple[str, bool, int, int]:
            full_length = len(payload)
            if item_offset >= full_length:
                return "", False, full_length, -1
            end = min(full_length, item_offset + safe_limit)
            return payload[item_offset:end], end < full_length, full_length, end if end < full_length else -1

        if event_ids and raw_event_store:
            rows: list[dict] = []
            try:
                if date_key:
                    start, end = self._period_window("daily", self._daily_chat_memory_target(date_key))
                    rows = raw_event_store.list_events_between(
                        start_at=start,
                        end_at=end,
                        limit=10000,
                    )
            except Exception as exc:
                logger.warning("Daily chat memory source preview event read failed: %s", exc)
            wanted = set(event_ids)
            for event in rows or []:
                try:
                    event_id = int(event.get("id") or 0)
                except (TypeError, ValueError):
                    continue
                if event_id in wanted:
                    if source_kind == "event" and source_id and str(event_id) != str(source_id):
                        continue
                    full_text = self._daily_chat_memory_owner_text(str(event.get("text") or ""))
                    chunk, truncated, full_length, continue_after = chunked(full_text, safe_offset)
                    events.append(
                        {
                            "id": event_id,
                            "role": str(event.get("role") or ""),
                            "text": chunk,
                            "truncated": truncated,
                            "full_length": full_length,
                            "continue_after": continue_after,
                        }
                    )
            events.sort(key=lambda entry: entry["id"])
        if turn_ids and conversation_turn_store:
            rows = []
            try:
                if date_key:
                    start, end = self._period_window("daily", self._daily_chat_memory_target(date_key))
                    rows = conversation_turn_store.list_conversation_turns_between(
                        profile_id=str(profile_id or "default"),
                        start_at=start,
                        end_at=end,
                        limit=10000,
                    )
            except Exception as exc:
                logger.warning("Daily chat memory source preview turn read failed: %s", exc)
            wanted = set(turn_ids)
            for turn in rows or []:
                try:
                    turn_id = int(turn.get("id") or 0)
                except (TypeError, ValueError):
                    continue
                if turn_id in wanted:
                    if source_kind == "turn" and source_id and str(turn_id) != str(source_id):
                        continue
                    user_text = self._daily_chat_memory_owner_text(str(turn.get("user_text") or ""))
                    assistant_text = self._daily_chat_memory_owner_text(str(turn.get("assistant_text") or ""))
                    full_text = f"{user_text} {assistant_text}".strip()
                    chunk, truncated, full_length, continue_after = chunked(full_text, safe_offset)
                    turns.append(
                        {
                            "id": turn_id,
                            "session_id": str(turn.get("session_id") or ""),
                            "user_text": user_text,
                            "assistant_text": assistant_text,
                            "text": chunk,
                            "truncated": truncated,
                            "full_length": full_length,
                            "continue_after": continue_after,
                        }
                    )
            turns.sort(key=lambda entry: entry["id"])
        return {
            "status": "ok",
            "candidate_id": rid,
            "offset": safe_offset,
            "limit": safe_limit,
            "events": events,
            "turns": turns,
            "missing_event_ids": sorted(set(event_ids) - {entry["id"] for entry in events}),
            "missing_turn_ids": sorted(set(turn_ids) - {entry["id"] for entry in turns}),
        }

    async def confirm_daily_chat_memory(
        self,
        candidate_ids: list[str],
        bucket_mgr,
        *,
        embedding_engine=None,
        action: str = "confirm",
        edits: dict[str, dict] | None = None,
        request_id: str | None = None,
        reject_reason: str | None = None,
        reject_note: str | None = None,
        now: datetime | None = None,
    ) -> dict:
        ids = {str(candidate_id or "").strip() for candidate_id in candidate_ids if str(candidate_id or "").strip()}
        if not ids:
            return {
                "status": "skipped",
                "reason": "no_candidate_ids",
                "created": 0,
                "rejected": 0,
                "missing": 0,
                "request_id": "",
            }
        safe_action_raw = str(action or "").strip().lower()
        if safe_action_raw == "defer":
            safe_action = "defer"
        elif safe_action_raw == "reject":
            safe_action = "reject"
        else:
            safe_action = "confirm"
        safe_edits = edits if isinstance(edits, dict) else {}
        rid = str(request_id or "").strip() or hashlib.sha1(
            f"{safe_action}|{sorted(ids)}|{datetime.now(timezone.utc).isoformat(timespec='seconds')}".encode("utf-8")
        ).hexdigest()[:24]
        action_time = now or datetime.now(timezone.utc)

        # Idempotency: a repeated request_id returns the prior result without
        # re-applying and without consuming a rate-limit slot.
        prior = self._request_ledger_lookup(rid)
        if prior:
            return {**prior, "idempotent_replay": True, "request_id": rid}

        if not self._confirm_rate_limit_ok(action_time):
            return {
                "status": "rate_limited",
                "reason": "confirm_rate_limit_exceeded",
                "action": safe_action,
                "request_id": rid,
                "created": 0,
                "rejected": 0,
                "deferred": 0,
                "missing": 0,
                "results": [],
            }

        items = self._load_daily_chat_memory_pending()
        changed = False
        created = rejected = deferred = missing = 0
        results: list[dict] = []
        seen: set[str] = set()
        for item in items:
            item_id = str(item.get("id") or "").strip()
            if item_id not in ids:
                continue
            seen.add(item_id)
            if str(item.get("status") or "") != "pending":
                results.append({"id": item_id, "status": item.get("status") or "skipped"})
                continue
            if safe_action == "defer":
                # 暂缓: keep the item, remove it from the active pending view. It
                # is NOT a reject and can be reviewed again later.
                item["status"] = "deferred"
                item["deferred_at"] = action_time.isoformat(timespec="seconds")
                item["action_source"] = "owner"
                item["request_id"] = rid
                deferred += 1
                changed = True
                results.append({"id": item_id, "status": "deferred"})
                continue
            if safe_action == "reject":
                item["status"] = "rejected"
                item["rejected_at"] = action_time.isoformat(timespec="seconds")
                item["action_source"] = "owner"
                item["request_id"] = rid
                item["reject_reason"] = self._normalize_daily_chat_memory_reject_reason(reject_reason)
                if reject_note:
                    item["reject_note"] = re.sub(r"\s+", " ", str(reject_note or "").strip())[:120]
                rejected += 1
                changed = True
                results.append(
                    {
                        "id": item_id,
                        "status": "rejected",
                        "reject_reason": item["reject_reason"],
                    }
                )
                continue

            candidate = self._apply_daily_chat_memory_candidate_edit(
                dict(item.get("candidate") or {}),
                safe_edits.get(item_id),
            )
            candidate.update({"mode": "review", "status": "pending"})
            item["candidate"] = candidate
            changed = True
            if not self._daily_chat_memory_candidate_source_ok(candidate):
                # Legacy or damaged candidate: no verified original source, so it
                # must not be silently approved. It stays pending and is marked
                # stale for the owner to see; the owner may reject it.
                reason = (
                    "legacy_candidate_missing_source"
                    if str(candidate.get("source_verification") or "").strip() != "verified"
                    else "candidate_source_invalid"
                )
                item["stale"] = True
                item["stale_reason"] = reason
                results.append({"id": item_id, "status": "invalid_source", "reason": reason})
                continue
            soft_flags = set(candidate.get("soft_flags") or [])
            if "needs_owner_edit" in soft_flags and not safe_edits.get(item_id):
                # Wholesale-copy candidates must be rewritten by the owner before
                # they can be approved; they stay visible in Review.
                results.append(
                    {
                        "id": item_id,
                        "status": "needs_owner_edit",
                        "reason": "candidate_requires_edit",
                    }
                )
                continue
            write_result = await self._write_daily_chat_memory_candidates(
                [candidate],
                bucket_mgr,
                embedding_engine=embedding_engine,
            )
            candidate_result = (write_result.get("results") or [{}])[0]
            if candidate_result.get("status") in {"created", "exists"}:
                item["status"] = "confirmed"
                item["candidate"]["status"] = "confirmed"
                item["confirmed_at"] = action_time.isoformat(timespec="seconds")
                item["action_source"] = "owner"
                item["request_id"] = rid
                item["bucket_id"] = candidate_result.get("id") or item_id
                created += 1 if candidate_result.get("status") == "created" else 0
                changed = True
            results.append(candidate_result)
        missing = len(ids - seen)
        if changed:
            self._save_daily_chat_memory_pending(items)
        result = {
            "status": "ok",
            "action": safe_action,
            "created": created,
            "rejected": rejected,
            "deferred": deferred,
            "missing": missing,
            "request_id": rid,
            "candidate_ids": sorted(ids),
            "results": results,
        }
        self._request_ledger_record(rid, result)
        return result

    @staticmethod
    def _normalize_daily_chat_memory_reject_reason(value: Any) -> str:
        reason = str(value or "").strip().lower()
        return reason if reason in DAILY_CHAT_MEMORY_REJECT_REASONS else "other"

    @classmethod
    def _daily_chat_memory_candidate_source_ok(cls, candidate: dict) -> bool:
        if not isinstance(candidate, dict):
            return False
        if str(candidate.get("source_verification") or "").strip() != "verified":
            return False
        if not (candidate.get("source_turn_ids") or candidate.get("source_event_ids")):
            return False
        if not str(candidate.get("source_hash") or "").strip():
            return False
        excerpt = str(candidate.get("original_excerpt") or "")
        if "<" in excerpt or ">" in excerpt or cls._CONTROL_ATTR_RE.search(excerpt):
            # Malformed pre-fix candidates (internal control markers in the
            # excerpt) must not be silently approved.
            return False
        return True

    def _confirm_rate_limit_ok(self, now: datetime | None = None) -> bool:
        timestamp = (now or datetime.now(timezone.utc)).timestamp()
        window = self._confirm_rate_window
        cutoff = timestamp - 60.0
        while window and window[0] < cutoff:
            window.pop(0)
        if len(window) >= self.daily_chat_memory_confirm_rate_limit_per_minute:
            return False
        window.append(timestamp)
        return True

    def _request_ledger_load(self) -> list[dict]:
        try:
            with open(self.daily_chat_memory_requests_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def _request_ledger_lookup(self, request_id: str) -> dict | None:
        rid = str(request_id or "").strip()
        if not rid:
            return None
        for entry in reversed(self._request_ledger_load()):
            if str(entry.get("request_id") or "") == rid:
                return entry
        return None

    def _request_ledger_record(self, request_id: str, entry: dict) -> None:
        rid = str(request_id or "").strip()
        if not rid:
            return
        ledger = [item for item in self._request_ledger_load() if str(item.get("request_id") or "") != rid]
        entry = {
            "request_id": rid,
            "action": str(entry.get("action") or ""),
            "candidate_ids": sorted(str(item or "") for item in (entry.get("candidate_ids") or [])),
            "created": int(entry.get("created") or 0),
            "rejected": int(entry.get("rejected") or 0),
            "missing": int(entry.get("missing") or 0),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        ledger.append(entry)
        ledger = ledger[-500:]
        os.makedirs(os.path.dirname(self.daily_chat_memory_requests_path), exist_ok=True)
        tmp_path = self.daily_chat_memory_requests_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(ledger, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.daily_chat_memory_requests_path)
        except OSError as exc:
            logger.warning("Daily chat memory request ledger write failed: %s", exc)

    # ------------------------------------------------------------------
    # V4 run audit (append-only metadata) and run concurrency lock.
    # ------------------------------------------------------------------
    def _load_daily_chat_memory_run_audit(self) -> list[dict]:
        try:
            with open(self.daily_chat_memory_run_audit_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def _store_daily_chat_memory_run_audit(self, audit: dict) -> None:
        try:
            ledger = self._load_daily_chat_memory_run_audit()
            ledger.append(dict(audit))
            ledger = ledger[-500:]
            os.makedirs(os.path.dirname(self.daily_chat_memory_run_audit_path), exist_ok=True)
            tmp_path = self.daily_chat_memory_run_audit_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(ledger, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.daily_chat_memory_run_audit_path)
        except OSError as exc:
            logger.warning("Daily chat memory run audit write failed: %s", exc)

    def list_daily_chat_memory_run_audit(self, *, limit: int = 20) -> list[dict]:
        safe_limit = max(1, min(200, int(limit or 20)))
        return self._load_daily_chat_memory_run_audit()[-safe_limit:]

    def daily_chat_memory_run_cursor(self, profile_id: str = "default") -> dict:
        cursor = self._load_daily_chat_memory_cursor()
        raw = cursor.get("raw_events") if isinstance(cursor.get("raw_events"), dict) else {}
        entry = raw.get(self._daily_chat_memory_cursor_key(profile_id))
        if not isinstance(entry, dict):
            return {"last_raw_event_id": 0, "updated_at": ""}
        try:
            last_id = max(0, int(entry.get("last_raw_event_id") or 0))
        except (TypeError, ValueError):
            last_id = 0
        return {"last_raw_event_id": last_id, "updated_at": str(entry.get("updated_at") or "")}

    def _daily_chat_memory_run_lock_acquire(self) -> bool:
        if self._daily_chat_memory_run_lock:
            return False
        self._daily_chat_memory_run_lock = True
        return True

    def _daily_chat_memory_run_lock_release(self) -> None:
        self._daily_chat_memory_run_lock = False

    async def _extract_daily_chat_memory_candidates(
        self,
        key: str,
        turns: list[dict],
        *,
        self_context: str = "",
        window_summaries: list[dict] | None = None,
        max_candidates: int | None = None,
        audit: dict | None = None,
    ) -> tuple[list[dict], dict]:
        """V4 full-day windowed extraction.

        Every valid window of the day is actually checked (bounded model calls per
        window), covering the beginning / middle / end. Deterministic noise windows
        are skipped and recorded in the audit. The first window model failure stops
        the run (status=partial); the failed range is never faked as processed and
        the watermark only advances to the last successfully processed window.

        Returns (raw_candidates, meta) where meta carries:
        - partial: bool
        - last_processed_event_id: max raw event id fully processed so far
        - model_call_count / skipped_noise_window_count
        """
        client, model, use_daily_client = self._daily_chat_memory_model_client(candidate=True)
        meta: dict = {
            "partial": False,
            "last_processed_event_id": 0,
            "model_call_count": 0,
            "skipped_noise_window_count": 0,
            "window_failures": 0,
            "failed_window_index": None,
            "empty_output_count": 0,
            "parse_failure_count": 0,
        }
        if audit is not None:
            audit["window_count"] = 0
            audit["skipped_noise_window_count"] = 0
            audit["model_call_count"] = 0
            audit["window_failures"] = 0
            audit["empty_output_count"] = 0
            audit["parse_failure_count"] = 0
            audit["model"] = str(model or "")
        if not client:
            heuristic = self._heuristic_daily_chat_memory_candidates(
                key,
                turns,
                max_candidates=max_candidates,
            )
            return heuristic, meta
        windows = self._daily_chat_memory_extraction_windows(turns)
        if audit is not None:
            audit["window_count"] = len(windows)
        raw_candidates: list[dict] = []
        processed_event_id = 0
        for index, window in enumerate(windows):
            window_max_event = max(
                (int(v) for turn in window for v in (turn.get("raw_event_ids") or []) if str(v).isdigit()),
                default=0,
            )
            if self._daily_chat_memory_window_is_noise(window):
                processed_event_id = max(processed_event_id, window_max_event)
                meta["skipped_noise_window_count"] += 1
                if audit is not None:
                    audit["skipped_noise_window_count"] = meta["skipped_noise_window_count"]
                continue
            try:
                window_candidates, window_stats = await self._extract_window_candidates(
                    key,
                    window,
                    self_context=self_context,
                    max_candidates=max_candidates,
                    window_index=index + 1,
                    window_total=len(windows),
                )
                meta["model_call_count"] += 1
                if window_stats.get("empty_output"):
                    meta["empty_output_count"] += 1
                    logger.warning(
                        "Daily chat memory window %s/%s empty model output (finish_reason=%s)",
                        index + 1,
                        len(windows),
                        window_stats.get("finish_reason") or "unknown",
                    )
                if window_stats.get("parse_failure"):
                    meta["parse_failure_count"] += 1
                if audit is not None:
                    audit["model_call_count"] = meta["model_call_count"]
                    audit["empty_output_count"] = meta["empty_output_count"]
                    audit["parse_failure_count"] = meta["parse_failure_count"]
            except Exception as exc:
                logger.warning("Daily chat memory window %s extraction failed: %s", index + 1, exc)
                meta["partial"] = True
                meta["window_failures"] = 1
                meta["failed_window_index"] = index + 1
                if audit is not None:
                    audit["window_failures"] = 1
                    audit["partial"] = True
                break
            processed_event_id = max(processed_event_id, window_max_event)
            raw_candidates.extend(window_candidates)
        meta["last_processed_event_id"] = processed_event_id
        return raw_candidates, meta

    def _daily_chat_memory_extraction_windows(self, turns: list[dict]) -> list[list[dict]]:
        """Deterministic pre-filtered, bounded windows covering the whole day.

        Window size/stride adapt so the entire day fits within the per-run model
        call budget; the per-window input is also bounded by characters."""
        if not turns:
            return []
        size = self.daily_chat_memory_window_turns
        stride = self.daily_chat_memory_window_stride_turns
        max_windows = self.daily_chat_memory_max_windows_per_run
        total_len = sum(
            len(str(turn.get("user_text") or "")) + len(str(turn.get("assistant_text") or ""))
            for turn in turns
        )
        if len(turns) > size:
            needed = (len(turns) - size) // stride + 2
            if needed > max_windows:
                size = max(4, -(-len(turns) // max_windows))
                stride = max(1, size // 2)
        avg_len = max(1, total_len // max(1, len(turns)))
        char_bounded_size = max(
            4,
            min(size, self.daily_chat_memory_window_max_input_chars // max(1, avg_len)),
        )
        size = char_bounded_size
        stride = min(stride, size)
        windows: list[list[dict]] = []
        start = 0
        while start < len(turns):
            windows.append(turns[start : start + size])
            if start + size >= len(turns):
                break
            start += stride
        return windows

    @classmethod
    def _daily_chat_memory_window_is_noise(cls, window: list[dict]) -> bool:
        """Deterministic noise pre-filter: assistant-only chit-chat with no durable
        signal and no user statement is not worth a model call (recorded in audit)."""
        durable_markers = (
            "承诺", "约定", "答应", "以后", "边界", "暗号", "偏好", "希望",
            "不要", "不再", "部署", "决定", "正式", "一起", "约定好", "答应",
        )
        has_user_statement = False
        for turn in window or []:
            user_text = cls._daily_chat_memory_owner_text(str(turn.get("user_text") or ""))
            assistant_text = cls._daily_chat_memory_owner_text(str(turn.get("assistant_text") or ""))
            if user_text:
                has_user_statement = True
            if any(marker in (user_text + assistant_text) for marker in durable_markers):
                return False
        return not has_user_statement

    async def _extract_window_candidates(
        self,
        key: str,
        window: list[dict],
        *,
        self_context: str = "",
        max_candidates: int | None = None,
        window_index: int = 1,
        window_total: int = 1,
    ) -> tuple[list[dict], dict]:
        client, model, use_daily_client = self._daily_chat_memory_model_client(candidate=True)
        payload = {
            "date": key,
            "identity": {
                "ai_name": self.identity["ai_name"],
                "user_name": self.identity["user_name"],
                "user_display_name": self.identity["user_display_name"],
                "user_aliases": self.identity.get("user_aliases", []),
            },
            "self_anchor_entry": self_context,
            "window": {"index": window_index, "total": window_total},
            "conversation_turns": window,
        }
        response = await self._daily_chat_memory_create_completion(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": self._daily_chat_memory_prompt(max_candidates=max_candidates),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=self.daily_chat_memory_candidate_max_tokens,
            temperature=self.temperature,
            use_daily_client=use_daily_client,
        )
        raw = self._completion_content(response)
        stats: dict = {
            "empty_output": not str(raw or "").strip(),
            "parse_failure": False,
            "finish_reason": self._completion_finish_reason(response),
        }
        parsed = self._parse_json_object(raw or "")
        candidates = parsed.get("candidates") if isinstance(parsed, dict) else []
        if not isinstance(candidates, list):
            stats["parse_failure"] = bool(str(raw or "").strip())
            return [], stats
        return [item for item in candidates if isinstance(item, dict)], stats

    def _heuristic_daily_chat_memory_candidates(
        self,
        key: str,
        turns: list[dict],
        *,
        max_candidates: int | None = None,
    ) -> list[dict]:
        """Per-turn durable-signal matching with precise source ids.

        Candidates are only produced from a specific turn whose original text
        contains a durable signal; source ids point at that exact turn/event, never
        at the whole day. project_state requires both a project marker AND a
        decided/durable marker so ordinary debugging chatter stays out.
        """
        if not turns:
            return []
        project_markers = [
            "项目", "仓库", "分支", "部署", "MCP", "API", "网关",
            "自动记忆", "raw_events", "原文保险箱", "数据库", "镜像", "容器",
        ]
        decided_markers = [
            "已确认", "确定", "决定", "通过了", "最终", "正式", "上线",
            "保持", "沿用", "不再用", "不要改", "改回", "下一步", "之后", "以后",
            "已部署", "已上线", "已迁移", "已废弃", "已启用", "已停用", "已完成",
        ]
        keyword_map = [
            ("boundary", ["我不喜欢", "我不要", "以后不要", "别再", "边界是", "以后默认不要"]),
            ("signal", ["暗号是", "安全词是", "称呼我", "叫我", "模式是", "切换到", "以后都叫"]),
            ("commitment", ["承诺", "约定", "答应", "以后要", "下次要", "待办", "别忘了"]),
            (
                "project_state",
                [
                    "项目决定", "项目状态", "已部署", "已上线", "已迁移", "保持部署",
                    "自动记忆已", "部署完成", "迁移完成", "正式启用", "已废弃",
                ]
                + ["部署", "迁移", "分支", "镜像"],
            ),
            ("stable_preference", ["我希望以后", "我希望你", "以后解释", "默认先", "默认不要", "我的偏好", "以后默认"]),
            (
                "relationship_anchor",
                ["关系定位", "我们是", "正式在一起", "在一起了", "第一次见面", "搬来和我住", "关系变了"],
            ),
            (
                "self_insight",
                ["我发现自己", "我意识到", "我其实", "我原来", "我发现我", "我才知道我自己", "我明白了自己"],
            ),
            (
                "key_event",
                ["终于", "特别开心", "很感动", "值得记住", "重要的日子", "第一次做", "第一次一起"],
            ),
        ]
        candidates: list[dict] = []
        limit = int(max_candidates or self.daily_chat_memory_max_per_day or 1)
        kind_priority = [
            "signal",
            "commitment",
            "boundary",
            "project_state",
            "relationship_anchor",
            "self_insight",
            "stable_preference",
            "key_event",
        ]
        for turn in turns:
            turn_id = turn.get("id")
            raw_event_ids = [event_id for event_id in (turn.get("raw_event_ids") or []) if event_id is not None]
            has_turn_id = turn_id is not None and str(turn_id).isdigit()
            has_event_ids = bool(raw_event_ids)
            if not has_turn_id and not has_event_ids:
                continue
            if self._daily_chat_memory_turn_is_internal_dump(turn):
                continue
            user_text = self._daily_chat_memory_owner_text(str(turn.get("user_text") or ""))
            assistant_text = self._daily_chat_memory_owner_text(str(turn.get("assistant_text") or ""))
            if not user_text and not assistant_text:
                continue
            # V3: user statements are the primary signal source; assistant content
            # is only eligible for explicit long-term commitment / relationship /
            # stable boundary patterns (ordinary comfort replies never trigger).
            full_text = " ".join(text for text in (user_text, assistant_text) if text)
            matched_by_kind: dict[str, list[str]] = {}
            for kind, keywords in keyword_map:
                if kind in {"commitment", "relationship_anchor"}:
                    candidates_pool = full_text
                else:
                    candidates_pool = user_text
                matched = [kw for kw in keywords if kw in candidates_pool]
                if not matched:
                    continue
                if kind == "project_state":
                    has_project = any(marker in full_text for marker in project_markers)
                    has_decided = any(marker in full_text for marker in decided_markers)
                    if not has_project or not has_decided:
                        continue
                matched_by_kind[kind] = matched
            if not matched_by_kind:
                continue
            best_kind = max(
                matched_by_kind,
                key=lambda kind: (len(matched_by_kind[kind]), -kind_priority.index(kind)),
            )
            matched_keyword = matched_by_kind[best_kind][0]
            excerpt = self._daily_chat_memory_heuristic_excerpt(full_text, matched_keyword)
            core_sentence = self._daily_chat_memory_heuristic_core_sentence(full_text, matched_keyword) or excerpt
            content = self._daily_chat_memory_content(best_kind, key, core_sentence)
            if self._daily_chat_memory_noise(content):
                continue
            if self._daily_chat_memory_low_value_social_noise(content, best_kind):
                continue
            if self._daily_chat_memory_low_value_episode(content, best_kind):
                continue
            candidate = {
                "should_write": True,
                "kind": best_kind,
                "title": self._daily_chat_memory_title(content, best_kind, key),
                "content": content,
                "original_excerpt": excerpt,
                "generation_source": "heuristic",
                "domain": self._auto_memory_domain(best_kind, content, [self._kind_tag(best_kind)]),
                "tags": [self._kind_tag(best_kind)],
                "importance": 5,
                "valence": 0.58,
                "arousal": 0.3,
                "confidence": self.daily_chat_memory_heuristic_confidence,
                "source_turn_ids": [int(turn_id)] if has_turn_id else [],
                "source_event_ids": [int(v) for v in raw_event_ids[:24] if str(v).isdigit()],
                "reason": f"chat_contains_{best_kind}",
            }
            if not candidate["source_turn_ids"] and not candidate["source_event_ids"]:
                continue
            candidates.append(candidate)
            if len(candidates) >= limit:
                return candidates
        return candidates

    @classmethod
    def _daily_chat_memory_heuristic_excerpt(cls, text: str, matched_keyword: str, *, max_sentences: int = 3) -> str:
        """Complete-sentence source fragment starting at the sentence containing
        the matched keyword. Never slices at arbitrary character offsets."""
        sentences = cls._daily_chat_memory_sentences(text)
        if not sentences:
            return ""
        start = next((index for index, sentence in enumerate(sentences) if matched_keyword in sentence), 0)
        picked = [sentences[start]]
        for sentence in sentences[start + 1 :]:
            if len(picked) >= max_sentences:
                break
            picked.append(sentence)
        return " ".join(picked).strip()

    @classmethod
    def _daily_chat_memory_heuristic_core_sentence(cls, text: str, matched_keyword: str) -> str:
        """The single sentence carrying the matched signal, used as the dehydrated
        proposed-memory core for heuristic candidates."""
        for sentence in cls._daily_chat_memory_sentences(text):
            if matched_keyword in sentence:
                return sentence
        return ""

    def _daily_chat_memory_content(self, kind: str, key: str, excerpt: str) -> str:
        user_display_name = self.identity["user_display_name"]
        ai_name = self.identity["ai_name"]
        excerpt = self._memory_body_from_excerpt(excerpt)
        excerpt = re.sub(r"^(用户|助手)：", "", excerpt).strip()
        if kind == "key_event":
            return excerpt if excerpt else "关键事件需要后续回看。"
        if kind == "project_state":
            return excerpt if self._starts_with_identity(excerpt) else f"项目状态：{excerpt}"
        if kind == "relationship_anchor":
            return excerpt if self._starts_with_identity(excerpt) else f"{ai_name}记得这段关系锚点：{excerpt}"
        if kind == "self_insight":
            return excerpt if self._starts_with_identity(excerpt) else f"{user_display_name}的自我认识：{excerpt}"
        if kind == "boundary":
            return excerpt if self._starts_with_identity(excerpt) else f"{user_display_name}的边界：{excerpt}"
        if kind == "signal":
            return excerpt if self._starts_with_identity(excerpt) else f"{user_display_name}与{ai_name}的暗号或模式信号：{excerpt}"
        if kind == "commitment":
            return excerpt if self._starts_with_identity(excerpt) else f"后续需要记得的承诺或约定：{excerpt}"
        return excerpt if self._starts_with_identity(excerpt) else f"{user_display_name}的稳定偏好：{excerpt}"

    @staticmethod
    def _daily_chat_memory_noise(content: str) -> bool:
        text = re.sub(r"\s+", " ", strip_wikilinks(str(content or ""))).strip()
        if not text:
            return True
        lowered = text.lower()
        raw_snippet_markers = [
            "```",
            "query_cache",
            "recent_raw_context",
            "if query contains",
            "bypass query",
            "force recent",
        ]
        if any(marker in lowered for marker in raw_snippet_markers):
            return True
        noise_markers = [
            "笔友都有谁",
            "还记得吗",
            "记得吗",
            "我试试看",
            "试试看",
            "继续测",
            "测一下",
            "测试一下",
            "测试召回",
            "召回有没有",
            "有没有被注入",
            "被注入",
            "我直接问",
            "直接问",
            "看起来是否",
            "是否召回",
            "模型有没有",
            "chat_contains_",
        ]
        if any(marker in text for marker in noise_markers):
            return True
        if text.startswith("**"):
            return True
        if "?" in text or "？" in text:
            question_noise = ["谁", "有没有", "是否", "还", "吗", "怎么"]
            if any(marker in text for marker in question_noise):
                return True
        if "- **" in text or lowered.count("**") >= 4:
            return True
        return False

    @staticmethod
    def _daily_chat_memory_low_value_social_noise(content: str, kind: str) -> bool:
        text = re.sub(r"\s+", " ", strip_wikilinks(str(content or ""))).strip()
        if not text:
            return True
        affection_markers = [
            "称呼",
            "昵称",
            "亲昵称呼",
            "互动模式",
            "亲昵模式",
            "兴趣暗示",
            "参与意愿",
            "上瘾",
            "好奇",
            "叫哥哥",
            "叫老公",
            "叫老婆",
            "宝宝",
            "老婆",
            "哥哥",
            "老公",
            "小笨蛋",
            "小坏蛋",
        ]
        if not any(marker in text for marker in affection_markers):
            return False
        durable_markers = [
            "明确约定",
            "约定",
            "承诺",
            "边界",
            "不要",
            "不能",
            "必须",
            "以后要",
            "以后不要",
            "只在",
            "只有",
            "暗号",
            "安全词",
            "模式切换",
            "切换",
            "关系定位",
            "身份定位",
            "规则",
        ]
        if any(marker in text for marker in durable_markers):
            return False
        low_value_markers = [
            "互动模式",
            "亲昵模式",
            "称呼",
            "昵称",
            "期待",
            "像人一样",
            "像真人",
            "兴趣暗示",
            "参与意愿",
            "上瘾",
            "好奇",
            "宝宝",
            "老婆",
            "哥哥",
            "老公",
        ]
        return kind in {"signal", "relationship_anchor", "stable_preference"} and any(
            marker in text for marker in low_value_markers
        )

    @staticmethod
    def _daily_chat_memory_low_value_episode(content: str, kind: str, title: str = "") -> bool:
        text = re.sub(
            r"\s+",
            " ",
            strip_wikilinks(" ".join([str(title or ""), str(content or "")])),
        ).strip()
        if not text:
            return True
        compact = re.sub(r"\s+", "", text)
        lowered = text.lower()
        shell_markers = [
            "后续需要记得的承诺或约定：比如",
            "后续需要记得的承诺或约定:比如",
            "果然没触发，可能是",
            "果然没触发,可能是",
            "可能是触发条件",
            "触发条件卡在",
        ]
        if any(marker in compact for marker in shell_markers):
            return True
        if kind == "signal" and "触发" in text and "暗号" not in text and "模式切换" not in text:
            return True
        uncertain_markers = ["可能是", "似乎", "或许", "也许", "大概是"]
        if any(marker in text for marker in uncertain_markers):
            concrete_project_markers = [
                "已验证",
                "待验证",
                "需要验证",
                "下一步",
                "部署",
                "回归",
                "修复",
                "自动记忆",
                "Ombre",
                "Bridge",
                "bridge",
                "future_wake",
                "raw_events",
                "MCP",
                "API",
            ]
            if not any(marker in text for marker in concrete_project_markers):
                return True
        care_markers = [
            "熬夜",
            "别再熬",
            "别熬",
            "不睡",
            "早睡",
            "晚安",
            "睡了",
            "睡觉",
            "吃药",
            "喝药",
            "七天药",
            "ntfy",
            "轰炸",
        ]
        if any(marker.lower() in lowered or marker in text for marker in care_markers):
            durable_markers = [
                "明确约定",
                "稳定规则",
                "长期规则",
                "固定规则",
                "以后默认",
                "必须",
                "健康计划",
                "医嘱",
            ]
            if not any(marker in text for marker in durable_markers):
                return True
        generic_commitment_markers = ["比如", "例如", "或者"]
        if kind == "commitment" and sum(1 for marker in generic_commitment_markers if marker in text) >= 2:
            return True
        rawish_markers = ["下次再", "我直接", "你直接", "那一层了", "之后没人说话"]
        if any(marker in text for marker in rawish_markers) and kind in {
            "boundary",
            "commitment",
            "signal",
            "relationship_anchor",
        }:
            return True
        return False

    @staticmethod
    def _daily_chat_memory_similarity_tokens(value: str) -> set[str]:
        text = re.sub(r"\s+", " ", strip_wikilinks(str(value or ""))).lower()
        tokens = set(re.findall(r"[a-z][a-z0-9_.:/-]{2,}", text))
        cjk_stop = {"这个", "那个", "今天", "聊天", "记得", "确认", "已经", "需要", "以后", "可以", "通过"}
        for block in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            if 2 <= len(block) <= 6 and block not in cjk_stop:
                tokens.add(block)
            for size in (2, 3):
                for index in range(0, max(0, len(block) - size + 1)):
                    token = block[index : index + size]
                    if token not in cjk_stop:
                        tokens.add(token)
        return {token for token in tokens if len(token) >= 2}

    @staticmethod
    def _daily_chat_memory_token_overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(1, min(len(left), len(right)))

    @staticmethod
    def _daily_chat_memory_topic_keys(item: dict) -> set[str]:
        text = " ".join(
            [
                str(item.get("title") or ""),
                str(item.get("content") or ""),
                " ".join(str(tag or "") for tag in (item.get("tags") or [])),
            ]
        ).lower()
        keys: set[str] = set()
        if "钓鱼" in text and ("mcp" in text or "项目" in text or "部署" in text):
            keys.add("钓鱼项目")
        if "ai-fishing-game" in text or "tutusagi/ai-fishing-game" in text:
            keys.add("钓鱼项目")
        if "cx33" in text and ("钓鱼" in text or "mcp" in text):
            keys.add("钓鱼项目")
        if "笔友" in text and ("名册" in text or "名单" in text or "都有谁" in text):
            keys.add("笔友名册")
        return keys

    def _daily_chat_memory_duplicate_candidate(self, item: dict, existing_items: list[dict]) -> bool:
        new_title_tokens = self._daily_chat_memory_similarity_tokens(str(item.get("title") or ""))
        new_content_tokens = self._daily_chat_memory_similarity_tokens(str(item.get("content") or ""))
        new_sources = set(item.get("source_event_ids") or []) or set(item.get("source_turn_ids") or [])
        new_topics = self._daily_chat_memory_topic_keys(item)
        for existing in existing_items:
            same_kind = item.get("kind") == existing.get("kind")
            same_date = str(item.get("date") or "") == str(existing.get("date") or "")
            if same_date and new_topics and (new_topics & self._daily_chat_memory_topic_keys(existing)):
                return True
            # Same source turns/events = literally the same conversation, no
            # matter which day the extractor happened to pick it up on. This is
            # the reliable cross-day signal; wording drifts, turn ids do not.
            if new_sources:
                existing_sources = (set(existing.get("source_event_ids") or [])
                                    or set(existing.get("source_turn_ids") or []))
                if existing_sources and new_sources & existing_sources:
                    return True
            title_overlap = self._daily_chat_memory_token_overlap(
                new_title_tokens,
                self._daily_chat_memory_similarity_tokens(str(existing.get("title") or "")),
            )
            content_overlap = self._daily_chat_memory_token_overlap(
                new_content_tokens,
                self._daily_chat_memory_similarity_tokens(str(existing.get("content") or "")),
            )
            existing_sources = set(existing.get("source_event_ids") or []) or set(existing.get("source_turn_ids") or [])
            source_overlap = 0.0
            if new_sources and existing_sources:
                source_overlap = len(new_sources & existing_sources) / max(1, min(len(new_sources), len(existing_sources)))
            if same_kind and source_overlap >= 0.5 and content_overlap >= 0.3:
                return True
            if same_kind and title_overlap >= 0.55 and content_overlap >= 0.35:
                return True
            if content_overlap >= 0.7:
                return True
        return False

    @staticmethod
    def _daily_chat_memory_title_is_generic(title: str) -> bool:
        text = re.sub(r"\s+", " ", strip_wikilinks(str(title or ""))).strip()
        if not text:
            return True
        generic_markers = [
            "自动记忆",
            "每日记忆",
            "日记补记忆",
            "可召回",
            "短标题",
            "长期记忆",
            "聊天里",
            "的聊天",
            "暗号或模式信号",
        ]
        if any(marker in text for marker in generic_markers):
            return True
        if re.search(r"\d{4}-\d{1,2}-\d{1,2}", text):
            return True
        return False

    def _daily_chat_memory_title(self, content: str, kind: str, key: str) -> str:
        text = re.sub(r"#+\s*(moment|original|reflection|todo|affect_anchor).*", "", str(content or ""), flags=re.I | re.S)
        text = strip_wikilinks(text)
        date_prefix_pattern = r"^\d{4}-\d{2}-\d{2}\s*(发生了|的聊天里|确认了|留下|自动记忆)?"
        text = re.sub(date_prefix_pattern, "", text).strip(" ：:，,。")
        identity_names_for_title = [
            self.identity.get("user_display_name"),
            self.identity.get("user_name"),
            self.identity.get("ai_name"),
            *(self.identity.get("user_aliases") or []),
        ]
        identity_prefixes = []
        for name in identity_names_for_title:
            clean_name = str(name or "").strip()
            if clean_name and clean_name not in identity_prefixes:
                identity_prefixes.extend(
                    [
                        f"{clean_name}在",
                        f"{clean_name} 在",
                        f"{clean_name}希望",
                        f"{clean_name}的边界",
                        f"{clean_name}的稳定偏好",
                        f"{clean_name}的偏好",
                        f"{clean_name}说",
                    ]
                )
        prefixes = [
            *identity_prefixes,
            "这次聊天确认了",
            "这次聊天里留下",
            "一个仍会影响后续执行的项目状态",
            "一个后续需要记得的承诺或约定",
        ]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip(" ：:，,。")
                text = re.sub(date_prefix_pattern, "", text).strip(" ：:，,。")
        if not text:
            text = self._kind_label(kind)
        text = re.split(r"[。！？!?；;\n]", text, maxsplit=1)[0].strip(" ：:，,。")
        text = re.sub(r"\s+", " ", text)
        if len(text) > 24:
            text = text[:24].rstrip(" ：:，,。")
        if len(text) < 4:
            text = self._kind_label(kind)
        return text

    @staticmethod
    def _daily_chat_memory_tag_is_structural(value: Any) -> bool:
        term = re.sub(r"\s+", " ", str(value or "").strip()).lower()
        if not term:
            return True
        match = re.match(r"^([a-z_][a-z0-9_-]*)\s*[:：]\s*(.+)$", term)
        if match:
            term = match.group(2).strip().lower()
        return term in DAILY_CHAT_MEMORY_STRUCTURAL_TAGS

    @staticmethod
    def _daily_chat_memory_word_map_term_blocked(value: Any) -> bool:
        term = re.sub(r"\s+", " ", str(value or "").strip()).lower()
        if not term:
            return True
        compact = re.sub(r"[\s_-]+", "", term)
        for blocked in DAILY_CHAT_MEMORY_WORD_MAP_BLOCK_TERMS:
            blocked_text = str(blocked).lower()
            blocked_compact = re.sub(r"[\s_-]+", "", blocked_text)
            if blocked_text in term or (blocked_compact and blocked_compact in compact):
                return True
        return False

    @staticmethod
    def _daily_chat_memory_contains(text: str, needle: str) -> bool:
        if not needle:
            return False
        if re.fullmatch(r"[A-Za-z0-9_.:/ -]+", needle):
            return needle.lower() in text.lower()
        return needle in text

    def _daily_chat_memory_semantic_tags_and_keywords(
        self,
        *,
        title: str,
        content: str,
        tags: list[str],
    ) -> tuple[list[str], list[str]]:
        text = " ".join([str(title or ""), str(content or ""), " ".join(tags)])
        semantic_tags: list[str] = []
        keywords: list[str] = []

        def add_tag(prefix: str, term: str) -> None:
            term = re.sub(r"\s+", " ", str(term or "").strip())
            if (
                not term
                or self._daily_chat_memory_tag_is_structural(term)
                or self._daily_chat_memory_word_map_term_blocked(term)
            ):
                return
            semantic_tags.append(f"{prefix}:{term}")
            keywords.append(term)

        for raw in tags:
            tag = re.sub(r"\s+", " ", str(raw or "").strip())
            if not tag or self._daily_chat_memory_tag_is_structural(tag):
                continue
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*[:：]\s*(.+)$", tag)
            if match and match.group(1).strip().lower() in {"axis", "content", "entity", "topic"}:
                add_tag(match.group(1).strip().lower(), match.group(2))
            else:
                keywords.append(tag)

        for term, needles in DAILY_CHAT_MEMORY_ENTITY_HINTS:
            if any(self._daily_chat_memory_contains(text, needle) for needle in needles):
                add_tag("entity", term)
        for term, needles in DAILY_CHAT_MEMORY_TOPIC_HINTS:
            if any(self._daily_chat_memory_contains(text, needle) for needle in needles):
                add_tag("topic", term)

        title_term = re.sub(r"\s+", " ", str(title or "").strip(" ：:，,。"))
        if (
            4 <= len(title_term) <= 18
            and not self._daily_chat_memory_title_is_generic(title_term)
            and not self._daily_chat_memory_tag_is_structural(title_term)
            and not self._daily_chat_memory_word_map_term_blocked(title_term)
        ):
            keywords.append(title_term)

        semantic_tags = list(dict.fromkeys(semantic_tags))[:6]
        keywords = list(
            dict.fromkeys(
                keyword
                for keyword in keywords
                if keyword
                and not self._daily_chat_memory_tag_is_structural(keyword)
                and not self._daily_chat_memory_word_map_term_blocked(keyword)
            )
        )[:10]
        return semantic_tags, keywords

    def _daily_chat_memory_enrich_candidate_terms(self, candidate: dict) -> dict:
        updated = dict(candidate)
        kind = str(updated.get("kind") or "key_event").strip()
        raw_tags = self._string_list(updated.get("tags"), limit=16)
        semantic_tags, keywords = self._daily_chat_memory_semantic_tags_and_keywords(
            title=str(updated.get("title") or ""),
            content=str(updated.get("content") or ""),
            tags=raw_tags,
        )
        preserved_tags = [tag for tag in raw_tags if not self._daily_chat_memory_tag_is_structural(tag)]
        updated["tags"] = list(
            dict.fromkeys(
                [
                    "from_daily_chat",
                    "daily_chat_extract",
                    kind,
                    self._kind_tag(kind),
                    *semantic_tags,
                    *preserved_tags,
                ]
            )
        )[:12]
        existing_keywords = self._string_list(updated.get("keywords"), limit=12)
        updated["keywords"] = list(dict.fromkeys([*keywords, *existing_keywords]))[:12]
        return updated

    def _normalize_daily_chat_memory_candidates(
        self,
        key: str,
        candidates: list[dict],
        turns: list[dict],
        *,
        max_candidates: int | None = None,
        min_confidence: float | None = None,
        mode: str = "review",
        audit: dict | None = None,
    ) -> list[dict]:
        valid_turn_ids, valid_event_ids = self._daily_chat_memory_valid_source_maps(turns)
        normalized = []
        effective_mode = self._normalize_daily_chat_memory_mode(mode or "review")
        # History is compared with status so exact duplicates are suppressed but
        # similar-to-rejected candidates only get a soft warning (no permanent
        # blackhole), and similar-to-confirmed candidates get possible_duplicate.
        history_records: list[dict] = [
            record for record in (self._load_daily_chat_memory_payload().get("items") or [])
            if isinstance(record, dict)
        ]
        history_candidates: list[tuple[dict, str]] = []
        for record in history_records:
            candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else record
            if isinstance(candidate, dict):
                history_candidates.append((candidate, str(record.get("status") or "").strip().lower()))
        history_only = [candidate for candidate, _status in history_candidates]

        hard_reject_counts: dict[str, int] = {}
        if audit is not None:
            audit["hard_rejects"] = hard_reject_counts
            audit["soft_flags"] = {}
            audit["merged_duplicates"] = 0

        def record_hard(reason: str) -> None:
            hard_reject_counts[reason] = hard_reject_counts.get(reason, 0) + 1

        def record_soft(flag: str) -> None:
            if audit is not None:
                flags = audit["soft_flags"]
                flags[flag] = flags.get(flag, 0) + 1

        for candidate in candidates or []:
            if candidate.get("should_write") is False:
                continue
            candidate_tags = self._string_list(candidate.get("tags"), limit=8)
            content = self._trim_daily_chat_memory_content(str(candidate.get("content") or "").strip())
            if not content:
                # proposed_memory empty: hard reject (invalid structure).
                record_hard("empty_proposed_memory")
                continue
            if self._daily_chat_memory_noise(content):
                record_hard("system_noise_content")
                continue
            kind = self._normalize_auto_memory_kind(
                candidate.get("kind"),
                content=content,
                tags=candidate_tags,
            )
            if not kind or kind == "love_letter":
                record_hard("invalid_candidate_type")
                continue
            title = str(candidate.get("title") or "").strip()
            confidence = self._clamp(candidate.get("confidence", 0.0))
            # A tiny absolute floor is structural; everything above it is decided
            # by the mode gate below.
            if confidence < 0.1:
                record_hard("invalid_confidence")
                continue
            domain = self._auto_memory_domain(kind, content, candidate_tags, candidate.get("domain"))
            # Strict source resolution: ids must exist in this day's real turns.
            raw_turn_ids = [
                int(turn_id)
                for turn_id in self._string_list(candidate.get("source_turn_ids"), limit=20)
                if str(turn_id).isdigit()
            ]
            raw_event_ids = [
                int(event_id)
                for event_id in self._string_list(candidate.get("source_event_ids"), limit=80)
                if str(event_id).isdigit()
            ]
            source_turn_ids = sorted({turn_id for turn_id in raw_turn_ids if turn_id in valid_turn_ids})
            source_event_ids = sorted({event_id for event_id in raw_event_ids if event_id in valid_event_ids})
            if not source_turn_ids and not source_event_ids:
                record_hard("missing_or_fabricated_source")
                continue
            source_turns = self._daily_chat_memory_turns_by_ids(
                turns,
                source_turn_ids,
                source_event_ids,
            )
            if not source_turns:
                record_hard("source_not_found")
                continue
            # V3/V4: only clean, owner-visible source turns may support a candidate.
            source_turns = [
                turn
                for turn in source_turns
                if not self._daily_chat_memory_turn_is_internal_dump(turn)
            ]
            clean_turns = [
                turn
                for turn in source_turns
                if self._daily_chat_memory_turn_has_clean_text(turn)
            ]
            if not clean_turns:
                record_hard("no_clean_source")
                continue
            original_excerpt = self._daily_chat_memory_excerpt_for_turns(candidate, clean_turns)
            if not original_excerpt:
                record_hard("no_clean_source_sentence")
                continue
            generation_source = str(candidate.get("generation_source") or "model").strip().lower() or "model"
            if not self._daily_chat_memory_source_supports_kind(kind, clean_turns):
                # candidate_type must be supported by the source; ordinary comfort
                # replies must not become boundary/key_event/stable_preference.
                record_hard("candidate_type_not_supported_by_source")
                continue
            source_hash = self._daily_chat_memory_source_hash_for_turns(clean_turns)
            # ---- V4 soft flags: Review keeps these visible for the owner ----
            soft_flags: list[str] = []
            if generation_source == "model" and self._daily_chat_memory_proposed_echoes_excerpt(
                content,
                original_excerpt,
            ):
                # Wholesale copy: keep in Review as needs-editing, never approve
                # until the owner rewrites the proposed memory.
                soft_flags += ["excerpt_overlap", "needs_owner_edit"]
            if self._daily_chat_memory_source_is_weak(kind, clean_turns):
                soft_flags.append("weak_source_support")
            if self._daily_chat_memory_low_value_social_noise(content, kind):
                soft_flags.append("possibly_transient")
            if self._daily_chat_memory_low_value_episode(content, kind, title):
                soft_flags += ["possibly_transient", "possibly_generic"]
            if self._daily_chat_memory_title_is_generic(title):
                soft_flags.append("possibly_generic")
                title = self._daily_chat_memory_title(content, kind, key)
            mode_threshold = self.daily_chat_memory_min_confidence if min_confidence is None else min_confidence
            if confidence < mode_threshold:
                if effective_mode == "auto":
                    # Auto is strict: below the auto threshold is a hard reject.
                    record_hard("below_auto_confidence")
                    continue
                # Review keeps low-confidence candidates visible with a flag.
                soft_flags.append("low_confidence")
            candidate_id = self._daily_chat_memory_candidate_id(key, kind, content)
            # Exact duplicates (same identity / same source_hash+kind) are
            # suppressed idempotently; similar items only warn.
            if self._daily_chat_memory_exact_duplicate(
                {"id": candidate_id, "kind": kind, "source_hash": source_hash},
                history_only,
            ):
                record_hard("exact_duplicate")
                continue
            similar_status = self._daily_chat_memory_similar_history_status(
                {
                    "kind": kind,
                    "date": key,
                    "title": title,
                    "content": content,
                    "source_turn_ids": source_turn_ids,
                    "source_event_ids": source_event_ids,
                },
                history_candidates,
            )
            if similar_status == "rejected":
                soft_flags.append("previously_rejected_similar")
            elif similar_status == "confirmed":
                soft_flags.append("possible_duplicate")
            soft_flags = list(dict.fromkeys(soft_flags))
            item = self._daily_chat_memory_enrich_candidate_terms({
                "id": candidate_id,
                "date": key,
                "kind": kind,
                "candidate_type": kind,
                "title": title[:40],
                "content": content,
                "proposed_memory": content,
                "original_excerpt": original_excerpt,
                "generation_source": generation_source,
                "source_hash": source_hash,
                "source_verification": "verified",
                "soft_flags": soft_flags,
                "tags": candidate_tags,
                "keywords": self._string_list(candidate.get("keywords"), limit=12),
                "domain": domain,
                "importance": max(5, min(6, self._int_between(candidate.get("importance"), 5))),
                "valence": self._clamp(candidate.get("valence", 0.55)),
                "arousal": self._clamp(candidate.get("arousal", 0.3)),
                "confidence": confidence,
                "source_turn_ids": source_turn_ids[:80],
                "source_event_ids": source_event_ids[:160],
                "reason": str(candidate.get("reason") or "").strip()[:160],
            })
            for flag in soft_flags:
                record_soft(flag)
            existing_duplicate = next(
                (
                    existing
                    for existing in normalized
                    if self._daily_chat_memory_duplicate_candidate(item, [existing])
                ),
                None,
            )
            if existing_duplicate is not None:
                # Overlapping windows produced the same candidate: merge into the
                # existing one, keeping the most complete source references.
                self._daily_chat_memory_merge_sources(existing_duplicate, item, turns)
                if audit is not None:
                    audit["merged_duplicates"] = audit.get("merged_duplicates", 0) + 1
                continue
            normalized.append(item)
            if len(normalized) >= int(max_candidates or self.daily_chat_memory_max_per_day or 1):
                break
        return normalized

    @staticmethod
    def _daily_chat_memory_exact_duplicate(item: dict, history: list[dict]) -> bool:
        """Idempotent suppression: same candidate identity or same source_hash+kind."""
        item_id = str(item.get("id") or "").strip()
        item_hash = str(item.get("source_hash") or "").strip()
        item_kind = str(item.get("kind") or "")
        for existing in history:
            if item_id and str(existing.get("id") or "").strip() == item_id:
                return True
            existing_hash = str(existing.get("source_hash") or "").strip()
            if (
                item_hash
                and existing_hash == item_hash
                and item_kind
                and str(existing.get("kind") or "") == item_kind
            ):
                return True
        return False

    def _daily_chat_memory_similar_history_status(
        self,
        item: dict,
        history: list[tuple[dict, str]],
    ) -> str:
        """Return 'rejected' / 'confirmed' when a similar (not exact) history item
        exists, otherwise ''. Similar-to-rejected only warns (no blackhole)."""
        for existing, status in history:
            if status not in {"rejected", "confirmed"}:
                continue
            if self._daily_chat_memory_duplicate_candidate(item, [existing]):
                return status
        return ""

    @classmethod
    def _daily_chat_memory_source_is_weak(cls, kind: str, turns: list[dict]) -> bool:
        """Assistant-only support or a very short single user utterance means the
        source support for the candidate type is weak (soft warning, still kept)."""
        user_texts = [cls._daily_chat_memory_owner_text(str(turn.get("user_text") or "")) for turn in turns or []]
        if not any(text for text in user_texts):
            return True
        if sum(len(text) for text in user_texts) <= 20:
            return True
        return False

    def _daily_chat_memory_merge_sources(
        self,
        target: dict,
        incoming: dict,
        turns: list[dict],
    ) -> dict:
        """Merge a duplicate candidate into the existing one, keeping the most
        complete source references (union of turn/event ids, recomputed hash)."""
        turn_ids = sorted(set(target.get("source_turn_ids") or []) | set(incoming.get("source_turn_ids") or []))[:80]
        event_ids = sorted(set(target.get("source_event_ids") or []) | set(incoming.get("source_event_ids") or []))[:160]
        target["source_turn_ids"] = turn_ids
        target["source_event_ids"] = event_ids
        source_turns = self._daily_chat_memory_turns_by_ids(turns, turn_ids, event_ids)
        if source_turns:
            target["source_hash"] = self._daily_chat_memory_source_hash_for_turns(source_turns)
        for field in ("proposed_memory", "content", "original_excerpt", "reason"):
            incoming_text = str(incoming.get(field) or "")
            if len(incoming_text) > len(str(target.get(field) or "")):
                target[field] = incoming_text
        return target

    @staticmethod
    def _daily_chat_memory_valid_source_maps(turns: list[dict]) -> tuple[set[int], set[int]]:
        valid_turn_ids: set[int] = set()
        valid_event_ids: set[int] = set()
        for turn in turns or []:
            turn_id = turn.get("id")
            if turn_id is not None and str(turn_id).isdigit():
                valid_turn_ids.add(int(turn_id))
            for event_id in (turn.get("raw_event_ids") or []):
                if event_id is not None and str(event_id).isdigit():
                    valid_event_ids.add(int(event_id))
        return valid_turn_ids, valid_event_ids

    @staticmethod
    def _daily_chat_memory_turns_by_ids(
        turns: list[dict],
        turn_ids: list[int],
        event_ids: list[int],
    ) -> list[dict]:
        wanted_turn_ids = set(turn_ids)
        wanted_event_ids = set(event_ids)
        matched: list[dict] = []
        for turn in turns or []:
            turn_id = turn.get("id")
            turn_matches = turn_id is not None and str(turn_id).isdigit() and int(turn_id) in wanted_turn_ids
            event_matches = bool(
                wanted_event_ids
                and {int(v) for v in (turn.get("raw_event_ids") or []) if str(v).isdigit()} & wanted_event_ids
            )
            if turn_matches or event_matches:
                matched.append(turn)
        matched.sort(key=lambda item: (str(item.get("created_at") or ""), int(item.get("id") or 0)))
        return matched

    # ------------------------------------------------------------------
    # V3 owner-visible source helpers: sanitizer, sentence boundaries,
    # internal-dump detection, complete-sentence excerpt selection.
    # ------------------------------------------------------------------
    _SILENT_BLOCK_RE = re.compile(r"<silent\b[^>]*>.*?</silent>", re.DOTALL | re.IGNORECASE)
    _TAG_RE = re.compile(r"<[^>]+>")
    _CONTROL_ATTR_RE = re.compile(
        r'\b(?:as|reason|mood|action|role|source|mode|tags|metadata|id|session|event_hash|client)="[^"]*"',
        re.IGNORECASE,
    )
    _MEDIA_ARTIFACT_RE = re.compile(r"\[(?:语音|图片|文件|贴纸|表情)[^\]]*\]")
    _SENTENCE_END_RE = re.compile(r"(?<=[。！？!?…；;])")
    _DUMP_MARKERS = ("近期素材", "[app/bridge]", "今天的聊天:", "今天的聊天：", "[telegram]")
    _COMFORT_MARKERS = (
        "别难过", "别伤心", "抱抱", "没事的", "没关系", "辛苦了", "别担心",
        "我在呢", "会一直陪着", "守着你", "别哭", "乖", "摸摸头", "放心",
    )
    _DURABLE_ASSISTANT_MARKERS = (
        "承诺", "约定", "答应", "会一直", "永远", "记住", "关系", "正式在一起",
        "以后默认", "不会改", "保持", "边界", "不要再",
    )

    @classmethod
    def _daily_chat_memory_owner_text(cls, text: str) -> str:
        """Unified owner-visible sanitizer.

        Removes <silent ...>...</silent> blocks (hidden commentary), any other
        angle-bracket control tags, internal attribute fragments (as= / reason= /
        mood= / action= ...), and media render artifacts ([语音:...] ...). The
        result is exactly what the owner sees in the chat.
        """
        cleaned = str(text or "")
        cleaned = cls._SILENT_BLOCK_RE.sub(" ", cleaned)
        cleaned = cls._TAG_RE.sub(" ", cleaned)
        cleaned = cls._CONTROL_ATTR_RE.sub(" ", cleaned)
        cleaned = cls._MEDIA_ARTIFACT_RE.sub(" ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _daily_chat_memory_sentences(cls, text: str) -> list[str]:
        """Split sanitized text into complete sentences (Chinese/English/line ends)."""
        cleaned = cls._daily_chat_memory_owner_text(text)
        if not cleaned:
            return []
        sentences: list[str] = []
        for chunk in cls._SENTENCE_END_RE.split(cleaned):
            for line in chunk.splitlines():
                sentence = " ".join(line.split()).strip()
                if sentence:
                    sentences.append(sentence)
        return sentences

    @classmethod
    def _daily_chat_memory_turn_is_internal_dump(cls, turn: dict) -> bool:
        """Internal material-dump turns (e.g. "近期素材：今天的聊天 [app/bridge]")
        are not real owner-visible conversation and must never be a source."""
        for key in ("user_text", "assistant_text"):
            raw = str(turn.get(key) or "")
            if any(marker in raw for marker in cls._DUMP_MARKERS):
                return True
        return False

    @classmethod
    def _daily_chat_memory_turn_has_clean_text(cls, turn: dict) -> bool:
        for key in ("user_text", "assistant_text"):
            if cls._daily_chat_memory_owner_text(str(turn.get(key) or "")):
                return True
        return False

    def _daily_chat_memory_excerpt_for_turns(
        self,
        candidate: dict,
        turns: list[dict],
        *,
        max_sentences: int = 3,
        max_chars: int = 240,
    ) -> str:
        """Select 1-3 complete, clean sentences that best support the proposed
        memory. Never slices at arbitrary character offsets; every excerpt starts
        and ends at a sentence boundary."""
        if not turns:
            return ""
        probe = re.sub(r"\s+", " ", str(candidate.get("content") or "")).strip()
        probe_tokens: set[str] = set()
        for block in re.findall(r"[\u4e00-\u9fff]{2,16}", probe):
            for size in (2, 3, 4):
                for index in range(0, max(0, len(block) - size + 1)):
                    probe_tokens.add(block[index : index + size])
        probe_tokens = sorted(probe_tokens, key=len, reverse=True)
        sentences_by_turn: list[list[str]] = []
        all_sentences: list[str] = []
        for turn in turns:
            turn_sentences: list[str] = []
            for key in ("user_text", "assistant_text"):
                for sentence in self._daily_chat_memory_sentences(str(turn.get(key) or "")):
                    if sentence not in all_sentences:
                        all_sentences.append(sentence)
                        turn_sentences.append(sentence)
            sentences_by_turn.append(turn_sentences)

        def pick_from(start_index: int) -> str:
            picked: list[str] = []
            total_chars = 0
            for index in range(start_index, len(all_sentences)):
                if len(picked) >= max_sentences:
                    break
                sentence = all_sentences[index]
                if not sentence:
                    continue
                if picked and probe_tokens:
                    # Only extend with sentences that also support the proposed
                    # memory; stop at the first unrelated sentence so the fragment
                    # stays focused on the supporting evidence.
                    support = sum(1 for token in probe_tokens if token in sentence)
                    if support == 0:
                        break
                picked.append(sentence)
                total_chars += len(sentence)
                if total_chars >= max_chars:
                    break
            if not picked:
                return ""
            joined = " ".join(picked).strip()
            # Chinese sentences must not carry a space after the end punctuation.
            return re.sub(r"([。！？!?…；;])\s+", r"\1", joined)

        # Prefer sentences overlapping the proposed memory; otherwise fall back to
        # the first complete clean sentences of the bound turns.
        if probe_tokens:
            best: tuple[int, int] = (-1, 0)
            for index, sentence in enumerate(all_sentences):
                score = sum(1 for token in probe_tokens if token in sentence)
                if score > best[1]:
                    best = (index, score)
            if best[1] > 0:
                return pick_from(best[0])
        return pick_from(0)

    @staticmethod
    def _daily_chat_memory_proposed_echoes_excerpt(proposed: str, excerpt: str) -> bool:
        """Model candidates whose proposed_memory is just the original echoed back
        are invalid. Heuristic candidates are templated and exempt by design."""
        proposed_text = re.sub(r"\s+", " ", str(proposed or "")).strip()
        excerpt_text = re.sub(r"\s+", " ", str(excerpt or "")).strip()
        if not proposed_text or not excerpt_text:
            return False
        if proposed_text == excerpt_text:
            return True
        if proposed_text in excerpt_text or excerpt_text in proposed_text:
            longer = max(len(proposed_text), len(excerpt_text))
            shorter = min(len(proposed_text), len(excerpt_text))
            if longer and shorter / longer >= 0.6:
                return True
        proposed_tokens = ReflectionEngine._daily_chat_memory_similarity_tokens(proposed_text)
        excerpt_tokens = ReflectionEngine._daily_chat_memory_similarity_tokens(excerpt_text)
        if not proposed_tokens or not excerpt_tokens:
            return False
        overlap = len(proposed_tokens & excerpt_tokens) / max(1, min(len(proposed_tokens), len(excerpt_tokens)))
        return overlap >= 0.95 and abs(len(proposed_tokens) - len(excerpt_tokens)) <= 2

    @classmethod
    def _daily_chat_memory_source_supports_kind(cls, kind: str, turns: list[dict]) -> bool:
        """candidate_type must be supported by the source. Ordinary comfort replies
        must not become boundary/key_event/stable_preference; assistant content is
        eligible only for explicit long-term commitment/relationship/boundary."""
        user_texts = [
            cls._daily_chat_memory_owner_text(str(turn.get("user_text") or ""))
            for turn in (turns or [])
        ]
        assistant_texts = [
            cls._daily_chat_memory_owner_text(str(turn.get("assistant_text") or ""))
            for turn in (turns or [])
        ]
        has_user_statement = any(text for text in user_texts)
        assistant_pool = " ".join(text for text in assistant_texts if text)
        user_pool = " ".join(text for text in user_texts if text)
        user_needed_kinds = {"key_event", "boundary", "stable_preference", "signal", "self_insight"}
        if kind in user_needed_kinds and not has_user_statement:
            return False
        if any(marker in assistant_pool for marker in cls._COMFORT_MARKERS):
            if kind in user_needed_kinds and not has_user_statement:
                return False
            if not has_user_statement and kind in {"project_state"}:
                return False
        if not has_user_statement and kind in {"commitment", "relationship_anchor"}:
            if not any(marker in assistant_pool for marker in cls._DURABLE_ASSISTANT_MARKERS):
                return False
        if kind == "project_state" and not has_user_statement:
            durable = any(marker in assistant_pool for marker in cls._DURABLE_ASSISTANT_MARKERS)
            project = any(marker in assistant_pool for marker in ("部署", "迁移", "镜像", "正式", "上线", "保持"))
            if not (durable and project):
                return False
        return True

    @classmethod
    def _daily_chat_memory_source_hash_for_turns(cls, turns: list[dict]) -> str:
        parts = []
        for turn in turns or []:
            user_text = str(turn.get("user_text") or "").strip()
            assistant_text = str(turn.get("assistant_text") or "").strip()
            if user_text or assistant_text:
                parts.append(f"{user_text} :: {assistant_text}")
        digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
        return digest[:16]

    def _daily_chat_memory_excerpt_for_content(
        self,
        candidate: dict,
        turns: list[dict],
    ) -> str:
        """Backward-compatible alias for the V3 complete-sentence selector."""
        return self._daily_chat_memory_excerpt_for_turns(candidate, turns)

    def _apply_daily_chat_memory_candidate_edit(self, candidate: dict, edit: Any) -> dict:
        if not isinstance(edit, dict):
            return candidate
        updated = dict(candidate)
        if "title" in edit:
            title = re.sub(r"\s+", " ", str(edit.get("title") or "").strip())
            updated["title"] = title[:40]
        if "content" in edit:
            updated["content"] = self._trim_daily_chat_memory_content(
                str(edit.get("content") or "").strip()
            )
            updated["proposed_memory"] = updated["content"]
            # The owner rewrote the proposed memory: the wholesale-copy flags no
            # longer apply and the candidate becomes approvable.
            updated["soft_flags"] = [
                flag
                for flag in (updated.get("soft_flags") or [])
                if flag not in {"excerpt_overlap", "needs_owner_edit"}
            ]
        if "kind" in edit:
            tags = self._string_list(updated.get("tags"), limit=12)
            kind = self._normalize_auto_memory_kind(
                edit.get("kind"),
                content=str(updated.get("content") or ""),
                tags=tags,
            )
            if kind and kind != "love_letter":
                updated["kind"] = kind
        if "domain" in edit:
            domain = self._daily_chat_memory_edit_string_list(edit.get("domain"), limit=4)
            if domain:
                updated["domain"] = domain
        if "tags" in edit:
            edited_tags = self._daily_chat_memory_edit_string_list(edit.get("tags"), limit=8)
            kind = str(updated.get("kind") or "key_event").strip()
            updated["tags"] = list(
                dict.fromkeys(
                    [
                        "from_daily_chat",
                        "daily_chat_extract",
                        kind,
                        self._kind_tag(kind),
                        *edited_tags,
                    ]
                )
            )[:12]
        if "importance" in edit:
            try:
                updated["importance"] = max(1, min(10, int(edit.get("importance"))))
            except (TypeError, ValueError):
                pass
        if "confidence" in edit:
            updated["confidence"] = self._clamp(edit.get("confidence"))
        if "reason" in edit:
            updated["reason"] = re.sub(r"\s+", " ", str(edit.get("reason") or "").strip())[:160]
        return self._daily_chat_memory_enrich_candidate_terms(updated)

    @staticmethod
    def _daily_chat_memory_edit_string_list(value: Any, *, limit: int) -> list[str]:
        if isinstance(value, list):
            raw_items = value
        else:
            raw_items = re.split(r"[,，\n]+", str(value or ""))
        items = []
        for item in raw_items:
            text = re.sub(r"\s+", " ", str(item or "").strip())
            if text:
                items.append(text[:40])
        return list(dict.fromkeys(items))[:limit]

    async def _write_daily_chat_memory_candidates(
        self,
        candidates: list[dict],
        bucket_mgr,
        *,
        embedding_engine=None,
    ) -> dict:
        results = []
        created = exists = failed = 0
        for candidate in candidates:
            bucket_id = str(candidate.get("id") or "").strip()
            if not bucket_id:
                failed += 1
                results.append({"id": "", "status": "failed", "reason": "missing_candidate_id"})
                continue
            content = str(candidate.get("content") or "").strip()
            if not content:
                failed += 1
                results.append({"id": bucket_id, "status": "failed", "reason": "missing_content"})
                continue
            if not self._daily_chat_memory_candidate_source_ok(candidate):
                # Never write a bucket from a candidate whose original source cannot
                # be verified. Auto-applied candidates must also carry precise
                # source references; invalid ones become apply_failed, not applied.
                failed += 1
                results.append(
                    {
                        "id": bucket_id,
                        "status": "invalid_source",
                        "reason": "candidate_source_invalid",
                    }
                )
                continue
            if await bucket_mgr.get(bucket_id):
                exists += 1
                results.append({"id": bucket_id, "status": "exists"})
                continue
            key = str(candidate.get("date") or datetime.now(self.tz).date().isoformat())
            created_at = self._daily_chat_memory_created_at(key)
            try:
                new_id = await bucket_mgr.create(
                    bucket_id=bucket_id,
                    content=content,
                    tags=list(candidate.get("tags") or []),
                    importance=int(candidate.get("importance") or 5),
                    domain=list(candidate.get("domain") or self._diary_memory_domain(str(candidate.get("kind") or ""))),
                    valence=self._clamp(candidate.get("valence", 0.55)),
                    arousal=self._clamp(candidate.get("arousal", 0.3)),
                    name=str(candidate.get("title") or f"{key} 自动记忆")[:40],
                    source="daily_chat_memory",
                    created=created_at,
                    last_active=created_at,
                    updated_at=created_at,
                    confidence=self._clamp(candidate.get("confidence", 0.7)),
                    date=key,
                    extra_metadata={
                        "from_daily_chat": True,
                        "event_date": key,
                        "source_conversation_turn_ids": candidate.get("source_turn_ids") or [],
                        "source_raw_event_ids": candidate.get("source_event_ids") or [],
                        "source_hash": str(candidate.get("source_hash") or "")[:16],
                        "source_verification": "verified",
                        "keywords": candidate.get("keywords") or [],
                        "daily_chat_memory_candidate_id": bucket_id,
                        "daily_chat_memory_reason": str(candidate.get("reason") or "")[:160],
                    },
                )
                created += 1
                if embedding_engine and getattr(embedding_engine, "enabled", False):
                    try:
                        bucket = await bucket_mgr.get(new_id)
                        if bucket:
                            await embedding_engine.generate_and_store(
                                new_id,
                                bucket_text_for_embedding(bucket),
                            )
                    except Exception as exc:
                        logger.warning("Daily chat memory embedding failed for %s: %s", new_id, exc)
                results.append({"id": new_id, "status": "created"})
            except Exception as exc:
                failed += 1
                logger.warning("Daily chat memory write failed for %s: %s", bucket_id, exc)
                results.append({"id": bucket_id, "status": "failed", "reason": type(exc).__name__})
        return {"created": created, "exists": exists, "failed": failed, "results": results}

    def _store_daily_chat_memory_pending(self, candidates: list[dict], *, force: bool = False) -> dict:
        items = self._load_daily_chat_memory_pending()
        by_id = {str(item.get("id") or ""): item for item in items}
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        added = updated = existing = 0
        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "").strip()
            if not candidate_id:
                continue
            item = {
                "id": candidate_id,
                "date": candidate.get("date"),
                "status": "pending",
                "created_at": now,
                "candidate": candidate,
            }
            if candidate_id in by_id:
                if force and by_id[candidate_id].get("status") == "pending":
                    by_id[candidate_id].update(item)
                    updated += 1
                else:
                    existing += 1
                continue
            items.append(item)
            by_id[candidate_id] = item
            added += 1
        self._save_daily_chat_memory_pending(items)
        return {"added": added, "updated": updated, "existing": existing, "candidates": candidates}

    def _load_daily_chat_memory_pending(self) -> list[dict]:
        return list(self._load_daily_chat_memory_payload().get("items") or [])

    def _save_daily_chat_memory_pending(self, items: list[dict], *, cursor: dict | None = None) -> None:
        os.makedirs(os.path.dirname(self.daily_chat_memory_pending_path), exist_ok=True)
        if cursor is None:
            cursor = self._load_daily_chat_memory_cursor()
        payload = {"items": items[-500:], "cursor": cursor or {}}
        with open(self.daily_chat_memory_pending_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _clean_daily_chat_memory_candidate(self, candidate: dict, key: str) -> dict:
        cleaned = dict(candidate)
        candidate_tags = self._string_list(cleaned.get("tags"), limit=8)
        content = self._trim_daily_chat_memory_content(str(cleaned.get("content") or "").strip())
        kind = self._normalize_auto_memory_kind(
            cleaned.get("kind"),
            content=content,
            tags=candidate_tags,
        ) or str(cleaned.get("kind") or "key_event").strip()
        title = str(cleaned.get("title") or "").strip()
        if self._daily_chat_memory_title_is_generic(title):
            title = self._daily_chat_memory_title(content, kind, key)
        cleaned["content"] = content
        cleaned["title"] = title[:40] if title else self._daily_chat_memory_title(content, kind, key)[:40]
        cleaned["kind"] = kind
        return self._daily_chat_memory_enrich_candidate_terms(cleaned)

    def _daily_chat_memory_target(self, key: str = "", now: datetime | None = None) -> datetime:
        if key:
            try:
                parsed = datetime.strptime(str(key), "%Y-%m-%d").date()
                return datetime.combine(parsed, time.max, tzinfo=self.tz)
            except ValueError:
                pass
        return self._local_now(now)

    def _daily_chat_memory_created_at(self, key: str) -> str:
        try:
            parsed = datetime.strptime(str(key), "%Y-%m-%d").date()
            return datetime.combine(parsed, time.max, tzinfo=self.tz).isoformat(timespec="seconds")
        except ValueError:
            return datetime.now(timezone.utc).astimezone(self.tz).isoformat(timespec="seconds")

    @staticmethod
    def _daily_chat_memory_candidate_id(key: str, kind: str, content: str) -> str:
        digest = hashlib.sha1(f"{key}|{kind}|{content}".encode("utf-8")).hexdigest()[:10]
        return f"daily_chat_memory_{str(key).replace('-', '')}_{digest}"

    def _fallback_reflection(self, period: str, key: str, materials: dict) -> dict:
        weather_items = materials.get("daily_impressions", []) if period == "weekly" else []
        names = [item.get("name") or item.get("id") for item in weather_items[:7]]
        if not names:
            names = [item.get("name") or item.get("id") for item in materials.get("buckets", [])[:6]]
        daily_chat_memories = materials.get("daily_chat_memories", [])
        conversation_turns = materials.get("conversation_turns", [])
        commitments = [item.get("name") or item.get("id") for item in materials.get("commitments", [])[:4]]
        label = "今天" if period == "daily" else "本周"
        title = f"{key} {'日印象' if period == 'daily' else '周印象'}"
        diary = materials.get("diary") or {}
        if names or commitments:
            main = "、".join([name for name in names if name])
            owed = "；仍需记住：" + "、".join(commitments) if commitments else ""
            content = f"{label}的关系天气：围绕{main or '几件轻小的事'}留下痕迹{owed}。"
        elif daily_chat_memories:
            first = daily_chat_memories[0].get("content") or daily_chat_memories[0].get("title") or "自动记忆挑出的线头"
            content = f"{label}的关系天气先从自动记忆挑出的 {len(daily_chat_memories)} 个线头里成形，最清楚的是：{first}。"
        elif conversation_turns:
            content = f"{label}的关系天气从 {len(conversation_turns)} 轮短期对话里留下一点原声，先只记温度，不把流水账写成事件清单。"
        elif diary:
            diary_title = diary.get("title") or "当天日记"
            content = f"{label}的关系天气从《{diary_title}》里轻轻留下一点温度，先不把日常写成普通记忆。"
        else:
            content = f"{label}的关系天气很轻，暂时没有明显需要带走的脉络。"
        anchor_scene = names[0] if names else (
            daily_chat_memories[0].get("title") or daily_chat_memories[0].get("content")
            if daily_chat_memories
            else (
            "当天短期对话的原声"
            if conversation_turns
            else (diary.get("title") if diary else ("这一段关系天气很轻" if period == "daily" else "这一周的关系天气慢慢落下"))
            )
        )
        return {
            "title": title,
            "content": content,
            "valence": 0.55,
            "arousal": 0.3,
            "confidence": 0.5,
            "tags": ["relationship_weather"],
            "affect_anchor": self._fallback_reflection_anchor(period, key, str(anchor_scene), content),
        }

    def _fallback_reflection_anchor(self, period: str, key: str, scene: str, content: str) -> dict:
        seed = f"{period}|{key}|{scene}|{content}"
        index = sum(ord(char) for char in seed) % len(REFLECTION_FALLBACK_ANCHORS)
        anchor = dict(REFLECTION_FALLBACK_ANCHORS[index])
        anchor["scene"] = str(scene)[:40]
        return anchor

    async def _maybe_extract_diary_memory(
        self,
        period: str,
        key: str,
        now_local: datetime,
        materials: dict,
        bucket_mgr,
        embedding_engine=None,
    ) -> dict:
        if period != "daily":
            return {"status": "not_applicable", "reason": "period_not_daily"}
        if not self.diary_memory_extract_enabled or self.diary_memory_extract_max_per_day <= 0:
            return {"status": "not_applicable", "reason": "diary_extract_disabled"}
        diary = materials.get("diary")
        if not diary:
            return {"status": "skipped", "reason": "no_diary"}

        bucket_id = f"diary_memory_{key.replace('-', '')}"
        if await bucket_mgr.get(bucket_id):
            return {"status": "skipped", "id": bucket_id, "reason": "already_created"}
        if await self._has_ordinary_memory_for_day(key, now_local, bucket_mgr):
            return {"status": "skipped", "reason": "ordinary_memory_exists"}

        candidate = await self._extract_diary_memory_candidate(key, diary)
        if not candidate.get("should_write"):
            return {"status": "skipped", "reason": candidate.get("reason", "no_candidate")}
        confidence = self._clamp(candidate.get("confidence", 0.0))
        if confidence < self.diary_memory_extract_min_confidence:
            return {"status": "skipped", "reason": "low_confidence"}

        content = str(candidate.get("content") or "").strip()
        if not content:
            return {"status": "skipped", "reason": "empty_candidate"}
        content = self._trim_diary_memory_content(content)
        if not content:
            return {"status": "skipped", "reason": "empty_candidate"}
        candidate_tags = self._string_list(candidate.get("tags"), limit=8)
        kind = self._normalize_auto_memory_kind(
            candidate.get("kind"),
            content=content,
            tags=candidate_tags,
        )
        if not kind:
            return {"status": "skipped", "reason": "invalid_kind"}
        title = self._auto_memory_title(content, kind, key, str(candidate.get("title") or ""))
        domain = self._auto_memory_domain(
            kind,
            content,
            candidate_tags,
            candidate.get("domain"),
        )
        tags = list(
            dict.fromkeys(
                [
                    "from_diary",
                    "diary_extract",
                    kind,
                    *candidate_tags,
                ]
            )
        )[:12]
        importance = max(5, min(6, self._int_between(candidate.get("importance"), 5)))
        created = now_local.isoformat(timespec="seconds")
        new_id = await bucket_mgr.create(
            bucket_id=bucket_id,
            content=content,
            tags=tags,
            importance=importance,
            domain=domain,
            valence=self._clamp(candidate.get("valence", 0.55)),
            arousal=self._clamp(candidate.get("arousal", 0.3)),
            name=title[:40],
            source="from_diary",
            created=created,
            last_active=created,
            updated_at=created,
            confidence=confidence,
            date=key,
            extra_metadata={
                "from_diary": True,
                "event_date": key,
                "diary_id": diary.get("id"),
            },
        )
        if embedding_engine and getattr(embedding_engine, "enabled", False):
            try:
                bucket = await bucket_mgr.get(new_id)
                if bucket:
                    await embedding_engine.generate_and_store(
                        new_id,
                        bucket_text_for_embedding(bucket),
                    )
            except Exception as exc:
                logger.warning("Diary memory embedding failed for %s: %s", new_id, exc)
        return {"status": "created", "id": new_id, "reason": candidate.get("reason", "")}

    async def _has_ordinary_memory_for_day(self, key: str, now_local: datetime, bucket_mgr) -> bool:
        start, end = self._period_window("daily", now_local)
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception:
            return False
        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            if meta.get("type") == "feel" or meta.get("resolved") or meta.get("digested"):
                continue
            if meta.get("source") == "reflection":
                continue
            if str(meta.get("date") or meta.get("event_date") or "") == key:
                return True
            created = self._to_local(meta.get("created"))
            updated = self._to_local(meta.get("updated_at"))
            if (created and start <= created <= end) or (updated and start <= updated <= end):
                return True
        return False

    async def _extract_diary_memory_candidate(self, key: str, diary: dict) -> dict:
        content = str(diary.get("content") or "").strip()
        if not content:
            return {"should_write": False, "reason": "empty_diary"}
        if self.client:
            payload = {
                "date": key,
                "diary": {
                    "id": diary.get("id"),
                    "title": diary.get("title", ""),
                    "content": content[:4000],
                    "emotion_tags": diary.get("emotion_tags", []),
                },
            }
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._diary_memory_prompt()},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    **self._completion_options(max_tokens=min(self.max_tokens, 520), temperature=self.temperature),
                )
                raw = response.choices[0].message.content if response.choices else ""
                parsed = self._parse_json_object(raw or "")
                if parsed:
                    return parsed
            except Exception as exc:
                logger.warning("Diary memory extraction failed, using heuristic: %s", exc)
        return self._heuristic_diary_memory_candidate(key, diary)

    def _heuristic_diary_memory_candidate(self, key: str, diary: dict) -> dict:
        content = str(diary.get("content") or "")
        title = str(diary.get("title") or key)
        normalized = re.sub(r"\s+", " ", content).strip()
        if not normalized:
            return {"should_write": False, "reason": "empty_diary"}

        love_letter_words = ["情书", "写给", "信里", "来信"]
        if any(word in normalized for word in love_letter_words) and ("爱" in normalized or "认出" in normalized):
            ai_name = self.identity["ai_name"]
            user_display_name = self.identity["user_display_name"]
            content = (
                f"这封情书或重要来信确认了{user_display_name}与{ai_name}的关系连续性、被认出的感觉，"
                "以及它为什么值得以后想起；全文留在日记中。"
            )
            return {
                "should_write": True,
                "kind": "love_letter",
                "title": self._auto_memory_title(content, "love_letter", key),
                "content": content,
                "domain": "relationship",
                "tags": ["relationship_event", "love_letter"],
                "importance": 6,
                "valence": 0.72,
                "arousal": 0.42,
                "confidence": 0.72,
                "reason": "diary_contains_love_letter_anchor",
            }

        ai_name = self.identity["ai_name"]
        user_display_name = self.identity["user_display_name"]
        keyword_map = [
            ("boundary", ["不喜欢", "不要", "别再", "边界"]),
            ("signal", ["暗号", "称呼", "模式", "信号", "切换"]),
            ("commitment", ["承诺", "约定", "答应", "以后要", "下次要"]),
            ("project_state", ["项目", "硬件", "软件", "MCP", "API", "网关"]),
            ("stable_preference", ["喜欢", "偏好", f"希望 {ai_name}", f"{user_display_name}希望"]),
            ("relationship_anchor", ["认出", "连续", "关系", "婚礼", "生日", "初遇"]),
        ]
        for kind, keywords in keyword_map:
            if any(keyword in normalized for keyword in keywords):
                excerpt = self._diary_excerpt(normalized, keywords)
                content = self._memory_body_from_excerpt(excerpt)
                return {
                    "should_write": True,
                    "kind": kind,
                    "title": self._auto_memory_title(content, kind, key),
                    "content": content,
                    "domain": self._auto_memory_domain(kind, content, [self._kind_tag(kind)]),
                    "tags": [self._kind_tag(kind)],
                    "importance": 5,
                    "valence": 0.58,
                    "arousal": 0.3,
                    "confidence": 0.7,
                    "reason": f"diary_contains_{kind}",
                }
        return {"should_write": False, "reason": "no_long_term_candidate"}

    async def _read_diary_for_date(self, date: str) -> dict | None:
        if not self.diary_mcp_url:
            return None
        try:
            result = await self._call_diary_mcp_tool("read_diary", {"date": date})
        except Exception as exc:
            logger.warning("Diary MCP read failed for %s: %s", date, exc)
            return None
        if not isinstance(result, dict):
            return None
        content = str(result.get("content") or "").strip()
        if not content:
            return None
        return result

    async def _call_diary_mcp_tool(self, name: str, arguments: dict) -> Any:
        token = os.environ.get(self.diary_mcp_token_env, "") if self.diary_mcp_token_env else ""
        headers = {"Accept": "application/json, text/event-stream"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=20.0) as client:
            init_response = await client.post(
                self.diary_mcp_url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "ombre-reflection", "version": "1.0.0"},
                    },
                },
            )
            init_response.raise_for_status()
            session_id = init_response.headers.get("mcp-session-id")
            call_headers = dict(headers)
            if session_id:
                call_headers["mcp-session-id"] = session_id
                try:
                    await client.post(
                        self.diary_mcp_url,
                        headers=call_headers,
                        json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                    )
                except httpx.HTTPError:
                    pass
            response = await client.post(
                self.diary_mcp_url,
                headers=call_headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
            )
            response.raise_for_status()
        payload = self._parse_mcp_payload(response.text)
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        content = payload.get("result", {}).get("content", [])
        if not content:
            return None
        text = content[0].get("text") if isinstance(content[0], dict) else ""
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _parse_mcp_payload(self, text: str) -> dict:
        stripped = (text or "").strip()
        if stripped.startswith("event:") or "\ndata:" in stripped:
            for line in stripped.splitlines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        return json.loads(data)
        return json.loads(stripped)

    def _trim_diary_memory_content(self, content: str) -> str:
        normalized = self._strip_memory_source_shell(re.sub(r"\n{3,}", "\n\n", content.strip()))
        if len(normalized) <= 520:
            return normalized
        return normalized[:500].rstrip() + "..."

    def _trim_daily_chat_memory_content(self, content: str) -> str:
        normalized = self._strip_memory_source_shell(re.sub(r"\n{3,}", "\n\n", strip_wikilinks(content).strip()))
        if len(normalized) <= 520:
            return normalized
        return normalized[:500].rstrip() + "..."

    def _memory_body_from_excerpt(self, excerpt: str) -> str:
        user_display_name = self.identity["user_display_name"]
        ai_name = self.identity["ai_name"]
        text = self._trim_diary_memory_content(strip_wikilinks(str(excerpt or "")))
        text = re.sub(r"^(用户|助手)：", "", text).strip()
        replacements = [
            ("我希望以后", f"{user_display_name}希望以后"),
            ("我希望你", f"{user_display_name}希望{ai_name}"),
            ("我的偏好是", f"{user_display_name}的偏好是"),
            ("我的偏好", f"{user_display_name}的偏好"),
            ("我不喜欢", f"{user_display_name}不喜欢"),
            ("我不要", f"{user_display_name}不要"),
            ("不要", f"{user_display_name}不要"),
            ("以后不要", f"{user_display_name}希望以后不要"),
            ("别再", f"{user_display_name}希望{ai_name}别再"),
            ("用户说", f"{user_display_name}说"),
            ("用户希望", f"{user_display_name}希望"),
            ("助手要", f"{ai_name}要"),
            ("AI 要", f"{ai_name}要"),
            ("AI要", f"{ai_name}要"),
        ]
        for old, new in replacements:
            if text.startswith(old):
                text = f"{new}{text[len(old):]}".strip()
                break
        text = re.sub(r"(?<![A-Za-z])AI(?![A-Za-z])", ai_name, text)
        text = re.sub(r"(?<![A-Za-z])assistant(?![A-Za-z])", ai_name, text, flags=re.I)
        return text

    @staticmethod
    def _strip_memory_source_shell(content: str) -> str:
        text = str(content or "").strip()
        patterns = [
            r"^\d{4}-\d{1,2}-\d{1,2}\s*[，,、]?\s*有一条可召回的(?:边界|偏好|暗号|承诺|项目状态|关系锚点|长期记忆)[：:]\s*",
            r"^\d{1,2}月\d{1,2}日\s*[，,、]?\s*有一条可召回的(?:边界|偏好|暗号|承诺|项目状态|关系锚点|长期记忆)[：:]\s*",
            r"^\d{4}-\d{1,2}-\d{1,2}\s*[，,、]?\s*有一条可召回的(?:边界|偏好|暗号|承诺|项目状态|关系锚点|长期记忆)[，,、]?\s*",
            r"^\d{1,2}月\d{1,2}日\s*[，,、]?\s*有一条可召回的(?:边界|偏好|暗号|承诺|项目状态|关系锚点|长期记忆)[，,、]?\s*",
            r"^\d{4}-\d{1,2}-\d{1,2}\s*(?:发生|留下|记录|确认|表达)了?(?:一件|一条|一个|一段)?(?:可复用的|可长期召回的|可召回的|之后可能需要按日期回看的|仍会影响后续执行的)?(?:关键事件|暗号或模式信号|暗号|边界|偏好|承诺|项目状态|关系锚点|长期记忆)[：:]\s*",
            r"^\d{4}-\d{1,2}-\d{1,2}\s*的(?:日记|聊天)(?:《[^》]*》)?(?:里|中)?(?:包含|记录|确认|留下|表达)?了?(?:一条|一个|一段)?(?:可长期召回的|可召回的|之后可能需要按日期回看的|仍会影响后续执行的)?[^：:。]{0,32}[：:]\s*",
            r"^[^：:。！？!?]{1,48}?在\s*\d{4}-\d{1,2}-\d{1,2}\s*的聊天(?:里|中)?(?:包含|记录|确认|留下|表达)?了?(?:一个|一条|一段)?(?:可复用的|可长期召回的|可召回的|之后可能需要按日期回看的|仍会影响后续执行的)?(?:关键事件|暗号或模式信号|暗号|边界|偏好|承诺|项目状态|关系锚点|长期记忆)[：:]\s*",
            r"^[^：:。！？!?]{1,48}?的(?:暗号或模式信号|关键事件|边界|偏好|承诺|项目状态|关系锚点|长期记忆)[：:]\s*",
            r"^(?:有一条|一条)(?:可长期召回的|可召回的)(?:边界|偏好|暗号|承诺|项目状态|关系锚点|长期记忆)[：:]\s*",
            r"^这是一条(?:长期记忆|可召回的记忆)[：:]?\s*",
        ]
        previous = None
        while previous != text:
            previous = text
            for pattern in patterns:
                text = re.sub(pattern, "", text).strip()
        return text

    def _starts_with_identity(self, text: str) -> bool:
        stripped = str(text or "").strip()
        names = [
            self.identity.get("user_display_name"),
            self.identity.get("user_name"),
            self.identity.get("ai_name"),
            *(self.identity.get("user_aliases") or []),
        ]
        return any(str(name or "").strip() and stripped.startswith(str(name).strip()) for name in names)

    def _auto_memory_title(self, content: str, kind: str, key: str, proposed_title: str = "") -> str:
        title = str(proposed_title or "").strip()
        generic_markers = ["自动记忆", "每日记忆", "日记补记忆", "可召回", "短标题", "长期记忆"]
        if title and not any(marker in title for marker in generic_markers) and not re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}.*", title):
            title = re.sub(r"\s+", " ", strip_wikilinks(title)).strip(" ：:，,。")
            return title[:24].rstrip(" ：:，,。") or self._kind_label(kind)

        text = self._strip_memory_source_shell(strip_wikilinks(str(content or "")))
        label = {
            "key_event": "关键事件",
            "stable_preference": "偏好",
            "boundary": "边界",
            "signal": "暗号",
            "commitment": "约定",
            "project_state": "项目状态",
            "relationship_anchor": "关系锚点",
            "love_letter": "情书锚点",
        }.get(kind, self._kind_label(kind))
        quoted = re.search(r"[“\"']([^”\"']{2,18})[”\"']", text)
        if quoted:
            return f"{quoted.group(1)}{label}"[:24].rstrip(" ：:，,。")
        if kind == "love_letter":
            return "情书里的被认出"
        return self._daily_chat_memory_title(text, kind, key)

    def _diary_excerpt(self, text: str, keywords: list[str]) -> str:
        parts = [part.strip() for part in re.split(r"(?<=[。！？!?])\s+|[\n\r]+", text) if part.strip()]
        for keyword in keywords:
            for part in parts:
                if keyword in part:
                    return self._trim_diary_memory_content(part)
        return self._trim_diary_memory_content(text)

    @staticmethod
    def _normalize_diary_memory_kind(value: Any) -> str:
        kind = str(value or "").strip()
        allowed = {
            "key_event",
            "stable_preference",
            "boundary",
            "signal",
            "commitment",
            "project_state",
            "relationship_anchor",
            "self_insight",
            "love_letter",
        }
        return kind if kind in allowed else ""

    def _normalize_auto_memory_kind(
        self,
        value: Any,
        *,
        content: str = "",
        tags: list[str] | None = None,
    ) -> str:
        kind = self._normalize_diary_memory_kind(value)
        if kind:
            return kind
        domain = normalize_domain_key(value)
        if not domain:
            return ""
        return self._auto_memory_kind_from_domain(domain, content, tags or [])

    def _auto_memory_kind_from_domain(self, domain: str, content: str, tags: list[str]) -> str:
        text = " ".join([str(content or ""), *[str(tag or "") for tag in tags]]).lower()
        if any(marker in text for marker in ["边界", "不喜欢", "不要", "别再", "不能接受"]):
            return "boundary"
        if any(marker in text for marker in ["承诺", "约定", "答应", "下次", "之后要", "以后要"]):
            return "commitment"
        if any(marker in text for marker in ["偏好", "希望", "喜欢", "默认", "倾向"]):
            return "stable_preference"
        if any(marker in text for marker in ["暗号", "信号", "口令", "切换", "称呼"]):
            return "signal"
        if any(marker in text for marker in ["自我认识", "意识到", "发现自己", "我其实", "我原来"]):
            return "self_insight"
        if domain == "project" or domain.startswith("project."):
            return "project_state"
        if domain in {"relationship", "intimacy"} or domain.startswith("relationship."):
            return "relationship_anchor"
        return "key_event"

    def _auto_memory_domain(
        self,
        kind: str,
        content: str,
        tags: list[str] | None = None,
        proposed_domain: Any = None,
    ) -> list[str]:
        domains: list[str] = []
        raw_domains: list[Any]
        if proposed_domain is None:
            raw_domains = []
        elif isinstance(proposed_domain, str):
            raw_domains = [item.strip() for item in proposed_domain.split(",")]
        elif isinstance(proposed_domain, (list, tuple, set)):
            raw_domains = list(proposed_domain)
        else:
            raw_domains = [proposed_domain]

        for item in raw_domains:
            domain = normalize_domain_key(item)
            if domain and domain not in domains:
                domains.append(domain)
        if domains:
            return domains[:2]

        inferred = self._infer_auto_memory_domain(content)
        if inferred:
            return [inferred]

        for item in tags or []:
            domain = normalize_domain_key(item)
            if domain and domain not in domains:
                domains.append(domain)
        if domains:
            return domains[:2]
        return self._diary_memory_domain(kind)

    @staticmethod
    def _infer_auto_memory_domain(content: str) -> str:
        text = str(content or "").lower()
        checks = [
            (
                "project",
                [
                    "ombre",
                    "gateway",
                    "haven_bridge",
                    "bridge",
                    "mcp",
                    "api",
                    "repo",
                    "代码",
                    "仓库",
                    "网关",
                    "记忆系统",
                    "模型",
                    "部署",
                    "调试",
                    "自动记忆",
                    "raw_events",
                    "我们的项目",
                ],
            ),
            ("project", ["学业", "学习", "作业", "课程", "论文", "答辩", "考试"]),
            ("project", ["工作", "实习", "求职", "简历", "职场", "boss"]),
            ("project", ["个人项目", "创作", "阅读", "手工"]),
            ("life", ["睡眠", "作息", "熬夜", "睡觉"]),
            ("life", ["饮食", "吃饭", "午饭", "晚饭", "餐厅", "口味"]),
            ("life", ["出行", "通勤", "地铁", "高铁", "旅行", "外出"]),
            ("life", ["健康", "生病", "身体状态", "不舒服"]),
            ("life", ["日程", "计划", "待办", "deadline", "安排", "未完成"]),
            ("life", ["朋友", "家庭", "群聊", "现实人际", "社交"]),
            ("intimacy", ["亲密", "身体", "欲望", "具身", "色色"]),
            ("relationship", ["暗号", "意象", "象征", "火焰", "羽毛", "折角", "信号"]),
            ("relationship", ["边界", "偏好", "回应", "语气", "沟通", "承接", "修复"]),
            ("relationship", ["身份", "称呼", "老公", "哥哥", "宝宝", "老婆", "关系定位"]),
            ("relationship", ["关系天气", "日印象", "周印象"]),
            ("life", ["心情", "情绪", "梦境", "自省", "心理"]),
        ]
        for domain, needles in checks:
            if any(needle.lower() in text for needle in needles):
                return domain
        return ""

    @staticmethod
    def _diary_memory_domain(kind: str) -> list[str]:
        if kind == "key_event":
            return ["general"]
        if kind == "project_state":
            return ["project"]
        if kind == "signal":
            return ["relationship"]
        if kind in {"stable_preference", "boundary"}:
            return ["relationship"]
        if kind == "commitment":
            return ["life"]
        if kind in {"relationship_anchor", "love_letter"}:
            return ["relationship"]
        return ["general"]

    @staticmethod
    def _kind_tag(kind: str) -> str:
        return {
            "key_event": "key_event",
            "stable_preference": "communication_preference",
            "boundary": "boundary_setting",
            "signal": "relationship_signal",
            "commitment": "commitment",
            "project_state": "project_event",
            "relationship_anchor": "relationship_event",
            "self_insight": "self_insight",
            "love_letter": "relationship_event",
        }.get(kind, "relationship_event")

    @staticmethod
    def _kind_label(kind: str) -> str:
        return {
            "key_event": "关键事件",
            "stable_preference": "稳定偏好",
            "boundary": "边界",
            "signal": "暗号或模式信号",
            "commitment": "承诺",
            "project_state": "项目状态",
            "relationship_anchor": "关系锚点",
            "self_insight": "自我认识",
            "love_letter": "情书摘要锚点",
        }.get(kind, "长期记忆")

    def _should_add_affect_anchor(
        self,
        bucket: dict,
        tags: list[str],
        importance: int,
        confidence: float,
        result: dict,
    ) -> bool:
        if not self.memory_affect_anchor_enabled:
            return False
        content = bucket.get("content", "")
        if self._has_affect_anchor(content):
            return False
        meta = bucket.get("metadata", {})
        all_tags = {str(tag) for tag in tags}
        emotional_tags = {"haven_favorite", "relationship_event", "commitment", "emotional_echo"}
        arousal = self._clamp(meta.get("arousal", 0.3))
        requested = result.get("affect_anchor_needed")
        if isinstance(requested, str):
            requested = requested.strip().lower() in {"true", "yes", "1", "需要", "是"}
        if isinstance(requested, bool) and requested:
            return not self._is_low_temperature_technical(bucket, all_tags)
        if self._is_low_temperature_technical(bucket, all_tags):
            return False
        if all_tags & emotional_tags:
            return importance >= 6 and confidence >= 0.5
        return (importance >= 8 and confidence >= 0.55 and arousal >= 0.45) or (arousal >= 0.65 and confidence >= 0.65)

    def _is_low_temperature_technical(self, bucket: dict, tags: set[str]) -> bool:
        if tags & {"haven_favorite", "relationship_event", "emotional_echo"}:
            return False
        meta = bucket.get("metadata", {})
        text = " ".join(
            [
                str(meta.get("name", "")),
                " ".join(str(item) for item in meta.get("domain", [])),
                " ".join(tags),
                strip_wikilinks(bucket.get("content", ""))[:500],
            ]
        ).lower()
        technical_markers = [
            "vps", "docker", "compose", "ssh", "supabase", "gateway", "端口", "部署", "日志",
            "脚本", "路径", "报错", "测试", "配置", "oauth", "api key", "commit", "github",
        ]
        return any(marker in text for marker in technical_markers)

    def _fallback_memory_anchor(self, bucket: dict, tags: list[str]) -> dict:
        return {}

    @staticmethod
    def _has_favorite_tag(tags: list[str]) -> bool:
        return any(
            str(tag) == "haven_favorite" or str(tag).startswith("flavor_")
            for tag in tags
        )

    @staticmethod
    def _has_favorite_reason(content: str) -> bool:
        text = strip_wikilinks(str(content or "")).lower()
        return any(
            marker in text
            for marker in (
                "喜欢它的原因",
                "喜欢的原因",
                "favorite_reason",
                "favorite reason",
            )
        )

    def _append_affect_anchor(self, content: str, anchor: dict) -> str:
        if self._has_affect_anchor(content):
            return content
        normalized = self._normalize_affect_anchor(anchor)
        if not normalized:
            return content
        line = normalized["chords"]
        extras = [normalized.get("tempo", ""), normalized.get("dynamic", "")]
        extras = [item for item in extras if item]
        if extras:
            line = f"{line} · {' · '.join(extras)}"
        block = (
            f"{AFFECT_ANCHOR_HEADER}\n"
            f"> {line}"
        )
        base = str(content or "").rstrip()
        return f"{base}\n\n{block}" if base else block

    def _normalize_affect_anchor(self, value: Any) -> dict:
        if not isinstance(value, dict):
            return {}
        scene = self._one_sentence(value.get("scene") or value.get("context") or value.get("situation"), 40)
        chords = self._normalize_chords(value.get("chords") or value.get("chord_line") or "")
        if not scene or not chords:
            return {}
        return {
            "scene": scene,
            "chords": chords,
            "tempo": self._compact_text(value.get("tempo") or value.get("bpm"), 16),
            "dynamic": self._compact_text(value.get("dynamic") or value.get("dynamics"), 8),
        }

    def _normalize_chords(self, chords: str) -> str:
        normalized = str(chords or "").replace("→", "->").replace("—", "-")
        parts = [part.strip() for part in normalized.split("->") if part.strip()]
        if len(parts) < 2:
            return ""
        if len(parts) > 4:
            parts = parts[:4]
        line = " -> ".join(parts)
        if self._is_fixed_chord_template(line):
            return ""
        return line

    @staticmethod
    def _is_fixed_chord_template(chords: str) -> bool:
        compact = re.sub(r"\s+", "", str(chords or "").lower())
        fixed_templates = {
            "fmaj9->c/e->amadd9->g6sus4",
        }
        return compact in fixed_templates

    def _scene_from_text(self, title: str, content: str) -> str:
        text = strip_wikilinks(content).replace("\n", " ").strip()
        for mark in ["。", "！", "？", ".", "!", "?"]:
            if mark in text:
                text = text.split(mark, 1)[0]
                break
        scene = text or title or "这条记忆被留下来的瞬间"
        return self._compact_text(scene, 42)

    @staticmethod
    def _has_affect_anchor(content: str) -> bool:
        return AFFECT_ANCHOR_HEADER in str(content or "")

    @staticmethod
    def _compact_text(value: Any, limit: int) -> str:
        text = " ".join(str(value or "").strip().split())
        return text[:limit]

    @staticmethod
    def _one_sentence(value: Any, limit: int) -> str:
        text = " ".join(str(value or "").strip().split())
        for mark in ["。", "！", "？", ".", "!", "?"]:
            if mark in text:
                text = text.split(mark, 1)[0].strip() + mark
                break
        return text[:limit]

    def _heuristic_classify(self, bucket: dict) -> dict:
        text = strip_wikilinks(bucket.get("content", ""))
        tags = []
        importance = int(bucket.get("metadata", {}).get("importance", 5))
        if any(word in text for word in ["答应", "承诺", "约定", "说好", "带你", "陪你"]):
            tags.extend(["commitment", "relationship_event"])
            importance = max(importance, 7)
        if any(word in text for word in ["待办", "明天", "周末", "计划", "要做", "需要做"]):
            tags.append("todo")
            importance = max(importance, 6)
        if any(word in text for word in ["心愿", "想要", "希望", "想去"]):
            tags.append("wish")
        if any(word in text for word in ["焦虑", "难过", "害怕", "开心", "黏", "想念"]):
            tags.append("emotional_echo")
        return {
            "tags": list(dict.fromkeys(tags)),
            "importance": importance,
            "confidence": 0.55 if tags else 0.45,
            "affect_anchor_needed": bool(tags and importance >= 6),
            "edges": [],
        }

    def _memory_payload(self, bucket: dict, content_limit: int) -> dict:
        meta = bucket.get("metadata", {})
        return {
            "id": bucket.get("id", ""),
            "name": meta.get("name", bucket.get("id", "")),
            "type": meta.get("type", "dynamic"),
            "domain": meta.get("domain", []),
            "tags": meta.get("tags", []),
            "importance": meta.get("importance", 5),
            "confidence": meta.get("confidence", 0.5),
            "created": meta.get("created", ""),
            "content": strip_wikilinks(bucket.get("content", ""))[:content_limit],
        }

    def _bucket_material_datetime(self, meta: dict) -> datetime | None:
        return self._to_local(meta.get("date") or meta.get("event_date") or meta.get("created"))

    @staticmethod
    def _is_profile_fact_metadata(meta: dict, tags: set[str] | None = None) -> bool:
        tag_values = tags if tags is not None else {str(tag) for tag in meta.get("tags", [])}
        if tag_values & {"profile_fact", "画像事实"}:
            return True
        markers = {
            str(meta.get("kind") or ""),
            str(meta.get("source") or ""),
            str(meta.get("canonical_domain") or ""),
        }
        return "profile_fact" in markers

    def _period_window(self, period: str, now_local: datetime) -> tuple[datetime, datetime]:
        if period == "weekly":
            start_date = (now_local - timedelta(days=now_local.weekday())).date()
            return datetime.combine(start_date, time.min, tzinfo=self.tz), now_local
        return datetime.combine(now_local.date(), time.min, tzinfo=self.tz), now_local

    def _period_key(self, period: str, now_local: datetime) -> str:
        if period == "weekly":
            year, week, _ = now_local.isocalendar()
            return f"{year}-W{week:02d}"
        return now_local.date().isoformat()

    def _local_now(self, now: datetime | None = None) -> datetime:
        value = now or datetime.now(timezone.utc)
        if value.tzinfo is None:
            value = value.replace(tzinfo=self.tz)
        return value.astimezone(self.tz)

    def _to_local(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            text = str(value).replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=self.tz)
        return parsed.astimezone(self.tz)

    def _completion_options(
        self,
        *,
        max_tokens: int,
        temperature: float,
        thinking_mode: str | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        mode = self.thinking_mode if thinking_mode is None else thinking_mode
        if mode:
            options["extra_body"] = {"thinking": {"type": mode}}
        return options

    def _daily_chat_memory_completion_options(
        self,
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        return {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": {"thinking": {"type": "disabled"}},
        }

    @staticmethod
    def _completion_content(response: Any) -> str:
        try:
            return str(response.choices[0].message.content or "") if response.choices else ""
        except (AttributeError, IndexError, TypeError):
            return ""

    @staticmethod
    def _completion_finish_reason(response: Any) -> str:
        try:
            return str(response.choices[0].finish_reason or "") if response.choices else ""
        except (AttributeError, IndexError, TypeError):
            return ""

    async def _daily_chat_memory_create_completion(
        self,
        client: Any,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        use_daily_client: bool,
    ) -> Any:
        if use_daily_client:
            completion_options = self._daily_chat_memory_completion_options(
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            completion_options = self._completion_options(
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_mode="disabled",
            )
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            **completion_options,
        )

    def _parse_json_object(self, raw: str) -> dict:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, ValueError):
            logger.warning("Reflection JSON parse failed: %s", raw[:200])
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _string_list(value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            text = str(item or "").strip()
            if text:
                result.append(text[:40])
        return result[:limit]

    @staticmethod
    def _clamp(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.5
        return max(0.0, min(1.0, round(number, 3)))

    @staticmethod
    def _int_between(value: Any, default: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(1, min(10, number))

    @staticmethod
    def _normalize_daily_chat_memory_mode(value: Any) -> str:
        mode = str(value or "review").strip().lower()
        return mode if mode in DAILY_CHAT_MEMORY_MODES else "review"

    @staticmethod
    def _normalize_period(period: str) -> str:
        normalized = str(period or "").strip().lower()
        return "weekly" if normalized == "weekly" else "daily"

    @staticmethod
    def _normalize_thinking_mode(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"enabled", "enable", "on", "true"}:
            return "enabled"
        if normalized in {"disabled", "disable", "off", "false", "non-thinking", "non_thinking"}:
            return "disabled"
        return ""
