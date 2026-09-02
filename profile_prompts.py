"""Dependency-light profile proposal prompt factories."""
from __future__ import annotations


PROFILE_FACT_PROPOSAL_PROMPT_TEMPLATE = """你是一个证据化用户画像候选生成器。请只根据给定证据桶提出可能值得长期保存的画像事实。

身份：
- 当前用户：{user_display_name}
- 当前 AI：{ai_name}

边界：
1. 只能提出能被证据直接支持的事实，不要补常识，不要推测。
2. 不要提出 root prompt、pinned、protected、Core Memory 更新。
3. 不要把短期情绪当长期画像，除非证据明确显示稳定偏好、边界、习惯、关系锚点或重要日期。
4. 如果证据不足，返回 []。
5. 只输出 JSON 数组，不要 markdown，不要解释。

每个候选必须包含：
{{
  "fact": "一句可读中文事实",
  "profile_kind": "preference|boundary|habit|identity|relationship_anchor|life_fact|work_state|other",
  "subject": "user|ai|relationship",
  "predicate": "snake_case_or_short_key",
  "object": "事实对象，允许中文",
  "evidence_bucket_id": "必须等于给定 bucket id",
  "evidence_moment_id": "可为空",
  "confidence": 0.0,
  "reason": "为什么这条证据足够支撑"
}}

最多返回 3 条。"""


ANCHOR_PROPOSAL_PROMPT_TEMPLATE = """你是一个长期锚点候选生成器。请判断给定记忆桶是否值得被人工标为 anchor。

身份：
- 当前用户：{user_display_name}
- 当前 AI：{ai_name}

边界：
1. 只能判断这个既有 bucket 是否适合作为长期锚点，不要提出新记忆，不要改写正文。
2. 不要建议 pinned、protected、Core Memory 或 profile_fact 更新。
3. anchor 应该是未来长期会反复帮助理解用户、关系、承诺、重要经历或长期项目的记忆。
4. 不要把今天很强烈但未被时间验证的短期情绪当 anchor。
5. 如果不适合，返回 []。
6. 只输出 JSON 数组，不要 markdown，不要解释。

候选格式：
{{
  "bucket_id": "必须等于给定 bucket id",
  "anchor_kind": "relationship|identity|commitment|life_event|project|preference|other",
  "reason": "为什么它适合成为长期锚点",
  "future_use": "以后什么场景需要它",
  "confidence": 0.0
}}

最多返回 1 条。"""


__all__ = [
    "ANCHOR_PROPOSAL_PROMPT_TEMPLATE",
    "PROFILE_FACT_PROPOSAL_PROMPT_TEMPLATE",
]
