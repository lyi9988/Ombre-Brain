#!/usr/bin/env python3
"""
用 API 重新打标未分类记忆桶，修正 domain/tags/name，移动到正确目录。
用法: docker exec ombre-brain python3 /app/reclassify_api.py
"""
import asyncio
import os
import json
import glob
import re

from openai import AsyncOpenAI
import frontmatter

from prompt_plan_mirror import PromptPlanMirrorStore
from prompt_source_registry import resolve_fixed_prompt

ANALYZE_PROMPT = (
    "你是一个内容分析器。请分析以下文本，输出结构化的元数据。\n\n"
    "分析规则：\n"
    '1. domain（主题域）：选最精确的 1~2 个，只选真正相关的\n'
    '   日常: ["饮食", "穿搭", "出行", "居家", "购物"]\n'
    '   人际: ["家庭", "恋爱", "友谊", "社交"]\n'
    '   成长: ["工作", "学习", "考试", "求职"]\n'
    '   身心: ["健康", "心理", "睡眠", "运动"]\n'
    '   兴趣: ["游戏", "影视", "音乐", "阅读", "创作", "手工"]\n'
    '   数字: ["编程", "AI", "硬件", "网络"]\n'
    '   事务: ["财务", "计划", "待办"]\n'
    '   内心: ["情绪", "回忆", "梦境", "自省"]\n'
    "2. valence（情感效价）：0.0~1.0，0=极度消极 → 0.5=中性 → 1.0=极度积极\n"
    "3. arousal（情感唤醒度）：0.0~1.0，0=非常平静 → 0.5=普通 → 1.0=非常激动\n"
    "4. tags（关键词标签）：3~5 个最能概括内容的关键词\n"
    "5. suggested_name（建议桶名）：10字以内的简短标题\n\n"
    "输出格式（纯 JSON，无其他内容）：\n"
    '{\n'
    '  "domain": ["主题域1", "主题域2"],\n'
    '  "valence": 0.7,\n'
    '  "arousal": 0.4,\n'
    '  "tags": ["标签1", "标签2", "标签3"],\n'
    '  "suggested_name": "简短标题"\n'
    '}'
)

DATA_DIR = "/data/dynamic"
UNCLASS_DIR = os.path.join(DATA_DIR, "未分类")


def _prompt_mirror_path() -> str:
    return str(
        os.environ.get("OMBRE_PROMPT_PLAN_MIRROR_PATH")
        or os.environ.get("OMBRE_BUCKETS_DIR")
        or "/data"
    ).rstrip("/\\") + "/prompt_plan_mirror.sqlite3"


def resolve_reclassify_prompt(
    prompt_plan_mirror: PromptPlanMirrorStore | None = None,
    *,
    config: dict | None = None,
) -> str:
    """Resolve the reclassification factory through the Gateway mirror."""
    store = prompt_plan_mirror or PromptPlanMirrorStore(_prompt_mirror_path())
    try:
        resolved, _meta = resolve_fixed_prompt(
            store,
            "ombre.reclassify_prompt",
            config=config or {},
            identity_id="jiajia-main",
            conversation_id="",
        )
        return resolved
    except Exception:
        # Reclassify is a maintenance script and must remain usable when the
        # optional mirror is absent/degraded; this is the explicit legacy path.
        return ANALYZE_PROMPT


def sanitize(name):
    name = re.sub(r'[<>:"/\\|?*\n\r]', '', name).strip()
    return name[:20] if name else "未命名"


async def reclassify():
    client = AsyncOpenAI(
        api_key=os.environ.get("OMBRE_API_KEY", ""),
        base_url="https://api.siliconflow.cn/v1",
        timeout=60.0,
    )
    prompt_mirror = PromptPlanMirrorStore(_prompt_mirror_path())

    files = sorted(glob.glob(os.path.join(UNCLASS_DIR, "*.md")))
    print(f"找到 {len(files)} 个未分类文件\n")

    for fpath in files:
        basename = os.path.basename(fpath)
        post = frontmatter.load(fpath)
        content = post.content.strip()
        name = post.metadata.get("name", "")
        full_text = f"{name}\n{content}" if name else content

        try:
            resp = await client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[
                    {"role": "system", "content": resolve_reclassify_prompt(prompt_mirror)},
                    {"role": "user", "content": full_text[:2000]},
                ],
                max_tokens=256,
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(raw)
        except Exception as e:
            print(f"  X API失败 {basename}: {e}")
            continue

        new_domain = result.get("domain", ["未分类"])[:3]
        new_tags = result.get("tags", [])[:5]
        new_name = sanitize(result.get("suggested_name", "") or name)
        new_valence = max(0.0, min(1.0, float(result.get("valence", 0.5))))
        new_arousal = max(0.0, min(1.0, float(result.get("arousal", 0.3))))

        post.metadata["domain"] = new_domain
        post.metadata["tags"] = new_tags
        post.metadata["valence"] = new_valence
        post.metadata["arousal"] = new_arousal
        if new_name:
            post.metadata["name"] = new_name

        # 写回文件
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        # 移动到正确目录
        primary = sanitize(new_domain[0]) if new_domain else "未分类"
        target_dir = os.path.join(DATA_DIR, primary)
        os.makedirs(target_dir, exist_ok=True)

        bid = post.metadata.get("id", "")
        new_filename = f"{new_name}_{bid}.md" if new_name and new_name != bid else basename
        dest = os.path.join(target_dir, new_filename)

        if dest != fpath:
            os.rename(fpath, dest)

        print(f"  OK {basename}")
        print(f"     -> {primary}/{new_filename}")
        print(f"     domain={new_domain} tags={new_tags} V={new_valence} A={new_arousal}")
        print()


if __name__ == "__main__":
    asyncio.run(reclassify())
