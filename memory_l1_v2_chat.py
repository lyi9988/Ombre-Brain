"""The model fills recall metadata for an existing Chat L1 draft."""

CHAT_PROMPT_ID = "ombre.memory_l1_v2_chat_prompt"

CHAT_L1_PROMPT = """为顾衍主动提交的这张记忆补充召回信息。读 agent_draft 指向的原稿，附近聊天用于核对人物、日期和语境。原稿中的感受、印象和未确定的事按原意理解。

按原稿主线从目录选一个 kind、一种 stability 和一个主主题。确有帮助时再选副主题。关键词用以后能认出这件事的具体词组。人物用输入里的编号。已有标题由程序沿用，缺标题时给一个短标题。

提交时间由程序填写。event_time 填原稿所述事情有依据的发生时间，只有提交时间时，事件日期留空。出处用材料编号即可。按 Schema 返回一张卡的召回字段。"""


def build_chat_response_format(source_refs, people_refs, catalog, *, has_title, include_review):
    from memory_l1_v2 import build_model_response_format

    native = build_model_response_format(source_refs, people_refs, catalog, include_review=include_review)
    schema_info = native["json_schema"]
    schema_info["name"] = "memory_l1_v2_chat_metadata"
    cards = schema_info["schema"]["properties"]["cards"]
    cards.update(minItems=1, maxItems=1)
    card = cards["items"]
    omitted = {"body", "people_links", "scope_notes"}
    if has_title:
        omitted.add("title")
    for field in omitted:
        card["properties"].pop(field)
    card["required"] = [field for field in card["required"] if field not in omitted]
    card["properties"]["sources"]["minItems"] = 0
    return native
