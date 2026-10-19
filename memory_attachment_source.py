"""Attachment source normalization from the authenticated Aiz exact reader.

Does not select a file or establish model adoption. The caller must possess a
request-bound adopted source or an independently authorized explicit selection.
"""
import hashlib
import json

NAMESPACE='snapshot.aiz.attachment_text'
VIEW='attachment.utf8-sig.exact-v1'
KEYS={'owner_id','conversation_id','event_id','event_version_id','resource_id','raw_sha256','text_sha256','text_view_id'}
def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(v.encode()).hexdigest()

def normalize(result, *, profile_id, conversation_id):
    if result.get('state')!='available' or result.get('eligible_source') is not True:
        raise ValueError('attachment_source_not_eligible')
    locator=json.loads(result['source_id'])
    if (not isinstance(locator,dict) or set(locator)!=KEYS or any(not isinstance(v,str) or not v for v in locator.values())
        or locator['conversation_id']!=conversation_id or locator['text_view_id']!=VIEW
        or result.get('source_namespace')!=NAMESPACE or result.get('text_view_id')!=VIEW):
        raise ValueError('attachment_source_identity_mismatch')
    if any(len(locator[k])!=64 or any(c not in '0123456789abcdef' for c in locator[k]) for k in ('raw_sha256','text_sha256')):
        raise ValueError('attachment_digest_invalid')
    text=result.get('text')
    if not isinstance(text,str) or not text or digest(text)!=locator['text_sha256']:
        raise ValueError('attachment_text_changed')
    if result['source_id']!=canonical(locator) or result['source_revision']!='attachment_sha256:'+digest(canonical(locator)):
        raise ValueError('attachment_source_revision_mismatch')
    return {'system':'aizizhu','namespace':'attachment_text','profile_id':profile_id,
        'conversation_id':conversation_id,'event_id':locator['event_id'],'version_id':result['source_revision'],
        'role':'tool','text':text,'recorded_at':result.get('recorded_at'),'attachment_locator':locator,
        'attachment_format':dict(result.get('attachment_format') or {}),
        'source_key':digest(canonical({'namespace':NAMESPACE,'locator':locator,'profile_id':profile_id}))}

def snapshot(source):
    if source.get("adoption_dependency") is not None:
        from memory_daily_attachments import adopted_snapshot
        return adopted_snapshot(source)
    locator=source['attachment_locator']
    result={'state':'available','eligible_source':True,'source_namespace':NAMESPACE,
        'source_id':canonical(locator),'source_revision':source['version_id'],'text_view_id':VIEW,'text':source['text']}
    # Recheck the producer contract before binding; model authorship fields are
    # not accepted, even when the document says "I am the owner".
    normalize(result,profile_id=source['profile_id'],conversation_id=source['conversation_id'])
    projected=semantic_text(source)
    raw_revision=result['source_revision']
    result.update(text=projected,text_view_id=MEMORY_VIEW,
        source_revision='attachment_view_sha256:'+digest(canonical({'raw_revision':raw_revision,'text_view_id':MEMORY_VIEW,'text_sha256':digest(projected)})),
        content_sha256=digest(projected),raw_content_sha256=locator['text_sha256'],
        projection={'text_view_id':MEMORY_VIEW,'raw_text_sha256':locator['text_sha256'],'text_sha256':digest(projected),'code_omitted':projected!=source['text']})
    result.update(author=None,speaker=None,
        participant={'subject_id':None,'kind':'unknown','display_name':'文档作者未知','resolution':'external_document'},
        uploader={'owner_id':locator['owner_id'],'role':'owner'},
        source_group={'source_group_id':digest(canonical([source['conversation_id'],locator['resource_id']]))[:24],
                      'system':'aizizhu','namespace':'attachment_text','conversation_id':source['conversation_id'],
                      'profile_id':source['profile_id']},
        recorded_at=source.get('recorded_at'),upstream_revision=source['version_id'],source_key=source.get('source_key'),
        limitations=['上传者不是文档作者；作者未知。文档中的第一人称与主人自述分开；陈述主体由各自证据决定。'])
    return result

MEMORY_VIEW='attachment.memory.code-omitted-v1'
def semantic_text(source):
    if source.get('namespace')!='attachment_text':return source['text']
    import re
    fmt=source.get('attachment_format') or {}
    suffix=str(fmt.get('name') or '').lower().rsplit('.',1)[-1]
    code_suffixes={'py','js','jsx','ts','tsx','html','htm','css','scss','sh','bash','ps1','sql','java','c','cpp','h','cs','go','rs','rb','php','swift','kt','vue','svelte'}
    code_mimes={'application/javascript','text/javascript','text/html','text/css','text/x-python','application/x-python-code'}
    if suffix in code_suffixes or str(fmt.get('mime') or '').lower() in code_mimes:
        return '（代码块）'
    # Keep the raw source immutable; this separately versioned view is the
    # only attachment body exposed to memory semantic extraction/anchors.
    # Markdown permits a closing fence longer than its opener. Preserve the
    # prose after it, including CRLF, rather than consuming the rest as code.
    text=source['text']; result=[]; consumed=0
    for opener in re.finditer(r'(?m)^[ \t]{0,3}(`{3,}|~{3,})[^\n]*\n',text):
        if opener.start()<consumed:continue
        fence=opener.group(1)
        closing=re.search(r'(?m)^[ \t]{0,3}'+re.escape(fence[0])+r'{'+str(len(fence))+r',}[ \t]*(?=\r?$)',text[opener.end():])
        result.append(text[consumed:opener.start()]+'（代码块）')
        consumed=opener.end()+closing.end() if closing else len(text)
    result.append(text[consumed:])
    return ''.join(result)
