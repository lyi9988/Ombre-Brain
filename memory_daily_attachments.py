"""Resolve adopted attachment evidence from completed canonical turns.

Author identity remains unknown. An adopted source depends on both the exact
attachment and the effective successful assistant event that adopted it.
"""
import copy
import json
import os
import httpx
from memory_attachment_source import canonical, digest, snapshot

NAMESPACE = 'snapshot.aiz.adopted_attachment_text'
REF_KEYS = {'source_id','source_revision','body_sha256','request_id','turn_id',
            'conversation_id','profile_id','logical_request_id','attempt_id'}

def validate_ref(ref, *, profile_id, conversation_id):
    if (not isinstance(ref, dict) or set(ref) != REF_KEYS
            or any(not isinstance(v, str) or not v for v in ref.values())
            or ref['profile_id'] != profile_id or ref['conversation_id'] != conversation_id):
        raise ValueError('adoption_reference_invalid')
    locator = json.loads(ref['source_id'])
    if locator.get('conversation_id') != conversation_id:
        raise ValueError('adoption_conversation_mismatch')
    if ref['source_revision'] != 'attachment_sha256:' + digest(canonical(locator)):
        raise ValueError('adoption_attachment_revision_mismatch')
    return ref


def with_adoption(source, dependency):
    source = copy.deepcopy(source)
    ref = validate_ref(dependency['ref'], profile_id=source['profile_id'], conversation_id=source['conversation_id'])
    if (ref['source_id'] != canonical(source['attachment_locator'])
            or ref['source_revision'] != source['version_id']
            or any(not isinstance(dependency.get(k), str) or not dependency[k]
                   for k in ('event_id', 'version_id'))):
        raise ValueError('adoption_dependency_invalid')
    source['adoption_dependency'] = copy.deepcopy(dependency)
    source['source_key'] = digest(canonical({'attachment_key':source['source_key'],
        'adoption_event_id':dependency['event_id'],'adoption_event_version':dependency['version_id']}))
    return source


def adopted_snapshot(source):
    base = dict(source); dependency = base.pop('adoption_dependency')
    # Validate again rather than promoting arbitrary metadata to eligibility.
    with_adoption(base, dependency)
    result = snapshot(base)
    locator = {'attachment_locator':source['attachment_locator'], 'adoption':dependency}
    result.update(source_namespace=NAMESPACE, source_id=canonical(locator),
        source_revision='adopted_attachment_sha256:' + digest(canonical({
            'attachment_revision':result['source_revision'], 'adoption':dependency})),
        adoption_dependency=copy.deepcopy(dependency))
    return result


class AdoptedAttachmentReader:
    namespace = NAMESPACE
    def __init__(self, config):
        self.config=config
        self.options=(config.get('gateway') or {}).get('canonical_continuation') or {}
        self.profile_id=str((config.get('persona') or {}).get('profile_id') or 'haven_xiaoyu')
        self.conversation_id=self.options.get('conversation_id','conv-jiajia-main')

    async def _fetch(self, kind, source_id, version):
        token=os.environ.get('OMBRE_CANONICAL_BRIDGE_TOKEN','')
        if not self.options.get('enabled') or not token or not self.options.get('base_url'):
            return {'state':'unavailable','reason':'adoption_reader_unconfigured'}
        async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as client:
            response=await client.get(str(self.options['base_url']).rstrip('/')+'/bridge/v1/memory-source',
                params={'kind':kind,'source_id':source_id,'version':version},
                headers={'Authorization':'Bearer '+token,'X-Guyan-Device-ID':str(self.options.get('device_id') or 'ombre-gateway-operit')})
            response.raise_for_status()
            return response.json().get('source',{})

    async def capture(self, dependency):
        ref=validate_ref(dependency['ref'],profile_id=self.profile_id,conversation_id=self.conversation_id)
        event=await self._fetch('canonical_event',dependency['event_id'],dependency['version_id'])
        if (event.get('state')!='available' or event.get('eligible_source') is not True
                or event.get('source_id')!=dependency['event_id'] or event.get('source_revision')!=dependency['version_id']
                or ref not in event.get('attachment_adoption_refs',[])):
            return {'state':'unavailable','reason':'adoption_event_not_current'}
        from memory_l1_source_readers import AizAttachmentTextSourceReader
        source=await AizAttachmentTextSourceReader(self.config).capture_source(ref)
        if source.get('namespace')!='attachment_text':return source
        # The resource HTTP read is an await boundary. The successful event
        # may have been withdrawn while the file was being read.
        event=await self._fetch('canonical_event',dependency['event_id'],dependency['version_id'])
        if (event.get('state')!='available' or event.get('eligible_source') is not True
                or event.get('source_id')!=dependency['event_id'] or event.get('source_revision')!=dependency['version_id']
                or ref not in event.get('attachment_adoption_refs',[])):
            return {'state':'unavailable','reason':'adoption_event_changed_during_read'}
        return with_adoption(source,dependency)

    async def read(self, binding):
        locator=json.loads(binding['source_id'])
        if set(locator)!={'attachment_locator','adoption'}:
            return {'state':'unavailable','reason':'adoption_locator_invalid'}
        if canonical(locator['attachment_locator'])!=locator['adoption']['ref']['source_id']:
            return {'state':'unavailable','reason':'adoption_locator_mismatch'}
        source=await self.capture(locator['adoption'])
        return adopted_snapshot(source) if source.get('namespace')=='attachment_text' else source
