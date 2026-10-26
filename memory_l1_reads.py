"""Owner reads of fixed L1 members, shared statements, scene prose and L0."""
from __future__ import annotations

import json
import asyncio
import time

from memory_authority import MemoryNotFound
from memory_l1_contract import L1ContractError
from memory_l1_index import INDEX_VERSION, MemoryL1Index, scene_memory_id
from memory_l1_service import L1OwnerError


def read_query(query, resource):
    allowed = {"revision"}
    if resource in ("statements", "scenes"): allowed |= {"offset", "limit"}
    if resource == "statements": allowed.add("scene_id")
    if resource == "source": allowed |= {"start", "end"}
    if set(query) - allowed:
        raise L1OwnerError("unknown_query_field", field_path="query." + sorted(set(query) - allowed)[0])
    if hasattr(query, "getlist") and any(len(query.getlist(key)) != 1 for key in query):
        raise L1OwnerError("duplicate_query_field", field_path="query")
    def integer(name, default=None, minimum=0, maximum=None):
        raw = query.get(name)
        if raw is None:
            if default is None: raise L1OwnerError("required_query_field", field_path="query." + name)
            return default
        if not isinstance(raw, str) or not raw.isascii() or not raw.isdigit():
            raise L1OwnerError("integer_required", field_path="query." + name)
        value = int(raw)
        if value < minimum or (maximum is not None and value > maximum):
            raise L1OwnerError("query_out_of_range", field_path="query." + name)
        return value
    result = {"revision": integer("revision", minimum=1)}
    if resource in ("statements", "scenes"):
        result.update(offset=integer("offset", 0), limit=integer("limit", 20, 1, 200))
    if "scene_id" in query:
        if not query["scene_id"].strip(): raise L1OwnerError("string_required", field_path="query.scene_id")
        result["scene_id"] = query["scene_id"]
    if "start" in query or "end" in query:
        result.update(start=integer("start"), end=integer("end"))
        if result["end"] <= result["start"]: raise L1OwnerError("invalid_codepoint_range", field_path="query.end")
    return result


class MemoryL1Reads:
    def __init__(self, revision_service, sources):
        self.revisions = revision_service
        self.authority = revision_service.authority
        self.index = MemoryL1Index(self.authority)
        self.sources = sources

    async def recall_source_status(self, items):
        """Verify requested bindings without returning text or retaining copies."""
        if not isinstance(items, list):
            raise L1OwnerError("array_required", field_path="items")
        started = time.perf_counter()
        requests, dependencies, epochs = [], {}, {}
        # Fixed-revision members are immutable. Share their reads only during
        # this batch; live memory/root eligibility is checked again after IO.
        bases, members = {}, {}
        member_batches, member_hits = 0, 0

        def base(memory_id, revision):
            key = (memory_id, revision)
            if key not in bases:
                bases[key] = self.base(memory_id, revision)
            return bases[key]

        def batch_members(memory_id, revision, module, ids):
            nonlocal member_batches, member_hits
            ids = list(dict.fromkeys(ids))
            missing = [key for key in ids if (memory_id, revision, module, key) not in members]
            member_hits += len(ids) - len(missing)
            if missing:
                found = self.index.members(memory_id, revision, module, missing)
                member_batches += 1
                for key in missing:
                    members[(memory_id, revision, module, key)] = found.get(key)
            return {key: members[(memory_id, revision, module, key)] for key in ids}

        def member(memory_id, revision, module, key):
            return batch_members(memory_id, revision, module, [key])[key]

        for item in items:
            required = {"memory_id", "revision", "binding_ids"}
            if (not isinstance(item, dict) or not required <= set(item) or set(item) - required - {"allow_historical"}
                    or not isinstance(item["memory_id"], str) or type(item["revision"]) is not int or item["revision"] <= 0
                    or not isinstance(item["binding_ids"], list)
                    or type(item.get("allow_historical", False)) is not bool
                    or any(not isinstance(value, str) for value in item["binding_ids"])):
                raise L1OwnerError("invalid_recall_source_request", field_path="items")
            memory = self.authority.get_memory(item["memory_id"])
            if (not memory or memory["state"] != "active" or memory["recall_policy"] != "enabled"
                    or (not item.get("allow_historical") and memory["active_revision"] != item["revision"])):
                requests.append((item, {}, "memory_not_current_or_eligible"))
                continue
            epochs[(item["memory_id"], item["revision"])] = memory["active_revision"]
            try:
                result, identity = base(item["memory_id"], item["revision"])
            except MemoryNotFound:
                requests.append((item, {}, "revision_unavailable"))
                continue
            if not identity:
                requests.append((item, {}, "legacy"))
                continue
            root_id, root_revision, _ = self._shared_root(
                item["memory_id"], item["revision"], identity,
                base_reader=base, member_reader=member)
            root = self.authority.get_memory(root_id)
            if (not root or root["state"] != "active" or root["recall_policy"] != "enabled"
                    or (not item.get("allow_historical") and root["active_revision"] != root_revision)):
                requests.append((item, {}, "dependency_changed"))
                continue
            dependencies[(item["memory_id"], item["revision"])] = (root_id, root["active_revision"])
            bindings = batch_members(root_id, root_revision, "evidence_bindings", item["binding_ids"])
            requests.append((item, bindings, "ok"))
        semaphore = asyncio.Semaphore(8)
        tasks = {}
        async def check(binding):
            async with semaphore:
                return await self.sources.locate(binding, verify_only=True)
        for _item, bindings, _state in requests:
            for binding in bindings.values():
                if binding:
                    key = json.dumps(binding, sort_keys=True, ensure_ascii=False)
                    if key not in tasks:
                        tasks[key] = asyncio.create_task(check(binding))
        preparation_ms = max(0, int((time.perf_counter() - started) * 1000))
        source_started = time.perf_counter()
        try:
            if tasks:
                await asyncio.gather(*tasks.values())
            source_ms = max(0, int((time.perf_counter() - source_started) * 1000))
            final_started = time.perf_counter()
            results = []
            for item, bindings, state in requests:
                latest = self.authority.get_memory(item["memory_id"])
                if (not latest or latest["state"] != "active" or latest["recall_policy"] != "enabled"
                        or latest["active_revision"] != epochs.get((item["memory_id"], item["revision"]), item["revision"])):
                    state = "memory_not_current_or_eligible"
                dependency = dependencies.get((item["memory_id"], item["revision"]))
                if dependency:
                    root = self.authority.get_memory(dependency[0])
                    if (not root or root["state"] != "active" or root["recall_policy"] != "enabled"
                            or root["active_revision"] != dependency[1]):
                        state = "dependency_changed"
                statuses = {}
                for key, binding in bindings.items():
                    if not binding:
                        statuses[key] = {"state": "unavailable", "eligible_source": False,
                                         "reason": "binding_not_in_revision"}
                    else:
                        statuses[key] = tasks[json.dumps(binding, sort_keys=True, ensure_ascii=False)].result()
                results.append({"memory_id": item["memory_id"], "revision": item["revision"],
                                "state": state, "sources": statuses,
                                "historical": bool(latest and latest["active_revision"] != item["revision"])})
                from memory_relation_dependencies import dependency_snapshot
                connection = self.authority._connect()
                try:
                    dependency = dependency_snapshot(connection, item['memory_id'])
                finally:
                    connection.close()
                results[-1].update(dependency_token=dependency['token'],
                    l1_sync_required=dependency['l1_sync_required'], sync_scope=dependency['sync_scope'],
                    relation_guards=dependency['guards'])
            return {"items": results, "diagnostics": {
                "timing_ms": {"prepare": preparation_ms, "source_reads": source_ms,
                              "final_authority": max(0, int((time.perf_counter() - final_started) * 1000))},
                "member_batches": member_batches, "member_cache_hits": member_hits,
                "unique_binding_reads": len(tasks),
            }}
        finally:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks.values(), return_exceptions=True)

    def base(self, memory_id, revision):
        memory = self.authority.get_memory(memory_id)
        if not memory: raise MemoryNotFound(memory_id)
        if type(revision) is not int or revision <= 0: raise L1OwnerError("invalid_revision", field_path="revision")
        try: identity = self.index.identity(memory_id, revision)
        except L1ContractError as exc:
            if exc.code == "revision_not_found": raise MemoryNotFound(memory_id) from None
            raise
        result = {"status": "ok", "memory_id": memory_id, "revision": revision,
                  "current_revision": memory["active_revision"], "projection_digest": identity["content_digest"] if identity else None,
                  "index_version": INDEX_VERSION if identity else None, "restrictions": []}
        if not identity: result["restrictions"].append("旧版本没有L1结构；保留原读取与编辑，不自动补标。")
        if revision != memory["active_revision"]: result["restrictions"].append("此为固定历史版本，当前采用版本另示。")
        if memory["state"] != "active": result["restrictions"].append("本记忆当前生命周期为" + memory["state"] + "；owner查看不授予聊天采用资格。")
        conn = self.authority._connect()
        try:
            coverage = conn.execute("SELECT json_extract(metadata_json,'$.diary_l1_bundle.coverage') FROM memory_revisions WHERE memory_id=? AND revision=?",
                                    (memory_id, revision)).fetchone()[0]
        finally: conn.close()
        if coverage:
            result["coverage"] = json.loads(coverage)
            if result["coverage"]["state"] != "complete": result["restrictions"].append("此版本第一遍输入覆盖不完整；原文全文仍独立保留。")
        return result, identity

    def _shared_root(self, memory_id, revision, identity, *, base_reader=None, member_reader=None):
        entry = identity["entry"]
        parent, scene_ref = entry.get("parent_ref"), entry.get("scene_ref")
        if not parent or not scene_ref:
            return memory_id, revision, None
        root_id, root_revision = parent["memory_id"], parent["revision"]
        (base_reader or self.base)(root_id, root_revision)
        scene = (member_reader or self.index.member)(root_id, root_revision, "scenes", scene_ref["scene_id"])
        if (not scene or scene["scene_revision_id"] != scene_ref["scene_revision_id"]
                or scene["statement_refs"] != entry.get("shared_statement_refs", [])):
            raise L1OwnerError("scene_dependency_mismatch", status=409, field_path="entry.scene_ref")
        return root_id, root_revision, scene

    def page(self, memory_id, revision, module, *, offset=0, limit=20, scene_id=None):
        result, identity = self.base(memory_id, revision)
        if module not in ("scenes", "statements"): raise L1OwnerError("invalid_module", field_path="module")
        if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= 200:
            raise L1OwnerError("invalid_pagination", field_path="page")
        root_id, root_revision, shared_scene = self._shared_root(memory_id, revision, identity) if identity else (memory_id, revision, None)
        if module == "statements" and shared_scene:
            if scene_id is not None and scene_id != shared_scene["scene_id"]:
                raise L1OwnerError("scene_dependency_mismatch", field_path="query.scene_id")
            scene_id = shared_scene["scene_id"]
            result["root_ref"] = {"memory_id": root_id, "revision": root_revision}
        else: root_id, root_revision = memory_id, revision
        if scene_id is not None and not self.index.member(root_id, root_revision, "scenes", scene_id):
            raise MemoryNotFound(scene_id)
        items, total = self.index.page(root_id, root_revision, module, offset=offset, limit=limit, scene_id=scene_id)
        if module == "scenes":
            items = [{**item, "child": self.child_summary(memory_id, revision, item)} for item in items]
        result.update(items=items, page={"offset": offset, "limit": limit, "total": total,
                                      "next_offset": offset + len(items) if offset + len(items) < total else None})
        return result

    def child_summary(self, root_id, root_revision, scene):
        child_id = scene_memory_id(root_id, scene["scene_id"])
        memory = self.authority.get_memory(child_id)
        conn = self.authority._connect()
        try:
            row = conn.execute("""SELECT revision FROM memory_revisions WHERE memory_id=?
                AND json_extract(metadata_json,'$.memory_l1.entry.parent_ref.memory_id')=?
                AND json_extract(metadata_json,'$.memory_l1.entry.parent_ref.revision')=?
                AND json_extract(metadata_json,'$.memory_l1.entry.scene_ref.scene_id')=?
                AND json_extract(metadata_json,'$.memory_l1.entry.scene_ref.scene_revision_id')=?
                ORDER BY revision DESC LIMIT 1""", (child_id, root_id, root_revision, scene["scene_id"], scene["scene_revision_id"])).fetchone()
        finally: conn.close()
        state = "ready" if row else "stale" if memory else "not_prepared"
        if memory and memory["state"] != "active": state = "stale"
        issue = None
        if not row:
            statuses = self.authority.list_projection_status(root_id)
            relevant = [s for s in statuses if s["memory_revision"] == root_revision and s["projector"] == "l1_scene:" + scene["scene_id"]]
            if relevant and relevant[-1]["status"] == "degraded":
                state = "failed"
                issue = {"code": relevant[-1].get("details", {}).get("error_code", "scene_preparation_failed")}
        return {"state": state, "memory_id": child_id, "revision": row[0] if row else None,
                "current_revision": memory["active_revision"] if memory else None, "issue": issue}

    async def scene(self, memory_id, revision, scene_id):
        result, _ = self.base(memory_id, revision)
        scene = self.index.member(memory_id, revision, "scenes", scene_id)
        if scene is None: raise MemoryNotFound(scene_id)
        child = self.child_summary(memory_id, revision, scene)
        if child["state"] == "ready":
            child["detail"] = await self.revisions.detail(child["memory_id"], child["revision"])
        # Load only referenced statements, retaining their scene order. No
        # parent body or unrelated statements need be deserialized here.
        statements = [self.index.member(memory_id, revision, "statements", ref) for ref in scene["statement_refs"]]
        if any(item is None for item in statements): raise L1OwnerError("scene_dependency_mismatch", status=409, field_path="scene.statement_refs")
        binding_ids = set()
        def collect(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "evidence_refs": binding_ids.update(value)
                    else: collect(value)
            elif isinstance(node, list):
                for value in node: collect(value)
        collect(statements)
        bindings = [self.index.member(memory_id, revision, "evidence_bindings", ref) for ref in sorted(binding_ids)]
        result.update(root_ref={"memory_id": memory_id, "revision": revision}, scene=scene,
                      child=child, shared_statements=statements, sources=[b for b in bindings if b])
        return result

    def source_list(self, memory_id, revision):
        result, identity = self.base(memory_id, revision)
        root_id, root_revision, scene = self._shared_root(memory_id, revision, identity) if identity else (memory_id, revision, None)
        # Source bindings are explicit metadata, never an implicit read of all
        # external diaries. Actual text reads happen only at locate().
        conn = self.authority._connect()
        try:
            rows = conn.execute("SELECT content_json FROM memory_l1_members WHERE memory_id=? AND revision=? AND module='evidence_bindings' ORDER BY ordinal",
                                (root_id, root_revision)).fetchall()
        finally: conn.close()
        result["sources"] = [json.loads(row[0]) for row in rows]
        if scene: result["root_ref"] = {"memory_id": root_id, "revision": root_revision}
        return result

    async def relation_material(self, memory_id, revision):
        """Exact adopted counterparts for L2; unrelated prose is never loaded."""
        from memory_relation_dependencies import relation_context, dependency_snapshot
        from memory_l1_contract import semantic_sha256
        from memory_query_evidence import semantic_source_refs
        conn = self.authority._connect()
        try:
            context = relation_context(conn, [memory_id])
            dependency = dependency_snapshot(conn, memory_id)
        finally:
            conn.close()
        refs = {}
        for relation in context['relations']:
            for side in ('from', 'to'):
                key = (relation[side + '_memory_id'], relation[side + '_revision'])
                if key != (memory_id, revision):
                    refs.setdefault(key, []).append({'relation_id': relation['relation_id'], 'kind': relation['kind'],
                        'object_type': relation[side + '_type'], 'object_id': relation[side + '_id']})
        updates, issues = [], []
        for (mid, rev), paths in sorted(refs.items()):
            memory = self.authority.get_memory(mid)
            record = self.authority.get_memory_revision(mid, rev)
            if not memory or memory['state'] != 'active' or memory['recall_policy'] != 'enabled' or not record:
                issues.append({'memory_id': mid, 'revision': rev, 'state': 'unavailable'})
                continue
            document = (record['metadata'].get('memory_l1') or {})
            if not document:
                issues.append({'memory_id': mid, 'revision': rev, 'state': 'semantic_revision_unavailable'})
                continue
            whole = any(path['object_type'] == 'memory' for path in paths)
            ids = {path['object_id'] for path in paths if path['object_type'] == 'statement'}
            for path in paths:
                if path['object_type'] == 'scene':
                    scene = next((s for s in document.get('scenes', []) if s['scene_id'] == path['object_id']), None)
                    if scene:
                        ids.update(scene['statement_refs'])
            statements = [s for s in document.get('statements', []) if whole or s['statement_id'] in ids]
            selected = {**document, 'statements': statements,
                'scenes': [s for s in document.get('scenes', []) if set(s['statement_refs']) & {v['statement_id'] for v in statements}],
                'relations': [r for r in document.get('relations', []) if r['relation_id'] in {p['relation_id'] for p in paths}]}
            referenced = semantic_source_refs(selected)
            bindings = [b for b in document.get('evidence_bindings', []) if not referenced or b['binding_id'] in referenced]
            status = (await self.recall_source_status([{'memory_id': mid, 'revision': rev,
                'binding_ids': [b['binding_id'] for b in bindings], 'allow_historical': True}]))['items'][0]
            if status['state'] != 'ok' or any(s.get('eligible_source') is not True for s in status['sources'].values()):
                issues.append({'memory_id': mid, 'revision': rev, 'state': status['state'],
                    'source_states': {key: value['state'] for key, value in status['sources'].items()}})
                continue
            updates.append({'memory_id': mid, 'revision': rev, 'historical': memory['active_revision'] != rev,
                'relations': paths, 'statements': statements, 'scenes': selected['scenes'],
                'entities': document.get('entities', []), 'source_refs': bindings,
                'sync_scope': status.get('sync_scope')})
        result = {'updates': updates, 'guards': context['guards'], 'issues': issues,
            'dependency_token': dependency['token']}
        result['token'] = semantic_sha256(result)
        return result

    async def locate(self, memory_id, revision, binding_id, *, start=None, end=None):
        result, identity = self.base(memory_id, revision)
        root_id, root_revision, _ = self._shared_root(memory_id, revision, identity) if identity else (memory_id, revision, None)
        binding = self.index.member(root_id, root_revision, "evidence_bindings", binding_id)
        if binding is None: raise MemoryNotFound(binding_id)
        result["source"] = await self.sources.locate(binding, start=start, end=end)
        return result
