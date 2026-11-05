"""Compare a manual summary only with adopted L1s indexed by its exact sources."""
from contextlib import closing, nullcontext
from copy import deepcopy
import hashlib
import json

from memory_l1_contract import canonical_json, semantic_sha256
from memory_relation_dependencies import pending_sync_scope


def _origin(binding):
    return (binding["source_namespace"], binding["source_id"], canonical_json(binding["source_revision"]))


def _records(authority, bindings, connection=None):
    origins = list(dict.fromkeys(_origin(b) for b in bindings))
    if not origins:
        return []
    found = {}
    with (closing(authority._connect()) if connection is None else nullcontext(connection)) as db:
        if connection is None:
            db.execute("BEGIN")
        for start in range(0, len(origins), 200):
            batch = origins[start:start + 200]
            selectors = ",".join("(?,?,?)" for _ in batch)
            rows = db.execute(f"""
                WITH chosen(ns,id,rev) AS (VALUES {selectors})
                SELECT DISTINCT m.memory_id,m.active_revision,r.body_sha256,r.metadata_json
                FROM chosen JOIN memory_l1_source_members s
                  ON (s.namespace,s.source_id,s.source_revision_json)=(chosen.ns,chosen.id,chosen.rev)
                JOIN memories m ON m.memory_id=s.memory_id AND m.active_revision=s.revision
                JOIN memory_revisions r ON r.memory_id=m.memory_id AND r.revision=m.active_revision
                WHERE m.state='active' AND m.recall_policy='enabled'
            """, [value for identity in batch for value in identity])
            for mid, rev, body_sha, payload in rows:
                if mid in found or pending_sync_scope(db, mid) is not None:
                    continue
                metadata = json.loads(payload)
                doc = metadata.get("memory_l1")
                if not doc or semantic_sha256(doc) != metadata.get("memory_l1_semantic_sha256"):
                    continue
                parent = (doc.get("entry") or {}).get("parent_ref")
                if parent and not db.execute(
                    "SELECT 1 FROM memories WHERE memory_id=? AND active_revision=? "
                    "AND state='active' AND recall_policy='enabled'", (parent["memory_id"], parent["revision"])).fetchone():
                    continue
                found[mid] = (mid, rev, body_sha, doc)
    return list(found.values())


def _references(document):
    # Model-local names (source:1, newly generated entity IDs, etc.) are not
    # evidence identities. Preserve source version, speaker and exact ranges.
    refs = {}
    for b in document.get("evidence_bindings", []):
        refs[b["binding_id"]] = canonical_json([list(_origin(b)), b.get("speaker_ref"), b.get("anchors", [])])
    for e in document.get("entities", []):
        refs[e["entity_id"]] = canonical_json(_translated({k: v for k, v in e.items() if k != "entity_id"}, refs))
    for s in document.get("statements", []):
        refs[s["statement_id"]] = canonical_json([s.get("content"), _translated(s.get("evidence_refs", []), refs)])
    for scene in document.get("scenes", []):
        refs[scene["scene_id"]] = canonical_json(_translated(
            {k: v for k, v in scene.items() if k not in {"generation", "scene_id", "scene_revision_id"}}, refs))
    return refs


def _translated(value, refs):
    if isinstance(value, str):
        return refs.get(value, value)
    if isinstance(value, list):
        return [_translated(v, refs) for v in value]
    if isinstance(value, dict):
        return {k: _translated(v, refs) for k, v in value.items()}
    return value


def _facts(doc):
    refs = _references(doc)
    return {s["statement_id"]: canonical_json(_translated(
        {k: v for k, v in s.items() if k not in {"statement_id", "statement_revision_id"}}, refs))
        for s in doc.get("statements", []) if s.get("content") and s.get("evidence_refs")}


def _used_bindings(doc):
    cited = {ref for statement in doc.get("statements", []) for ref in statement.get("evidence_refs", [])}
    return [b for b in doc.get("evidence_bindings", []) if b["binding_id"] in cited]


def _extra_meaning(doc):
    refs = _references(doc)
    extras = {("scene", refs[s["scene_id"]]) for s in doc.get("scenes", [])}
    for kind in ("relations", "attributes"):
        for row in doc.get(kind, []):
            extras.add((kind, canonical_json(_translated(
                {k: v for k, v in row.items() if k != "relation_id"}, refs))))
    return extras


def related_summary_statements(authority, evidence_bindings):
    origins = {_origin(b) for b in evidence_bindings}
    result = []
    for mid, rev, _, doc in _records(authority, evidence_bindings):
        selected = {b["binding_id"] for b in doc["evidence_bindings"] if _origin(b) in origins}
        statements = [deepcopy(s) for s in doc.get("statements", [])
                      if s.get("evidence_refs") and set(s["evidence_refs"]) <= selected]
        if statements:
            result.append({"memory_id": mid, "revision": rev, "statements": statements})
    return result


def check_summary_coverage(authority, proposal, *, connection=None):
    proposal = proposal.payload() if hasattr(proposal, "payload") else proposal
    doc = proposal["metadata"]["memory_l1"]
    proposed = _facts(doc)
    all_ids = [s["statement_id"] for s in doc.get("statements", [])]
    matches, covered, existing_extras, narrative_hashes = [], set(), set(), set()
    used = _used_bindings(doc)
    origins = {_origin(b) for b in used}
    for mid, rev, body_sha, existing in _records(authority, used, connection):
        if mid == proposal.get("proposal_id"):
            continue  # Retry after commit must reconcile its own original operation.
        if not any(_origin(b) in origins for b in _used_bindings(existing)):
            continue
        old = _facts(existing)
        included = set(old.values()).intersection(proposed.values())
        covered.update(sid for sid, signature in proposed.items() if signature in included)
        if included:
            existing_extras.update(_extra_meaning(existing))
            narrative_hashes.add(body_sha)
        matches.append({"memory_id": mid, "revision": rev,
                        "matched_statement_ids": [sid for sid, value in old.items() if value in included]})
    missing = [sid for sid in all_ids if sid not in covered]
    body = str(proposal.get("proposed_body") or "").strip()
    # A thought or a qualification may exist only in the narrative. Equal
    # statements do not authorize silently throwing that extra prose away.
    literal_facts = {s["content"].strip() for s in doc.get("statements", []) if s.get("content")}
    narrative_covered = hashlib.sha256(body.encode()).hexdigest() in narrative_hashes or (
        bool(body) and all(line.strip() in literal_facts for line in body.splitlines() if line.strip()))
    complete = bool(covered) and not missing and narrative_covered and _extra_meaning(doc) <= existing_extras
    return {"status": "linked" if complete else "overlap_review" if matches else "new",
            "matches": [match for match in matches if match["matched_statement_ids"]] if complete else matches,
            "covered_statement_ids": [sid for sid in all_ids if sid in covered],
            "unmatched_statement_ids": missing}


def summary_link_receipt(authority, candidate_id):
    with closing(authority._connect()) as db:
        row = db.execute("SELECT result_json FROM candidate_decisions WHERE candidate_id=? "
                         "AND to_status='linked' ORDER BY decision_seq DESC LIMIT 1", (candidate_id,)).fetchone()
    return json.loads(row[0]).get("linked_memories", []) if row else []
