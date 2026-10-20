# Memory ingestion local handoff

## Scope and baseline

User-authorized local integration only: source narrowing, then diary partial
resume, then a separate bounded long-chat slice. No recall scoring, L0/L2
implementation, Gateway, production database, paid model, push, merge or deployment.
Independent branch `codex/memory-ingestion-batches-20261020`, parent
`63293189aaf1331b69c18383876410bf469f5b3b`; exact remote Brain feature ref matched
via read-only HTTPS. Both existing current Brain worktrees were clean. Old
staging changes were preserved. Runtime/live identity was not queried.

Applied project lesson: distinguish local/ref/live and preserve unrelated work;
only temporary Authority databases are used for progress-table verification.
The attachments are candidate source material, not additional authorization.
Their `5168463` parent has identical versions of all five affected business
files at `6329318`. Applied source-narrowing patch first, follow-on patch second;
no whole-file overwrite and no ordinary-window rewrite.

## Batch 1 — source narrowing and diary partial resume

Candidate, semantic evidence (including entity refs), scene refs/ranges form
the explicit support union. Only those sources are persisted; sparse labels
remain stable (`source:3` stays `source:3`). Unknown/missing refs fail explicitly.
This validates declared support, not the semantic truth of model citations.

Diary task checkpoints are CAS-protected in `memory_diary_task_progress` in the
existing Authority database. Example: length 600, width 256, one call per run
produces calls `[0,1,2]`, coverage `[0,256)`, `[0,512)`, `[0,600)`;
reopening the same database starts at window 1 after the first run. All three
scene anchors still point to original-text coordinates. Complete replay makes
zero new model calls. A dispatched timeout/cancellation retains an uncertain
intent and requires an explicit outcome/retry decision.

Untouched pending partial candidates can be revised by CAS. Owner edits,
accept/reject/defer and committed revisions win. Partial Auto remains Review;
an all-in-one-run complete Auto task still follows the original Auto policy.
Completed working state compacts only after canonical candidate save.

Validation: 41 source-narrowing/diary-resume/ingestion tests, 7 full
ReflectionEngine runtime tests, 39 diary-service/Authority regressions passed
(87 unique tests). Actual native Schema request path was exercised with fixture
responses. The old runtime fixture needed `base_url`/`with_options` and a
native `semantic-v1` output format; production validation was not bypassed.
Initial default Python failed due to temporary-directory access and an
incompatible `frontmatter` installation; reused project Python/dependencies.
One existing `asyncio_mode` plugin warning remains; tests explicitly run coroutines.

Reproduce from this worktree in PowerShell, choosing a fresh basetemp:

```powershell
$env:PYTHONPATH='../../staging/p4-python-deps'
& ../aizizhu-recall-r1/.venv/Scripts/python.exe -m pytest -q tests/test_candidate_source_narrowing.py tests/test_diary_partial_resume.py tests/test_memory_l1_ingestion_p2c.py tests/test_memory_l1_runtime_p2c.py tests/test_memory_diary_l1_p2b.py tests/test_memory_authority_r1.py --basetemp=C:/Users/ZJR/.codex/tmp/memory-ingestion-review --tb=short
```

Migration/rollback: the progress table is additive and lazily created; tested
only in disposable SQLite files. No production migration ran. Any future
release needs a consistent database backup and schema compatibility check.
Retain progress on code rollback; older runtime cannot honor partial/unknown
receipts, so do not resume these tasks under the older code. Never restore an
old business database to roll back code, or clear uncertain intents blindly.
No retry-decision UI, historical-partial rebuild/migration, retention policy,
cross-day scheduler, production/model-quality/device acceptance is included.

| Integration check | Result |
| --- | --- |
| Registry/discovery | N/A: no tool ID or discovery change |
| Contract/call | Passed: original extraction entry and native Schema fixture |
| Prompt | Passed: original registry/wrapper/off path retained and exercised |
| Authority/receipts | Passed: temporary DB CAS, owner races, unknown intent |
| Release/adoption | Not run: local-only authorization |
