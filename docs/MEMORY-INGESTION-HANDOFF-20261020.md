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

Additional local review found and fixed a follow-on edge: a validated window
whose checkpoint write fails must retain its original dispatch intent, not be
treated as an invalid model response eligible for automatic retry. A dedicated
fault-injection test verifies zero replay after this failure.

Validation: 42 source-narrowing/diary-resume/ingestion tests, 7 full
ReflectionEngine runtime tests, 39 diary-service/Authority regressions passed
(88 unique tests). Actual native Schema request path was exercised with fixture
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

## Batch 2 — bounded long-chat end-to-end slice

Batch 1 final commit is `dc903ac7907c87d35cc4e3c61e675a971d312cb6`
(supersedes the initial local `c0a3e3f`, adding the validated-checkpoint failure
guard). This batch builds directly on it; integrate the two commits in order.
The second commit is the commit containing this section.

Protocol was written before implementation in [CHAT-SEGMENT-PROTOCOL.md](CHAT-SEGMENT-PROTOCOL.md).
`memory_chat_segments.py` is called only for oversized windows in the existing
Authority daily path. `memory_daily_windows.py` is unchanged. The source-binding
adapter and runtime projection supply bounded views; original text/hash and
source numbering remain intact. Candidate identity/semantic source annotations
also use the explicit support set instead of all sources in the selected turn.
No L0/L2 or Gateway/scoring file is changed.

Source-location example: budget 256, context 32, core width 128; a 600-codepoint
source has cores `[0,128)`, `[128,256)`, `[256,384)`, `[384,512)`, `[512,600)`.
The second request observes `[96,288)` plus at most 32 characters from each
adjacent source. A scene citing `[128,256)` points to the original source ID,
revision, `memory.source.exact-v1`, full `text_sha256` and exact `quote_sha256`.
Emoji remain one codepoint. When this source is second in the original window,
it stays `source:2`, including when it is the candidate's only binding.
The existing L0 copy contains the complete raw source, never the sliced view.

Resume example: one-call budget stops after one segment with raw watermark 0.
Restart reopens Authority and starts the next missing core. The last core
finishes before the existing daily writer advances event 11. Repeating a complete
task makes zero calls. A three-candidate result with quota one is drained over
three runs from the same preparation receipt, without rebilling. Same-source
unknown dispatch blocks restart, `force`, and changed generation settings.
Source version/hash changes produce a different task starting at core zero.
Auto/Review changes preserve previous receipts and candidate decisions.

Validation: 22 new tests in `tests/test_memory_chat_segments.py` passed (21 in
the final full-file run, then the corrected Prompt fixture passed separately).
68 related ordinary-window/source/ingestion/runtime/diary regressions passed;
the subsequent diary durability guard's 19-test file also passed. These groups
overlap and must not be added as independent coverage. No full suite was run.
Existing pytest `asyncio_mode` warning remains.

The new tests exercise actual ReflectionEngine, native request Schema,
source-binding/normalization, SQLite Authority, candidate preparation, original
Review/Auto writer, real local Bucket commit and the daily runner watermark.
Fixtures replace external model IO and the adoption source reader; this does
not prove a production source reader or live model's semantic quality. Cases
include raw hash/Unicode round trip, sparse source IDs, response-before-prepare
and prepare-before-cursor crashes, timeout/cancellation, `force`, concurrent
engines/CAS rollback, original edits, quota tail, empty results, invalid/unseen
ranges, context-only candidates, owner edit/accept/reject/defer, mode changes,
Prompt off and owner A/B/C wrapper, and unchanged 32000 fixture route output
budget with native client retries disabled.

Run the new file using the same interpreter/dependency environment above and
a fresh temporary `--basetemp`. `git diff --check` passed for both batches.

Migration/rollback: additional table `memory_chat_segment_progress` is created
only on first segmented task; uses the existing Authority DB, not a new store.
Its cursor, phase, response and preparation-receipt references are durable
working state. Full completion clears response payload while retaining receipt
IDs. Both new tables were exercised only in temporary test databases.
Before any future release, back up Authority and referenced source/Bucket state
consistently, test additive schema creation and reopen recovery. Roll back code
without reverting user data, retain the new tables, and prevent an old runtime
from running these incomplete/unknown tasks because it cannot honor their
receipts. No automated destructive down migration or silent receipt reset is
provided.

Not covered: manual `chat_l1` submission segmentation, attachments/actions,
cross-day scheduling/backlog draining, old-partial migration, a retry/rebuild
decision UI, retention cleanup, distant semantic dependencies beyond the bounded
context, live provider capacity/quality, production/device acceptance. Existing
deduplication still decides semantic repeats; durable execution does not claim
mathematically unique facts or globally exactly-once remote billing.

| Integration check | Batch 2 result |
| --- | --- |
| Registry/discovery | N/A: existing daily extraction, no new tool |
| Contract/call | Passed: native Schema fixture and full daily runner |
| Prompt | Passed: existing daily registry scope/off/owner wrapper; added fields are typed range protocol, no new hidden instruction prompt |
| Permissions/receipts | Passed locally: original policy/owner transitions, CAS and unknown intents; fixture source reader |
| Release/adoption | Not run: no push, merge, deployment, paid model or production migration |
