# RECALL-R1 authoritative architecture

Status: core authority contracts retained; natural semantic retrieval accepted
on 2026-09-30 is implemented in this feature lineage, not pending design.
Section 9 is the current retrieval contract; older fixed-trigger Fast/Deep
milestones below are historical, not rules for ordinary owner chat. The
20261001 timeout release records Gateway `54ad5ad` as deployed; reranker
failure/cache/attempt-telemetry repairs below are local and not yet released.
Owner acceptance, general retrieval quality and latency are still open.
The detailed companion document is
`coordination/active/20260930-recall-r1-natural-retrieval-design.md` at the
workspace root. Deployment facts belong to the RECALL-R1 implementation
report, not to this design document.

### Historical implementation note: 2026-09-29 correction

This describes the deployed `dde985f` behavior, not the accepted target design
in section 9. Its keyword/intent-gated Deep fallback and limited pronoun
context are not sufficient evidence of natural Recall quality.

The request-local retrieval query may include the nearest previous user
message for an anaphoric follow-up (a person, object, event, or causal
follow-up). Only messages already present in the request are considered;
assistant/tool messages never establish an antecedent. The previous user
message must be within eight message positions and at most 500 characters.
Acknowledgements stop the lookup. This context is a search cue, not an alias
write, a resolved identity, or a change to canonical/chat messages. The trace
records message positions, length, reason, and query hash.

An unclassified query starts with local Fast retrieval. If Fast selects
nothing and has an eligible weak candidate or a memory-seeking/contextual
cue, the same selector may run Deep once. A successful Fast result avoids
that work. Deep uses existing embedding, reranker and optional Query Planner;
it does not call the main conversation model for a routing decision. The
fallback has a 12-second default deadline, configurable through
`gateway.recall_deep_fallback_timeout_seconds` (clamped to 1–30 seconds).
Timeout returns the empty Fast result with an explicit diagnostic. External
cancellation propagates and cancels/joins any parallel Planner task.

Deep graph retrieval permits bucket reranking before bucket admission. A
moment reranker that runs only after admission cannot provide evidence for
a bucket already rejected. Provider enabled flags, Authority active/enabled
policy, evidence thresholds, archive/privacy restrictions and Composer's
single `ombre.memory_recall` projection remain authoritative. Fast cannot
run remote Semantic Rescue. Diagnostics preserve the Fast pass and the
Deep result separately. This correction requires natural owner acceptance;
passing local tests does not establish production recall quality.

## 1. Purpose

RECALL-R1 converges the existing Ombre memory writers and recall paths without
removing the current Bucket Markdown/YAML format or changing unrelated tool,
Dream, Darkroom, Reminder, Emotion, Needs, Goal, or canonical-conversation
authorities.

### Terminology: runtime data writes are not deployment

RECALL-R1 uses same-filesystem temporary files and atomic replacement when a
live Memory Bucket or immutable revision snapshot changes.  This protects one
runtime data commit from a crash; it is **not** the production deployment
strategy.

Production releases must come from clean, exact commits as complete immutable
release directories/images with manifests, revision labels, deployment-ledger
entries, and a `current` pointer (or container-image switch).  Per-file upload
and replacement is emergency repair only.  Git refs, release trees, host
checkouts, images, running containers, service health, owner acceptance, and
`production/current` remain separate evidence surfaces.

The product outcome is a natural Auto/Review memory experience backed by one
auditable commit path, owner-confirmed entity aliases, revision history,
rebuildable derived indexes, and local evidence plus ordinary semantic retrieval
with optional bounded Deep expansion.

## 2. Authorities

| Domain | Authority | Non-authorities |
|---|---|---|
| Canonical conversation source | Aizizhu `ConversationStore.events` | Ombre `raw_events`, `conversation_turns`, UI projections |
| Active memory body | Ombre Bucket Markdown/YAML, written only through `MemoryCommitService` | Candidate JSON, indexes, Composer, Reality |
| Candidate/decision/revision control | `MemoryAuthorityStore` SQLite | Bucket comments, model output, UI local state |
| Entity and owner-confirmed aliases | `MemoryAuthorityStore` revisioned entity/alias facts | retrieval aliases, Word Map, model guesses |
| Memory injection placement | Aizizhu Prompt Composer | Memory store, Reality |
| Derived recall indexes | Rebuildable projections keyed by memory revision | Never facts or write authorities |
| Darkroom | `DarkroomStore` | Memory Bucket and Recall |
| Reminder/Todo | `ReminderStore` | Memory Bucket and Recall indexes |
| Dream | `DreamEngine` records and claim log | Memory facts; persistence requires an explicit MemoryProposal |
| Emotion/Needs | Existing Persona/Needs authorities | Recall routing and admission |

## 3. Existing Ombre compatibility map

| Capability | RECALL-R1 treatment |
|---|---|
| `breath` | Read committed memories through Fast/Deep Recall |
| `read_bucket` | Read current body, revision metadata, and rings |
| `hold` | Normalize a single MemoryProposal; source-bound `feel` becomes a ring |
| `grow` | Normalize multiple proposals; never direct-merge a Bucket |
| `comment_bucket` | Append an immutable ring to the parent memory |
| `profile_fact` | Produce a profile/entity proposal; commit through the same service |
| `trace` | Create a new body/metadata revision or change recall/archive policy |
| `pulse` | Owner-safe authority/index health projection |
| `introspection` | Read memory; append rings or resolve through commit actions |
| `darkroom_*` | Remains separate, private, locked, revisioned authority |
| Reminders/Todos | Remain separate action-state authority; may reference memory IDs |
| Dream | Remains latent/non-factual; may explicitly produce a proposal |
| `whisper` | Remains a private domain and Dream material, not normal recall |
| `self_anchor` | Remains a protected identity entry point |

No listed capability may disappear or silently change semantics during M1.

## 4. Memory aggregate

```text
Memory
├── current body and metadata
├── immutable body revisions
├── immutable rings/comments
├── verified source references
├── revisioned entities and aliases
├── recall policy (enabled/manual_only/disabled)
└── derived-index health by memory revision
```

A body revision corrects or replaces the active factual/narrative statement.
A ring adds a later feeling, interpretation, or supplement without replacing
the original statement. A ring is a child of its parent memory and must not be
returned as a duplicate top-level memory.

## 5. Ingestion pipeline

```text
source writer
  -> MemoryProposal normalizer
  -> deterministic MemoryIngestionPolicy
       reject | queue_review | auto_accept
  -> owner decision when required
  -> MemoryCommitService
  -> Bucket + immutable revision/ring + SQLite transaction/outbox
  -> rebuildable projections
```

Writers describe evidence and proposed meaning. They do not choose a final
Bucket, mutate candidate state, raise alias trust, or update indexes.

## 6. Policy boundary

`MemoryIngestionPolicy` is a pure deterministic function. It may read only the
proposal, versioned owner policy, duplicate/authority metadata, and verified
source status. It has no database, filesystem, model, BucketManager, tool, or
network access.

The model may propose narrative text, types, entities, and aliases. It may not
grant itself Auto approval, owner alias trust, or a write capability.

Owner policy is versioned by source and memory type. Auto remains automatic;
Review remains owner-confirmed; Off remains disabled. Unknown/sensitive cases
default to Review, not silent acceptance or permanent rejection.

## 7. Commit state and recovery

For a body revision, the commit journal moves through:

```text
prepared -> body_written -> committed -> projected | degraded
```

The service validates `expected_revision`, the current Bucket body hash,
verified sources, and the idempotency key. Revision snapshots and active Bucket
updates use same-filesystem temporary files and atomic replacement. Candidate
decision, active revision pointer, source links, canonical entity/alias facts,
and a unique outbox event commit in one SQLite transaction. Startup reconcile
finishes or safely rolls back interrupted `prepared/body_written` operations.

Projection failure never rolls back a committed fact. It records `degraded`,
keeps the exact revision/outbox identity, and is retryable without generating
new memory text.

## 8. Non-negotiable invariants

1. A memory has at most one active body revision.
2. A committed body revision and a committed ring are immutable.
3. Restoring old content creates a new monotonic revision.
4. All memory writes pass through `MemoryCommitService`.
5. No business writer directly creates, merges, updates, comments, or deletes a Bucket.
6. Auto and Review use the same commit implementation.
7. Equal idempotency key plus equal payload returns the existing result.
8. Equal idempotency key plus different payload is a conflict.
9. Unverified sources cannot create a committed memory except explicit owner-authored input.
10. Owner-confirmed aliases survive every derived-index rebuild.
11. Automatic aliases are weak evidence until an owner or trusted source confirms them.
12. A projection/index never becomes a fact authority.
13. Darkroom, Dream, Reminder, Emotion, Needs, and Goals cannot directly mutate Memory.
14. Memory cannot directly mutate those authorities; cross-domain work uses typed references/events.
15. Emotion/Needs cannot trigger Recall or lower memory admission thresholds.
16. Internal planner prompts never enter the main chat context.
17. Migration invokes no model and produces no new semantic memory.
18. Recall emits stable route/admission/rejection reason codes.

## 9. Local Fast, ordinary semantic retrieval, and optional Deep

These are costs within the existing Recall pipeline, not three independent
Memory runtimes. Fast retains local anchors, aliases, entities and caches.
Ordinary meaningful chat receives a semantic retrieval opportunity without
requiring an explicit recall phrase, a keyword hit, a Deep route, or a
particular `context_mode`. Deep is optional query expansion or disambiguation
when the candidate evidence actually needs it. An empty result alone cannot
cause repeated Deep calls.

The default semantic opportunity applies to owner-facing chat scope. Existing
internal Planner, embedding/reranker, Memory ingestion/review, Diary/Digest,
and background consumers retain their own request scopes and cannot re-enter
main-chat Recall automatically, even through the same Gateway/provider.
Scope comes from trusted runtime request context, not keywords in a prompt.
Internal requests remain observable children without canonical chat writes.

One request-local RecallInput contains every staged user event in order,
bounded recent conversation with source versions, identity/conversation scope,
and context revision. The default recent scope is four completed dialogue
turns; the owner may adjust scope and budget. All current user events remain
represented. Long inputs require accounted-for segmentation, not silent loss
of the last or first message. Assistant history supplies search cues only,
never confirmed aliases or facts. Candidate generation, rerank, entity/topic
checks and admission consume the same RecallInput.

The target query views are `q_current` (all current user input) and `q_context`
(that input with bounded background). Ordinary chat embeds only `q_context`;
`q_current` remains the independent local evidence channel. Dual vectors are
optional expansion, not the default cost of every ordinary message. Long
current inputs require recorded chunks. Actual provider calls, vector counts,
input coverage and costs are observable. No main-model routing call is added.
Existing local channels and query embedding run concurrently. Empty or stale
indexes are diagnosed before spending a query request; they are incomplete,
not proof that Memory contains no match.
Identity, privacy, recall policy and active/index revision eligibility are
filtered before similarity Top-K, not after a global Top-K has discarded
eligible memories. Stored candidate vectors are never regenerated in chat.

Candidates from Memory/Moment/Ring and local/semantic channels merge with
parent Memory IDs and revisions. Authorized semantic candidates may establish
relevance without a literal phrase match; a `semantic_only` label is not by
itself a rejection. Identity equivalence and source privacy remain separate
constraints. Moment relevance cannot require its parent Bucket to have
already passed a lexical admission gate. The final projection groups records
by parent Memory and preserves ring/time/interpretation semantics.

Clear evidence can be selected directly. Before the final rerank, unresolved
entity/relation or query-coverage ambiguity may invoke the existing Query
Planner at most once, with the same current input, recent context and
necessary candidate evidence. Supplemental queries retain the original query
and feed the same selector. The merged bounded pool uses at most one rerank
round; an empty rerank output does not restart Deep. No recursive Planner or
second fact authority is introduced.
Provider/Planner cancellation and one overall recall deadline propagate.
Failures preserve available evidence and allow chat to continue.

The 2026-10-01 real-request regression exposed a later-rerank deadline
discarding an already completed semantic lookup. Deadline fallback now keeps
a request-local checkpoint of verified hits and revalidates both current and
context query hashes, current eligibility, and Memory ID/revision/body hash
before reusing scores. Partial/stale/disabled hits are excluded. It performs
no additional embedding/rerank call and does not claim that rerank succeeded.
External cancellation still propagates; timeout remains an incomplete reason.
This correctness repair is not evidence of a lower end-to-end latency.

### Reranker failure and timeout telemetry repair (not yet deployed)

Each rerank call owns its outcome rather than reading mutable last-request
health. HTTP failures, malformed/empty provider responses and partial result
sets are not successful no-matches and cannot populate the success cache.
Valid partial rows remain usable with `reranker_unavailable` in incomplete
reasons; no admission threshold is lowered and no extra retry is introduced.
Only finite scores with unique in-range indices are accepted.

Owner reranker configuration updates advance a process-local cache generation.
New requests cannot join old-config inflight work or reuse its cache; existing
requests keep their captured engine/config. Health distinguishes the current
model from the model of the last request and marks config identity mismatch.
Credentials and the private comparison fingerprint are never diagnostic fields.
The tested per-call httpx transport is retained, not silently pooled again.

Mounted credential activation is per named key, not per whole-file timestamp.
The initial mounted named credential supersedes a stale startup/config value;
later unrelated file or model changes preserve an explicit live owner override.
A changed named key activates on reload. An explicit empty/deleted managed key
stays cleared rather than inheriting an unrelated embedding credential. Missing
dedicated keys at initial startup retain the existing fallback contract.
Unreadable/malformed credential input preserves previous credentials, reports
the limitation and retries on the next reload. Diagnostic source labels and
equality-to-observed-mount booleans contain no key or key fingerprint.

The natural candidate builder evaluates its existing eligibility contract
directly, without first constructing legacy pools that it immediately replaces.
This removes duplicate predicates, not source checks or admission thresholds.
Revision-verified fresh fragment parsing is retained; a separate reuse design
is required before removing that correctness boundary.

Timeout diagnostics preserve initial and local-fallback attempts separately,
with phase, start offset, duration, numeric/count stages, cleanup and selection
wall time. First-pass timing keys are prefixed on fallback so durations cannot
be overwritten by the second pass. Cancelled final-rerank work retains its
elapsed time. Attempt envelopes include their child stages: do not add all
numbers as independent durations, and do not treat fallback snapshot evidence
as another provider call. This does not measure event-loop lag or device TTFT.

`selected`, `no_match`, `incomplete`, and `disabled` are distinct. A timeout,
missing index or uncovered query input cannot support a claim that a person
or event does not exist. Status-only wrapper text is not an injected Memory
item. Internal embedding/Planner/Rescue templates have editable internal
Source Registry scopes; only the one `ombre.memory_recall` projection enters
the owner's main-chat Composer. Emotion/Needs state does not lower evidence
thresholds, write aliases, or drive a Recall feedback loop.

Latency is a measured tradeoff: formerly skipped semantic work may add cost
to ordinary chat. Parallelism, vector/result caches and parent prepare
snapshots must remove duplicate work, but do not establish a fixed latency
promise. Owner acceptance covers current and paraphrased queries, multi-user
batches, pronouns, preferences, commitments, emotional experiences, topic
changes, ambiguity and negative/privacy cases. Inspector must show actual
selected/injected items and rejection reasons alongside first-token and total
timings. Local test counts or health codes do not close Recall quality.

## 10. Revision-aware cache contract

Caches are performance projections, never facts.  Every entry is keyed by the
authoritative revisions that can change its answer and is invalidated by the
same outbox/watermark stream used for derived indexes.

### Cache layers

1. **Prepare snapshot reuse**: preserve the current R5 parent-request snapshot
   so tool continuations do not rebuild Recall, Persona, Worldbook, and other
   stable request projections.
2. **Bucket/authority hot metadata**: in-process LRU keyed by memory ID plus
   active revision/body hash.  It contains metadata and bounded excerpts, not
   an alternate memory body authority.
3. **Query embedding exact cache**: normalized-query hash + embedding provider,
   model revision, dimensions, normalization revision.  Raw private query text
   is not required in the cache key.
4. **Planner exact cache**: current-query hash + bounded recent-context hashes +
   Fast-evidence hash + planner route revision + planner prompt revision.
5. **Rerank exact cache**: query hash + ordered candidate IDs/revision hashes +
   reranker provider/model revision + scoring/prompt revision. The current
   failure repair also isolates runtime configuration generations and stores
   only nonempty, completely validated successful provider results.
6. **Final retrieval-plan cache**: identity scope + query hash + Memory authority
   watermark + Alias/Entity watermark + derived-index watermark + Recall policy
   revision.  It has a short TTL and never stores the final assistant answer.

Same-key concurrent misses use single-flight so one provider request populates
the cache while peers await the same result.  Cache hits/misses, age, key
version, and invalidation reason are owner-safe Inspector metadata.

### What must not be cached as a reusable answer

- final companion responses;
- owner-visible emotional wording;
- live Emotion/Needs state as Recall evidence;
- uncommitted candidates;
- Darkroom body;
- Reminder state without its own revision;
- Dream surfacing claims;
- any result whose authority/index watermark is unknown.
- failed, empty-error or partially validated reranker responses.

Semantic caching of final LLM replies is intentionally excluded: a similar
sentence can occur under different relationship, Memory, tool, or emotional
state.  Semantic similarity remains retrieval evidence, not response reuse.

## 11. Model routing and prompts

Prompt scopes:

- `memory.query_planner`
- `memory.domain_sentinel`
- `memory.semantic_rescue`
- `ombre.memory_recall` (dynamic injection block)

Model routes:

- `memory_query_planner`
- `memory_domain_sentinel` (local/remote disabled by default)
- `memory_semantic_rescue` (disabled by default)
- `autonomy_reasoner` (reserved and disabled; unrelated to Recall)

Reality edits route choice through the Aizizhu Model Registry. Ombre consumes a
verified route mirror identified by route revision/hash; provider secrets remain
in runtime configuration. Prompt Composer controls prompt text and placement,
not model choice or memory facts.

### Prompt Composer compatibility contract

RECALL-R1 exposes exactly one main-chat dynamic source adapter:

```text
source_id: ombre.memory_recall
authority: Ombre Memory/Recall
body_mode: dynamic
```

The adapter returns a request-local `MemoryRecallProjection` containing:

- authority watermark and Recall policy revision;
- selected memory IDs and exact active revisions;
- selected ring IDs, if any;
- route (`fast` or `deep`) and stable reason codes;
- bounded owner-safe body plus token estimate;
- source/body hashes for Inspector provenance;
- cache/snapshot identity.

Ombre owns which committed Memory evidence qualifies and the projection body.
Prompt Composer owns whether the block is included and its role, lane, anchor,
depth, wrapper, priority, and token budget. Reality owns neither.

The dynamic adapter preserves existing Prompt Composer preset/binding
contracts. A Memory schema upgrade cannot create a second prompt insertion,
move the block implicitly, overwrite an owner wrapper, or change protected
protocol/history/tool nodes. Missing/degraded Recall produces an explicit
empty/degraded source result; it must not inject a stale cached body under a new
authority watermark.

`talk.continuation` reuses the exact parent `MemoryRecallProjection` through the
prepare snapshot unless a deliberate new Recall round is requested. It cannot
silently recall a different Memory set between a tool call and the final answer.

Internal requests remain separate Composer scopes:

- `memory.query_planner`
- `memory.domain_sentinel`
- `memory.semantic_rescue`

Their prompts and outputs never appear as main-chat history or as extra
`ombre.memory_recall` blocks. Model Request Trace/Inspector must show the final
physical order and prove that the Composer preview, compiled projection, and
raw request agree on source ID, revision/hash, role, position, and token count.

## 12. Owner UI contract

Reality exposes separate, collapsible sections under `更多 -> 中枢`:

- Memory: pending, remembered, people/aliases, recall diagnostics.
- Darkroom: private locked rooms and revisions.
- Dreams: latent/surfaced dream records.
- Goals and reminders: action state, not memory.
- Model routing: route-first model selection, with provider/model administration in secondary tabs.

Repeated items, revisions, rings, raw Markdown/YAML, and advanced index details
are collapsed by default. Mobile layouts show one primary editor at a time and
must remain usable at 320/390/430px.

## 13. Performance and quality acceptance

RECALL-R1 must compare the same owner query before/after with Inspector and
Gateway timing evidence.  It reports at minimum:

- routing decision (ordinary semantic route / optional Deep / explicit skip,
  with local evidence channels distinguished from remote work);
- prepare/snapshot time;
- embedding/planner/rerank request count and time;
- cache hit/miss by layer;
- candidates found/admitted/rejected and reason codes;
- Memory block actually injected;
- model first-byte and total time.

Expected direction, not an invented fixed SLA:

- ordinary meaningful owner chat gets a semantic opportunity without needing
  a keyword or memory-intent trigger; verified cache or explicit scope/policy
  skips may avoid a remote request, but a guessed non-memory topic may not;
- identity/date/anchor evidence remains a local channel, not a rule that
  suppresses the ordinary semantic opportunity;
- ambiguous candidate evidence may pay one bounded Planner/Deep expansion;
- a tool continuation reuses the parent prepare snapshot;
- no query performs repeated provider embedding/rerank for the same revisioned
  evidence within one logical turn.

Upstream model or browse latency remains visible and separate; RECALL-R1 must
not claim to fix provider latency it does not control.

### Request critical path and bounded parallelism

Prepare timing is measured as a dependency graph, not as a misleading sum of
independent stage durations.

The broader concurrency target is to capture an immutable request snapshot of configuration,
Composer binding, Memory authority/index watermarks, and identity scope.  The
following independent reads may then run concurrently under one cancellation
and deadline budget:

- Persona/relationship projection;
- Worldbook selection;
- active Reminder projection;
- eligible Dream lookup;
- local Fast Recall evidence (keyword, exact anchor/date, owner alias, entity);
- other owner-safe request metadata that does not depend on Recall output.

In the implemented natural path, local candidate work and semantic query work
are scheduled concurrently; that alone does not prove wall-clock overlap or
remove synchronous event-loop work. Broader projection parallelism remains a
measured target, not a completed performance claim. Ordinary owner chat is not
silently classified as no-memory merely because local keywords return zero.

### Historical feature milestones (not current routing or release state)

The feature lineage contains these older implementation milestones, superseded
by section 9 wherever their fixed-trigger routing conflicts:

- Memory authority, revision/ring commit coordination, migration audit/apply,
  unified Auto/Review commits, legacy writer convergence, and outbox-derived
  projections (`8f65b3b` through `f5e7360`).
- Local `skip / fast / deep` routing with stable reason codes; normal prepare
  does not call the remote Domain Sentinel (`0fffa13`).
- Exact-key Query Planner, semantic-query, rerank, and query-vector caches with
  singleflight.  Cache keys include the relevant model/prompt/index identity;
  final LLM replies are never cached (`0fffa13`, `352f36b`).
- Authority mode no longer rebuilds the full Moment graph in a chat request.
  The outbox projection worker owns incremental derived-index updates
  (`0fffa13`).
- Owner-confirmed aliases may force only their explicitly linked committed
  Memory IDs into Fast Recall.  Auto aliases remain non-authoritative and do
  not receive that privilege (`6bac6f4`).
- Composer receives one canonical request-local `ombre.memory_recall`
  projection.  When that source is present, the three legacy direct/targeted/
  diffused blocks are suppressed to prevent duplicate injection (`0fffa13`).
- Fast/tone routes keep Dream cue checks local.  Deep Recall and Dream share
  one exact query vector so Dream cannot trigger a second identical embedding
  request (`352f36b`).

These commits alone prove neither current live activation nor acceptance.
Use coordination release records for migration, frontend, image and deployment
state instead of treating an old M5 checklist as the current backlog. Natural
quality, latency, archive-state reconciliation and owner acceptance remain open.

The current natural selector forms ordinary candidates first, then may call
the Planner when evidence is unclear; it does not speculatively run a routing
model for every message. Supplemental queries join the same bounded pool
before one final rerank/admission stage. This differs from the older proposed
embedding/Planner race. Same-key misses use singleflight where implemented.

These stages remain dependency-ordered and are not speculatively duplicated:

- final candidate admission waits for the candidate union;
- Prompt compilation waits for the final Memory projection;
- the main talk-model call waits for the compiled context;
- tool execution waits for the model's tool call;
- a tool continuation reuses the parent prepare snapshot.

Moment/WordMap/Entity/Embedding index refresh is outbox/background work.  A
normal chat request must not rebuild an index or refresh a graph on its critical
path.  It uses the last verified revision, reports staleness, and falls back to
bounded direct evidence where allowed.

When the request deadline expires, unfinished optional sources return explicit
degraded metadata and are cancelled.  Cancellation propagates to provider HTTP
requests; late results cannot mutate the completed request snapshot or trigger
a second model answer.

Inspector renders both the dependency DAG and its wall-clock critical path:

- stage start/end/duration and parent dependencies;
- parallel overlap;
- cache/snapshot/single-flight status;
- cancelled/degraded stages;
- time to compiled context;
- model header/first byte/stream total.

Acceptance requires lower same-query wall-clock prepare and first-visible-token
time for ordinary, Fast, Deep, and tool-continuation fixtures.  Removing work
without reducing the measured critical path is not counted as a latency win.

## 14. Migration acceptance

Dry-run must preserve the semantic active-memory set, existing Bucket IDs and
bodies, candidate counts/statuses, and source references. It must produce no
model calls, embeddings, new canonical memories, Bucket writes, or regenerated
text. Every inconsistency is reported with an explicit reconcile action.

The audit also inventories the existing `IdentitySemanticStore`.  Apply mode
requires an explicit alias count and imports only aliases backed by evidence
Buckets that resolve to imported active Memories.  They enter the unified
authority as `trusted_source`; orphan evidence blocks migration.  The old
IdentitySemantic DB remains read-only backup evidence and cannot continue as a
second live alias authority after cutover.

Historical accepted candidates are reconciled without rewriting source data:

- A missing active Bucket is a documented historical deletion only when an
  exact-ID tombstone exists and its `deleted_at` is after candidate confirmation.
  The candidate remains committed history; no active Memory is recreated.
- A candidate/body hash mismatch is a documented legacy write projection only
  when the Bucket points to that same candidate, was updated after confirmation,
  and its current body is an exact suffix of the candidate with at most 128
  wrapper characters removed.  The current Bucket body is authoritative.
- Any other missing Bucket or divergent body remains a migration blocker.
  The audit reports both raw historical counts and unresolved counts; a
  reconciled history is not silently erased from the report.

The experience release is not complete until the owner has exercised Auto,
Review, alias confirmation, body revision, ring append, exact Fast Recall, and
ambiguous Deep Recall; then the exact immutable artifacts may advance
`production/current` with documentation and rollback evidence.

## 15. External design references (patterns only)

RECALL-R1 borrows patterns, not new runtime authorities or mandatory
dependencies:

- Graphiti: immutable episodes, provenance, valid/invalid temporal facts, and
  incremental watermarks.
- Letta/MemFS: compact in-context blocks versus discoverable archival memory,
  Markdown projections, and append-safe concurrent memory operations.
- LangGraph: thread state versus cross-thread long-term namespaces, explicit
  persistence migrations, and store-level semantic search.
- GPTCache: L1 exact plus optional L2 semantic cache layering.  Ombre applies
  exact revision-aware caching only to internal structured stages; it does not
  semantic-cache final companion replies.
