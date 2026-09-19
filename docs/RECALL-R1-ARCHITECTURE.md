# RECALL-R1 authoritative architecture

Status: design-frozen for M1 implementation.

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
rebuildable derived indexes, and an explainable Fast/Deep recall route.

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

## 9. Fast/Deep Recall

Fast is local and evidence-driven: exact anchors, dates, canonical identities,
owner-confirmed aliases, verified source records, and strong entity edges.

Deep runs only when the current query explicitly or strongly implies past
context and Fast evidence is insufficient. The Query Planner receives the
current query, up to three recent items, and an owner-safe Fast evidence
summary. It returns structured supplemental queries/must-terms only. Original
and supplemental candidates are merged before one semantic/graph retrieval and
at most one provider rerank. Deterministic admission owns the final decision.

Planner failure degrades to the original query within the same bounded recall
budget. It never fails the chat turn or recursively invokes itself.

The current R5 optimization that skips a guaranteed-empty Moment rerank is not
Recall-quality closure.  RECALL-R1 is not accepted until real owner queries
prove that strong candidates survive admission and every rejected candidate
has a stable reason code.  A trace that still only shows `N -> 0` without
rejection reasons fails acceptance.

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
   reranker provider/model revision + scoring/prompt revision.
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

- routing decision (`skip`, `fast`, `deep`);
- prepare/snapshot time;
- embedding/planner/rerank request count and time;
- cache hit/miss by layer;
- candidates found/admitted/rejected and reason codes;
- Memory block actually injected;
- model first-byte and total time.

Expected direction, not an invented fixed SLA:

- ordinary non-memory chat skips remote Recall work;
- exact identity/date/anchor questions use local Fast Recall;
- ambiguous past-context questions pay one bounded Planner/Deep path;
- a tool continuation reuses the parent prepare snapshot;
- no query performs repeated provider embedding/rerank for the same revisioned
  evidence within one logical turn.

Upstream model or browse latency remains visible and separate; RECALL-R1 must
not claim to fix provider latency it does not control.

## 14. Migration acceptance

Dry-run must preserve the semantic active-memory set, existing Bucket IDs and
bodies, candidate counts/statuses, and source references. It must produce no
model calls, embeddings, new canonical memories, Bucket writes, or regenerated
text. Every inconsistency is reported with an explicit reconcile action.

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
