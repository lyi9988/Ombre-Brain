# RECALL-R1 authoritative architecture

Status: design-frozen for M1 implementation.

## 1. Purpose

RECALL-R1 converges the existing Ombre memory writers and recall paths without
removing the current Bucket Markdown/YAML format or changing unrelated tool,
Dream, Darkroom, Reminder, Emotion, Needs, Goal, or canonical-conversation
authorities.

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

## 10. Model routing and prompts

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

## 11. Owner UI contract

Reality exposes separate, collapsible sections under `更多 -> 中枢`:

- Memory: pending, remembered, people/aliases, recall diagnostics.
- Darkroom: private locked rooms and revisions.
- Dreams: latent/surfaced dream records.
- Goals and reminders: action state, not memory.
- Model routing: route-first model selection, with provider/model administration in secondary tabs.

Repeated items, revisions, rings, raw Markdown/YAML, and advanced index details
are collapsed by default. Mobile layouts show one primary editor at a time and
must remain usable at 320/390/430px.

## 12. Migration acceptance

Dry-run must preserve the semantic active-memory set, existing Bucket IDs and
bodies, candidate counts/statuses, and source references. It must produce no
model calls, embeddings, new canonical memories, Bucket writes, or regenerated
text. Every inconsistency is reported with an explicit reconcile action.

The experience release is not complete until the owner has exercised Auto,
Review, alias confirmation, body revision, ring append, exact Fast Recall, and
ambiguous Deep Recall; then the exact immutable artifacts may advance
`production/current` with documentation and rollback evidence.
