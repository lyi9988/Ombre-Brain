# Gu Yan Ombre Fork — Authoritative AI Handoff

> Source of truth for AI/code agents working on Gu Yan's Ombre backend.
> GitHub source visibility is not production deployment.

## 1. Why this fork exists

`https://github.com/lyi9988/Ombre-Brain` is the long-term sovereign development repository for Gu Yan (`jiajia-main`). It was not created merely to bypass one push permission error or store one Canonical patch.

This fork is intended to own:

- the single authoritative `jiajia-main` Self Model;
- migration from existing Persona state without deleting history;
- Identity Core, Evolving Self, Self State, and Self Narrative;
- the single persona revision and commit path;
- Canonical continuation across Operit and Reality;
- Gu Yan-specific Gateway orchestration, projection, audit, and observable configuration.

`aizizhu` is the action/execution layer. It may submit Evidence or Proposals, but must not become a second persona authority. Upstream repositories are update sources, not Gu Yan's authority.

## 2. Four different meanings of current

Do not collapse these into one "latest" label.

| Meaning | Location | Ref/SHA at audit | Use |
|---|---|---|---|
| Running Ombre production | `hulala:/srv/ombre-brain/Ombre-Brain-Fork` | `main@1dac4385eff4fff6f28c2917664ae90d34fef5db` | Runtime and migration baseline; running old code |
| Yinglianchun upstream main | `Yinglianchun/Ombre-Brain` | `c758a4d3df96e8224fee325565157b7fb2ad69b1` | Read-only comparison source |
| Personal fork main | `lyi9988/Ombre-Brain:main` | `7b74699df6e908ecab65e3eb1fd7c7ce56ad2b8f` | Newer refactored line; not production and not automatically Gu Yan authority |
| Validated continuation snapshot | `lyi9988/Ombre-Brain:feature/canonical-continuation` | rooted at `dc2e840c2790e0679ed5bd9cd4a56e77ac6ec68b` before this document | Review/port source; not deployed |

The production SHA and personal fork main use different current architectures. Do not blindly merge the snapshot into fork main. Port deliberately with tests and production-state compatibility review.

## 3. Authoritative GitHub handoff entry

For every new Gu Yan Ombre task, fetch GitHub and begin here:

```text
repository: https://github.com/lyi9988/Ombre-Brain
handoff branch: handoff/guyan-ombre-current
this file: docs/AI_GUYAN_FORK_HANDOFF.md
```

The handoff branch is the navigation/governance entry. Documentation commits may make its HEAD newer than the implementation snapshot. Do not infer deployment from its HEAD.

Current Canonical implementation reference:

```text
branch: feature/canonical-continuation
validated source commits:
  134d84f656d44c7a9833559aae741f955c8c1ad7
  7599b6b0f2158c9358ea7d428926fb5ac96d4cdf
GitHub-compatible snapshot root:
  dc2e840c2790e0679ed5bd9cd4a56e77ac6ec68b
key files:
  canonical_continuation.py
  gateway.py
  config.example.yaml
  tests/test_canonical_continuation.py
  tests/test_gateway_canonical_integration.py
```

## 4. What exists where

### GitHub shows

- Canonical Continuation source and tests;
- this governance/handoff document;
- reviewable history and diffs;
- no production secrets or runtime databases.

### SSH on hulala shows

- running production source: `/srv/ombre-brain/Ombre-Brain-Fork`;
- production runtime state and containers;
- isolated worktree: `/tmp/ombre-canonical-adapter`;
- verified backup under `/srv/backups/ombre-brain/`.

`/tmp` is temporary and not the durable handoff authority. GitHub is the durable code handoff. `/srv` remains runtime authority until controlled deployment.

### Not true yet

- Canonical Continuation is not deployed to the running Gateway;
- production source has not advanced from `1dac438...`;
- the full Self Model has not been implemented;
- `aizizhu` is not yet fully converted into a Self Model consumer/proposer;
- a GitHub push does not change containers or services.

## 5. Already completed

- no-generation Canonical Bridge exists in `lyi9988/aizizhu`;
- Ombre adapter implements initial cursor baseline, incremental pull, real `user/assistant` insertion, reliable Outbox, and post-success cursor commit;
- isolated real HTTP E2E passed without an added continuity system prompt;
- core adapter tests passed (`5 passed`);
- original integration tests and HTTP E2E passed in the dependency-complete isolation environment;
- `instruction` defaults empty and `instruction_injected=false` is observable.

## 6. Remaining backend work

### Deploy current continuation patch

1. Generate a dedicated `CANONICAL_BRIDGE_TOKEN`; never reuse a GitHub PAT.
2. Back up and verify current production state.
3. Deploy patch A to `aizizhu` first; verify `/bridge/v1/` auth and no-generation ingest.
4. Deploy patch B to Ombre second; verify Gateway health and observability.
5. Test Reality -> Operit continuation without reminder words.
6. Test Operit -> Reality continuation.
7. Verify cursor, Outbox retry, idempotency, role sequence, and `instruction_injected=false`.
8. Roll back on duplicate generation, role corruption, cursor loss, or hidden prompt injection.

### Build the long-term Gu Yan line

1. Establish a deliberate Gu Yan integration branch from a reviewed architecture baseline; never choose by commit date alone.
2. Freeze dangerous direct long-term Persona writes while preserving reads/history.
3. Implement Self Model modules and migration.
4. Keep one database, revision sequence, commit authority, Identity Core, and Self Narrative.
5. Convert `aizizhu` to projection consumer and Evidence/Proposal producer.
6. Add audit, rollback, explicit prompt configuration, and reason-coded capability failures.

## 7. Hard prohibitions

- Do not treat personal fork `main` as production because it is newer.
- Do not reset the Gu Yan fork to P0luz or Yinglianchun main.
- Do not edit `/srv/ombre-brain/Ombre-Brain-Fork` during frontend decoration.
- Do not build Reality frontend in this repository.
- Do not create a second persona engine in `aizizhu`.
- Do not add hidden continuity prompts, silent filters, or unobservable restrictions.
- Do not delete or overwrite existing Persona databases during migration.
- Do not claim deployment from a GitHub push.

## 8. Reality frontend pointer

Reality frontend work belongs here, not in Ombre:

```text
repository: https://github.com/lyi9988/aizizhu
branch: handoff/reality-decoration-current
handoff: docs/AI_FRONTEND_HANDOFF.md
```

Frontend agents should treat Ombre as a backend contract/reference unless explicitly assigned backend work.
