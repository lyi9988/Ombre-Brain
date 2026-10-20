# Bounded long-chat ingestion v1

This slice applies inside an oversized ordinary daily-chat window, using the
existing daily window planner unchanged. It requires the existing Authority
path. It does not segment attachments, change L0 storage or any recall/L2 code.

* Sources retain exact raw text, original source key, snapshot hash and revision.
  A model view carries half-open Python Unicode codepoint offsets `[start,end)`
  in that exact text (emoji count as one codepoint, no Unicode normalization).
  Original `source:N` labels retain their original window position.
* Core ranges partition each source without gaps or overlap. Each view includes
  at most `min(128, budget//8)` characters of same-source context on each side;
  immediately adjacent sources may contribute that many characters each.
  Core width is `budget - 4*context`; total source text per request is <= the
  existing configured character budget. This is a source-text bound, not a
  provider token limit. Existing output tokens/routes/prompts remain unchanged.
* A candidate must cite the core source and overlap its core range if it supplies
  scene ranges. Context can support the same candidate but cannot independently
  create a candidate in that segment. The original source-narrowing union applies.
  Program anchors never exceed the observed ranges; model offsets are absolute,
  not slice-relative. Source text is stored whole by the existing L0 mechanism.
* Task identity includes full source versions/hashes, ordered plan, prompt,
  catalogue, route and extraction configuration. Changing source content/version
  creates a fresh task; no old cursor is reused. Changed generation settings
  cannot bypass an unresolved dispatch for the same source window.
  Auto/Review, confidence gates and scheduling quotas do not discard existing
  receipts; existing candidate decisions are retained when these controls change.
* Authority holds CAS progress and a dispatched intent before provider IO.
  The received response is saved before parsing/candidate preparation. Restart
  reuses received responses and prepared candidates; never repeats unknown IO.
  An empty valid candidate array is a completed segment. Invalid known output
  is an explicit blocked receipt, not silently consumed coverage.
* Each segment has one existing candidate-preparation receipt, even with zero
  candidates. Cursor advances only after this receipt exists. Candidate budgets
  drain retained proposals without another model call. Event watermark advances
  only after all segments of the original window can be handed to the existing
  writer. Owner edits/accept/reject/defer remain in the same Authority.

Targeted examples before implementation: Unicode boundary round trip; a middle
fragment citing `source:2` retains that number and a whole-source hash; 3 runs
at one call/run cover every core once; restart after response-before-prepare
makes zero extra calls; timeout blocks even with `force`; source edit creates a
different task; simultaneous invocations lose CAS before a second dispatch;
candidate quota drains without swallowing the tail; Review/Auto retain their
existing writer and owner decision semantics.

Semantic equivalence between independently generated candidates is still handled
by the existing deduplicator. Execution receipts guarantee no replay of completed
segments, not a mathematical guarantee that a model never repeats a fact. A
single bounded context cannot prove every arbitrarily distant dependency was
understood; full semantic quality and live provider capacity remain unverified.
