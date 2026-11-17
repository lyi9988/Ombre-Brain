"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const dashboard = fs.readFileSync(path.join(__dirname, "..", "dashboard.html"), "utf8");

function section(start, end) {
  const startIndex = dashboard.indexOf(start);
  const endIndex = dashboard.indexOf(end, startIndex);
  assert.ok(startIndex >= 0 && endIndex > startIndex, start);
  return dashboard.slice(startIndex, endIndex);
}

test("portrait candidates load every batch without replacing review inputs", () => {
  const area = "profile_fact_candidates";
  const candidates = Array.from({ length: 21 }, (_, i) => ({
    origin: "l1_v2", scope: "user", status: "candidate", candidate_id: `memory:${i}`,
    text: "同一句来源原话", display_text: `待审核句子 ${i}`,
  }));
  const appends = [];
  const list = {
    insertAdjacentHTML(position, html) {
      assert.equal(position, "beforeend");
      appends.push(html);
    },
    set innerHTML(_value) { throw new Error("existing owner edits were replaced"); },
  };
  const more = { dataset: { candidateArea: area }, previousElementSibling: { textContent: "" }, hidden: false };
  const context = vm.createContext({
    portraitState: { [area]: candidates, portrait: { user: { stable_revision: 6 } } }, more,
    document: { getElementById(id) {
      assert.equal(id, `portrait-${area}-rows`);
      return list;
    } },
    esc: String,
    renderPortraitRow: JSON.stringify,
  });
  vm.runInContext([
    section("const PORTRAIT_CANDIDATE_PAGE_SIZE", "let profileFacts"),
    section("function portraitRows", "function portraitStatusChip"),
    section("function renderPortraitRows", "function portraitRowDateLabel"),
  ].join("\n"), context);
  const first = vm.runInContext(`renderPortraitCandidates("${area}", "Candidates", portraitState.${area}, 6)`, context);
  assert.match(first, /已显示 8 \/ 21 条/);
  assert.match(first, /显示更多/);
  assert.equal((first.match(/"_approve":/g) || []).length, 8);
  assert.doesNotMatch(first, /"candidate_id":"memory:0"/);
  vm.runInContext("showMorePortraitCandidates(more)", context);
  assert.equal((appends[0].match(/"_approve":/g) || []).length, 8);
  assert.equal(more.previousElementSibling.textContent, "已显示 16 / 21 条");
  assert.equal(more.hidden, false);
  vm.runInContext("showMorePortraitCandidates(more)", context);
  assert.equal(more.previousElementSibling.textContent, "已显示 21 / 21 条");
  assert.equal(more.hidden, true);
  assert.match(appends[1], /"_delete":\{"area":"profile_fact_candidates","candidate_id":"memory:0"\}/);
  assert.match(appends[1], /"_approve":\{"id":"memory:0","revision":6\}/);
  const all = first + appends.join("");
  for (const row of candidates) {
    const approveIdentity = `"_approve":{"id":"${row.candidate_id}","revision":6}`;
    assert.equal(all.split(approveIdentity).length - 1, 1);
  }
  vm.runInContext("showMorePortraitCandidates(more)", context);
  assert.equal(appends.length, 2);
});
