"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

const dashboard = fs.readFileSync(path.join(__dirname, "..", "dashboard.html"), "utf8");
const start = dashboard.indexOf("function configSavePresentation(result) {");
const marker = "\n}\n\nasync function saveConfig";
const close = dashboard.indexOf(marker, start);
assert.ok(start >= 0 && close > start, "Dashboard save presentation function must exist");
const source = dashboard.slice(start, close + 2);
const present = vm.runInNewContext(`${source}\nconfigSavePresentation`, {});

test("warning and partial Gateway receipts retain input without refreshing", () => {
  for (const reason of ["gateway_timeout", "incomplete_acknowledgement", "embedding_runtime_not_confirmed"]) {
    const result = present({
      ok: true,
      local_update: "applied",
      persistence: {runtime_yaml: "saved", credentials: "saved"},
      gateway_activation: {state: "unconfirmed", reason},
    });
    assert.equal(result.tone, "warning");
    assert.equal(result.refresh, false);
    assert.match(result.text, /输入已保留/);
  }
});

test("legacy responses without activation receipts cannot display success or refresh", () => {
  const result = present({ok: true, local_update: "applied", persistence: {runtime_yaml: "saved"}});
  assert.equal(result.tone, "warning");
  assert.equal(result.refresh, false);
  assert.match(result.text, /不能确认已生效/);
});

test("failed persistence is negative and does not refresh", () => {
  const result = present({
    ok: false,
    error: "credential_persist_failed",
    local_update: "applied",
    persistence: {runtime_yaml: "saved", credentials: "failed"},
    gateway_activation: {state: "unconfirmed", reason: "persistence_failed"},
  });
  assert.equal(result.tone, "negative");
  assert.equal(result.refresh, false);
});

test("confirmed save is positive and refreshes", () => {
  const result = present({
    ok: true,
    local_update: "applied",
    persistence: {runtime_yaml: "saved", credentials: "saved"},
    gateway_activation: {state: "confirmed", provider_verified: false},
    attention_required: false,
  });
  assert.equal(result.tone, "positive");
  assert.equal(result.refresh, true);
  assert.match(result.text, /Gateway 已确认本次配置/);
  assert.match(result.text, /未执行供应商鉴权或模型测试/);
});

test("confirmed runtime-only credential apply keeps the input for a later save", () => {
  const result = present({
    ok: true, local_update: "applied", updated: ["reranker.api_key"],
    persistence: {runtime_yaml: "not_requested", credentials: "not_requested"},
    gateway_activation: {state: "confirmed", provider_verified: false},
    attention_required: false,
  });
  assert.equal(result.tone, "positive");
  assert.equal(result.refresh, false);
  assert.match(result.text, /仅在运行时应用/);
});
