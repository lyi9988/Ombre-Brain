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

test("legacy memory settings load, preserve drafts, and submit unclamped number limits", async () => {
  const elementMap = new Map();
  const elements = new Proxy({}, {get: (_target, id) => {
    if (!elementMap.has(id)) elementMap.set(id, {value: "1", innerHTML: "", textContent: "", style: {}});
    return elementMap.get(id);
  }});
  const cfg = {
    dehydration: {}, embedding: {}, reranker: {}, persona: {}, dream: {}, recall: {},
    reflection: {}, portrait: {}, self_anchor: {},
    memory_configuration_revision: "mc-" + "a".repeat(64),
    gateway: {
      inject_max_cards: 6, diffusion_inject_max_items: 5, inject_total_budget: 0,
      dynamic_top_k: 8, semantic_candidate_top_k: 20, moment_search_limit: 14,
      memory_detail_recall_max_ids: 9, memory_detail_recall_budget: 1000000,
    },
    memory_diffusion: {},
  };
  const calls = [];
  let nextStatus = 200;
  let confirmResult = true;
  const context = {
    console, Number, String, JSON, Math, Object, Array, Promise,
    BASE: "",
    document: {
      getElementById: id => elements[id],
      querySelector: selector => selector === ".tab.active" ? {dataset: {tab: "memory-config"}} : null,
    },
    confirm: () => confirmResult,
    authFetch: async (_url, options) => {
      if (!options) return {json: async () => structuredClone(cfg)};
      calls.push(JSON.parse(options.body));
      const status = nextStatus;
      nextStatus = 200;
      return {status, json: async () => status === 409 ? {error: {code: "memory_configuration_revision_conflict"}} : {ok: true}};
    },
    normalizeGatewayUpstream: value => value,
    normalizeGatewayModelEntry: value => typeof value === "string" ? {id: value} : value,
    renderGatewayUpstreams() {},
    esc: value => String(value ?? ""),
    setConfigStatus: value => { elements["memory-config-status"].innerHTML = value; },
    configSavePresentation: () => ({text: "saved", tone: "positive", refresh: false}),
  };
  const loadStart = dashboard.indexOf("async function loadConfig(explicitMemoryConflictReload) {");
  const loadEnd = dashboard.indexOf("\n}\n\nfunction reloadMemoryConfigurationAfterConflict", loadStart) + 2;
  const reloadStart = dashboard.indexOf("function reloadMemoryConfigurationAfterConflict() {");
  const reloadEnd = dashboard.indexOf("\n}\n\nfunction setConfigStatus", reloadStart) + 2;
  const saveStart = dashboard.indexOf("async function saveConfig(persist, persistEnv) {");
  const saveEnd = dashboard.indexOf("\n}\n\ncheckAuth().then", saveStart) + 2;
  assert.ok(loadStart >= 0 && loadEnd > loadStart && reloadStart >= 0 && reloadEnd > reloadStart && saveStart >= 0 && saveEnd > saveStart);
  const declarations = `let configSnapshot = null; let memoryConfigDraftSnapshot = null;
    let memoryConfigurationRevision = null; let memoryConfigConflict = false;
    const memoryConfigFormIds = ${JSON.stringify([
      "cfg-recent-context-enabled", "cfg-gateway-cooldown", "cfg-gateway-rounds",
      "cfg-recent-context-cooldown", "cfg-recent-context-reentry", "cfg-recent-context-budget",
      "cfg-recalled-memory-budget", "cfg-related-memory-budget", "cfg-inject-max-cards",
      "cfg-diffusion-inject-max-items", "cfg-inject-total-budget", "cfg-dynamic-top-k",
      "cfg-semantic-candidate-top-k", "cfg-moment-search-limit", "cfg-memory-detail-recall-enabled",
      "cfg-memory-detail-recall-max-ids", "cfg-memory-detail-recall-budget", "cfg-persona-context-enabled",
      "cfg-persona-context-rounds", "cfg-direct-render-mode", "cfg-retrieval-mode",
      "cfg-operit-context-rewrite-enabled", "cfg-word-map-hint-enabled", "cfg-query-planner-enabled",
      "cfg-query-resurface-enabled", "cfg-diffusion-enabled", "cfg-diffusion-topk", "cfg-diffusion-min",
      "cfg-chain-walk", "cfg-chain-hops", "cfg-chain-confidence", "cfg-chain-frontier",
    ])}; let gatewayUpstreams = [];
    function setConfigStatus(html) { document.getElementById("memory-config-status").innerHTML = html; }`;
  vm.runInNewContext(`${declarations}\n${dashboard.slice(loadStart, loadEnd)}\n${dashboard.slice(reloadStart, reloadEnd)}\n${dashboard.slice(saveStart, saveEnd)}`, context);
  await vm.runInNewContext("loadConfig()", context);
  assert.equal(vm.runInNewContext("memoryConfigurationRevision", context), "mc-" + "a".repeat(64));
  assert.equal(elements["cfg-inject-max-cards"].value, 6);
  assert.equal(elements["cfg-inject-total-budget"].value, 0);
  assert.equal(elements["cfg-memory-detail-recall-max-ids"].value, 9);
  assert.equal(elements["cfg-memory-detail-recall-budget"].value, 1000000);

  elements["cfg-inject-max-cards"].value = "12";
  cfg.gateway.inject_max_cards = 4;
  await vm.runInNewContext("loadConfig()", context);
  assert.equal(elements["cfg-inject-max-cards"].value, "12", "a server refresh must retain a dirty draft");

  elements["cfg-inject-max-cards"].value = "7";
  elements["cfg-diffusion-inject-max-items"].value = "8";
  elements["cfg-inject-total-budget"].value = "1000000";
  elements["cfg-dynamic-top-k"].value = "12";
  elements["cfg-semantic-candidate-top-k"].value = "15";
  elements["cfg-moment-search-limit"].value = "11";
  elements["cfg-memory-detail-recall-max-ids"].value = "10";
  elements["cfg-memory-detail-recall-budget"].value = "1000000";
  await vm.runInNewContext("saveConfig(true)", context);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].gateway.inject_max_cards, 7);
  assert.equal(calls[0].gateway.diffusion_inject_max_items, 8);
  assert.equal(calls[0].gateway.inject_total_budget, 1000000);
  assert.equal(calls[0].gateway.memory_detail_recall_max_ids, 10);
  assert.equal(calls[0].gateway.memory_detail_recall_budget, 1000000);
  assert.equal(calls[0].gateway.dynamic_top_k, 12);
  assert.equal(calls[0].gateway.semantic_candidate_top_k, 15);
  assert.equal(calls[0].gateway.moment_search_limit, 11);
  assert.equal(calls[0].expected_memory_revision, "mc-" + "a".repeat(64));
  assert.equal("api_key" in calls[0].gateway, false);

  for (const id of ["cfg-inject-max-cards", "cfg-diffusion-inject-max-items", "cfg-inject-total-budget",
    "cfg-memory-detail-recall-max-ids", "cfg-memory-detail-recall-budget"]) elements[id].value = "0";
  await vm.runInNewContext("saveConfig(false)", context);
  assert.equal(calls.length, 2);
  assert.equal(calls[1].gateway.inject_max_cards, 0);
  assert.equal(calls[1].gateway.diffusion_inject_max_items, 0);
  assert.equal(calls[1].gateway.inject_total_budget, 0);
  assert.equal(calls[1].gateway.memory_detail_recall_max_ids, 0);
  assert.equal(calls[1].gateway.memory_detail_recall_budget, 0);

  elements["cfg-inject-max-cards"].value = "201";
  await vm.runInNewContext("saveConfig(false)", context);
  assert.equal(calls.length, 2, "out-of-range numbers must not be sent to the API");
  assert.equal(elements["cfg-inject-max-cards"].value, "201", "invalid input remains visible for correction");
  elements["cfg-inject-max-cards"].value = "5";
  elements["cfg-dynamic-top-k"].value = "20";
  elements["cfg-semantic-candidate-top-k"].value = "5";
  await vm.runInNewContext("saveConfig(false)", context);
  assert.equal(calls.length, 3, "independent candidate channels may use different sizes");
  assert.equal(calls[2].gateway.dynamic_top_k, 20);
  assert.equal(calls[2].gateway.semantic_candidate_top_k, 5);

  elements["cfg-inject-max-cards"].value = "17";
  nextStatus = 409;
  await vm.runInNewContext("saveConfig(false)", context);
  assert.equal(calls.length, 4);
  assert.equal(calls[3].expected_memory_revision, "mc-" + "a".repeat(64));
  assert.equal(elements["cfg-inject-max-cards"].value, "17", "409 must retain the edited value");
  assert.equal(vm.runInNewContext("memoryConfigConflict", context), true);
  cfg.memory_configuration_revision = "mc-" + "b".repeat(64);
  cfg.gateway.inject_max_cards = 6;
  await vm.runInNewContext("loadConfig()", context);
  assert.equal(vm.runInNewContext("memoryConfigurationRevision", context), "mc-" + "a".repeat(64), "ordinary reload must not adopt the conflicting revision");
  assert.equal(elements["cfg-inject-max-cards"].value, "17", "ordinary reload must not discard the draft");
  await vm.runInNewContext("saveConfig(false)", context);
  assert.equal(calls.length, 4, "saving is paused until the owner explicitly reloads the server version");

  await vm.runInNewContext("reloadMemoryConfigurationAfterConflict()", context);
  assert.equal(vm.runInNewContext("memoryConfigurationRevision", context), "mc-" + "b".repeat(64));
  assert.equal(elements["cfg-inject-max-cards"].value, 6, "confirmed rebase explicitly replaces the old draft");
  assert.equal(vm.runInNewContext("memoryConfigConflict", context), false);
  await vm.runInNewContext("saveConfig(false)", context);
  assert.equal(calls.length, 5, elements["memory-config-status"].innerHTML);
  assert.equal(calls[4].expected_memory_revision, "mc-" + "b".repeat(64));
  delete cfg.memory_configuration_revision;
  await vm.runInNewContext("loadConfig()", context);
  await vm.runInNewContext("saveConfig(false)", context);
  assert.equal(calls.length, 5, "saving without a configuration revision must be blocked");
  assert.match(elements["memory-config-status"].innerHTML, /没有有效版本号/);
});
