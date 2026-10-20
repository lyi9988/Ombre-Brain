"use strict";

// Local real-DOM regression. Supply Playwright via the existing runtime/NODE_PATH.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const {chromium} = require("playwright");

const html = fs.readFileSync(path.join(__dirname, "..", "dashboard.html"), "utf8");
function section(start, end) {
  const first = html.indexOf(start), last = html.indexOf(end, first);
  assert.ok(first >= 0 && last > first, start);
  return html.slice(first, last);
}
const source = [
  section("let configSnapshot = null;", "let todoStatusFilter"),
  section("async function loadConfig(explicitMemoryConflictReload) {", "function reloadMemoryConfigurationAfterConflict"),
  section("function setConfigStatus(html) {", "async function reviewCredentialConflict"),
  section("async function saveConfig(persist, persistEnv) {", "checkAuth().then"),
].join("\n");

test("context settings preserve zero, reject invalid drafts, and read back switches in the real DOM", async t => {
  const browser = await chromium.launch({channel: "msedge", headless: true});
  t.after(() => browser.close());
  const page = await browser.newPage();
  const cfg = {
    dehydration: {}, embedding: {}, reranker: {}, persona: {}, dream: {}, recall: {},
    reflection: {}, portrait: {}, self_anchor: {}, memory_diffusion: {},
    memory_configuration_revision: "mc-" + "a".repeat(64),
    gateway: {recent_context_budget: 300, current_inner_state_interval_rounds: 15},
  };
  const posts = [];
  let gets = 0, failure = null;
  await page.route("http://dashboard.test/**", async route => {
    const request = route.request();
    assert.equal(new URL(request.url()).pathname, "/api/config", "no provider or other API calls");
    if (request.method() === "GET") {
      gets++;
      return route.fulfill({json: cfg});
    }
    const body = request.postDataJSON();
    posts.push(body);
    if (failure === "network") return route.abort();
    if (failure === "server") return route.fulfill({status: 500, json: {
      ok: false, error: "runtime_persist_failed", persistence: {runtime_yaml: "failed"},
    }});
    Object.assign(cfg.gateway, body.gateway);
    return route.fulfill({json: {ok: true, local_update: "applied",
      persistence: {runtime_yaml: body.persist ? "saved" : "not_requested", credentials: "not_requested"},
      gateway_activation: {state: "confirmed"},
    }});
  });
  // Keep the production markup and handlers; omit unrelated dashboard startup.
  await page.setContent(html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ""));
  await page.addScriptTag({content: `
    const BASE = 'http://dashboard.test';
    const authFetch = (url, options) => fetch(url, options);
    const esc = value => String(value ?? '');
    const normalizeGatewayUpstream = value => value;
    const normalizeGatewayModelEntry = value => value;
    function renderGatewayUpstreams() {}
    ${source}
  `});
  await page.evaluate(async () => {
    document.querySelectorAll('.tab.active').forEach(el => el.classList.remove('active'));
    document.querySelector('.tab[data-tab="memory-config"]').classList.add('active');
    await loadConfig();
  });
  const ids = ["cfg-recent-context-budget", "cfg-persona-context-rounds"];
  const switches = ["cfg-recent-context-enabled", "cfg-persona-context-enabled"];
  const fields = ["recent_context_budget", "current_inner_state_interval_rounds"];
  const state = () => page.evaluate(({ids, switches}) => ({
    values: ids.map(id => document.getElementById(id).value),
    switches: switches.map(id => document.getElementById(id).value),
  }), {ids, switches});
  async function edit(values, enabled = ["true", "true"]) {
    await page.evaluate(({ids, switches, values, enabled}) => {
      ids.forEach((id, i) => { document.getElementById(id).value = values[i]; });
      switches.forEach((id, i) => { document.getElementById(id).value = enabled[i]; });
      setConfigStatus('');
    }, {ids, switches, values, enabled});
  }
  async function clickSave(persist = true) {
    await page.evaluate(persist => document.querySelector(
      '#memory-config-view button[onclick="saveConfig(' + persist + ')"]').click(), persist);
  }
  assert.deepEqual(await state(), {values: ["300", "15"], switches: ["true", "true"]});
  for (const [values, enabled, expected] of [
    [["0", "0"], ["true", "true"], [0, 0]],
    [["1234", "27"], ["true", "true"], [1234, 27]],
    [["5000", "201"], ["true", "true"], [5000, 201]],
    [["321", "18"], ["false", "false"], [0, 0]],
    [["0", "19"], ["true", "true"], [0, 19]],
    [["400", "0"], ["true", "true"], [400, 0]],
  ]) {
    for (const persist of [false, true]) {
      const before = gets;
      await edit(values, enabled);
      const readback = page.waitForResponse(response => response.url().endsWith('/api/config')
        && response.request().method() === 'GET');
      await clickSave(persist);
      await readback;
      await page.waitForFunction(({ids, switches, expected}) => ids.every((id, i) =>
        document.getElementById(id).value === String(expected[i])
        && document.getElementById(switches[i]).value === (expected[i] > 0 ? 'true' : 'false')),
      {ids, switches, expected});
      await page.waitForFunction(() => document.getElementById('memory-config-status').textContent.includes('Gateway 已确认'));
      assert.ok(gets > before, "successful save must read back the mock server");
      fields.forEach((field, i) => assert.equal(posts.at(-1).gateway[field], expected[i]));
      assert.equal(posts.at(-1).persist, persist);
      assert.deepEqual(await state(), {
        values: expected.map(String), switches: expected.map(value => value > 0 ? "true" : "false"),
      });
    }
  }
  for (const enabled of [["true", "true"], ["false", "false"]]) {
    for (let i = 0; i < ids.length; i++) {
      for (const invalid of ["-1", "1.5", "1e2", "", "not-a-number", "9007199254740992"]) {
        const values = ["811", "29"];
        values[i] = invalid;
        await edit(values, enabled);
        const draft = await state(), before = posts.length;
        await clickSave();
        await page.waitForFunction(() => document.getElementById('memory-config-status').textContent.includes('须为非负整数'));
        assert.equal(posts.length, before, `${ids[i]} invalid while ${enabled[i]} must not submit`);
        assert.deepEqual(await state(), draft);
      }
    }
  }
  for (const kind of ["server", "network"]) {
    failure = kind;
    await edit(["0", "38"], ["true", "false"]);
    const draft = await state(), before = gets;
    await clickSave();
    await page.waitForFunction(() => /落盘失败|请求失败/.test(document.getElementById('memory-config-status').textContent));
    assert.equal(gets, before, "failure must not reload over the draft");
    assert.deepEqual(await state(), draft);
    await page.evaluate(() => loadConfig());
    assert.deepEqual(await state(), draft, "later GET preserves each edited value and switch");
  }
  console.log("Verified 12 successful save/readback cases, 24 invalid cases, 2 failed-save draft cases; mock API only.");
});
