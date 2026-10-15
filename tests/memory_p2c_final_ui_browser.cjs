// Real Edge DOM coverage for the P2-C source-task panel and config draft path.
// All API responses are synthetic; no model, provider, or production endpoint is called.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const asset = fs.readFileSync(path.join(root, 'dashboard_assets/chat-memory.js'), 'utf8');
const dashboard = fs.readFileSync(path.join(root, 'dashboard.html'), 'utf8');
const inlineScripts = [...dashboard.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)].map(m => m[1]);
assert.equal(inlineScripts.length, 1, 'dashboard keeps one inline application script');
const dashboardScript = inlineScripts[0]
  .replace('const BASE = location.origin + PATH_PREFIX;', 'const BASE = location.origin;')
  .replace(/checkAuth\(\)\.then\(function\(ok\)\s*\{\s*if \(ok\) loadBuckets\(\);\s*\}\);/, '')
  .replace(/loadDashboardModule\('\/dashboard-assets\/chat-memory\.js\?v=[^']*'\);/, '');
const dashboardMarkup = dashboard
  .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '')
  .replace(/<link\b[^>]*>/gi, '');
const inlineStyles = (dashboard.match(/<style[^>]*>[\s\S]*?<\/style>/gi) || []).join('\n');

function configFixture() {
  const bounds = {
    diary_l1_input_window_chars: { minimum: 256, maximum: 2000000 },
    diary_l1_max_windows: { minimum: 1, maximum: 10000 },
    diary_l1_scene_limit: { minimum: 1, maximum: 10000 },
    diary_l1_max_tokens: { minimum: 1, maximum: 100000 },
    diary_l1_scene_min_value: { minimum: 0, maximum: 1 },
    action_l1_jobs_per_run: { minimum: 0, maximum: 10000 },
    daily_chat_memory_candidate_max_tokens: { minimum: 1, maximum: 100000 },
    daily_chat_memory_max_windows_per_run: { minimum: 1, maximum: 10000 },
    daily_chat_memory_window_max_input_chars: { minimum: 256, maximum: 2000000 },
    daily_chat_memory_max_per_day: { minimum: 0, maximum: 10000 },
    daily_chat_memory_review_max_per_day: { minimum: 0, maximum: 10000 },
  };
  return {
    memory_configuration_revision: 'mc-' + 'a'.repeat(64),
    dehydration: { model: '', base_url: '', api_key_masked: '', max_tokens: 1024, temperature: 0.1 },
    embedding: { enabled: false, model: '', base_url: '', api_key_masked: '', has_own_api_key: false, effective_base_url: '' },
    reranker: { enabled: false, model: '', base_url: '', timeout_seconds: 12, candidate_limit: 20, score_weight: 0.65, api_key_masked: '', has_own_api_key: false, api_ready: false, effective_base_url: '' },
    persona: { enabled: false, event_recording_enabled: false, model: '', base_url: '', api_key_masked: '', api_ready: false },
    dream: { enabled: false, auto_enabled: false, surface_enabled: false, inject_enabled: false, retain_after_inject: false, model: '', base_url: '', api_key_masked: '', api_ready: false, daily_hour: 3, daily_probability: 0.4, min_material_count: 5, material_window_hours: 48, identity_anchor_id: '' },
    gateway: { upstreams: [], cooldown_hours: 6, skip_recent_rounds: 5, recent_context_cooldown_hours: 6, recent_context_reentry_idle_hours: 24, recent_context_budget: 300, recalled_memory_budget: 900, related_memory_budget: 220, inject_max_cards: 2, diffusion_inject_max_items: 2, inject_total_budget: 1800, dynamic_top_k: 10, semantic_candidate_top_k: 50, moment_search_limit: 50, memory_detail_recall_enabled: false, memory_detail_recall_max_ids: 3, memory_detail_recall_budget: 1200, current_inner_state_interval_rounds: 15, direct_render_mode: 'auto', retrieval_mode: 'graph', operit_context_rewrite_enabled: false, word_map_hint_enabled: false, query_planner_enabled: false, domain_sentinel_enabled: true, domain_sentinel_model: '', domain_sentinel_base_url: '', domain_sentinel_api_ready: false },
    recall: { query_resurface_enabled: false },
    memory_diffusion: { enabled: false, top_k: 5, min_score: 0.5, chain_walk_enabled: false, chain_hops: 1, chain_confidence: 0.5, chain_frontier: 10 },
    reflection: { enabled: true, auto_enabled: true, daily_enabled: true, daily_min_memory_items: 5, daily_conversation_turn_limit: 12, daily_chat_memory_mode: 'review', daily_chat_memory_turn_limit: 0, memory_affect_anchor_enabled: true, relationship_weather_affect_anchor_enabled: true, diary_l1_input_window_chars: 4096, diary_l1_max_windows: 10, diary_l1_scene_limit: 12, diary_l1_max_tokens: 2048, diary_l1_scene_min_value: 0.2, diary_l1_source_kind: 'mcp', action_l1_jobs_per_run: 5, daily_chat_memory_candidate_max_tokens: 2048, daily_chat_memory_max_windows_per_run: 10, daily_chat_memory_window_max_input_chars: 50000, daily_chat_memory_max_per_day: 5, daily_chat_memory_review_max_per_day: 10, model: '', thinking_mode: '', base_url: '', api_key_masked: '', api_ready: false, generation_parameter_bounds: bounds },
    portrait: { enabled: false, auto_enabled: false, auto_initial_enabled: false, daily_enabled: false, material_limit: 18, first_run_material_limit: 160, api_ready: false, state_path: '' },
    self_anchor: { entry_bucket_id: '' },
    merge_threshold: 90,
    transport: 'synthetic-test',
    buckets_dir: 'synthetic-test',
  };
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  let passed = 0;
  const stage = process.env.EDITOR_SCREENSHOT_DIR;
  if (stage) fs.mkdirSync(stage, { recursive: true });

  async function sourceFixture(width = 390, tasks = []) {
    const page = await browser.newPage({ viewport: { width, height: 844 } });
    page.on('pageerror', error => console.error('PAGEERROR', error));
    await page.route('**/*', route => route.abort());
    await page.setContent(inlineStyles + '<div class="content"><div class="bucket-bulk-message" id="daily-chat-memory-message"></div><div id="daily-chat-memory-pending"></div></div>');
    await page.evaluate((initialTasks) => {
      window.BASE = '';
      window.getActiveTab = () => 'other';
      window.confirm = () => true;
      window.esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
      window.escAttr = window.esc;
      window.jsString = value => String(value).replace(/'/g, "\\'");
      window.items = [{ id: 'draft-one', status: 'pending', authority_revision: 1,
        candidate: { title: '合成候选', content: '初始合成草稿', kind: 'preference', tags: [], domain: ['日常'],
          importance: 5, confidence: 0, original_excerpt: '合成来源',
          narrative: { event_time: { precision: 'day', value: '2026-10-09' } } }, display: {} }];
      window.sourceTasks = structuredClone(initialTasks);
      window.sourceTaskGets = 0;
      window.sourceTaskGetDelayQueue = [];
      window.sourceTaskRetries = [];
      window.failSourceTaskGet = false;
      window.failSourceTaskRetry = false;
      window.failGet = false;
      window.loadBuckets = () => {};
      window.authFetch = async (url, options) => {
        const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => structuredClone(data) });
        if (url.includes('/api/memory-authority/source-tasks')) {
          const retryMatch = url.match(/\/source-tasks\/([^/]+)\/retry$/);
          if (retryMatch && options && options.method === 'POST') {
            const body = JSON.parse(options.body);
            window.sourceTaskRetries.push({ url, body });
            if (window.failSourceTaskRetry) return reply({ error: 'synthetic_retry_conflict' }, 409);
            const taskId = decodeURIComponent(retryMatch[1]);
            const task = window.sourceTasks.find(entry => entry.task_id === taskId);
            if (!task || task.phase !== body.expected_phase) return reply({ error: 'task_phase_changed' }, 409);
            task.phase = 'queued';
            return reply({ status: 'queued', action_reexecuted: false, task });
          }
          window.sourceTaskGets++;
          const snapshot = structuredClone(window.sourceTasks);
          const delay = window.sourceTaskGetDelayQueue.shift() || 0;
          if (delay) await new Promise(resolve => setTimeout(resolve, delay));
          if (window.failSourceTaskGet) return reply({ error: 'synthetic_offline' }, 503);
          return reply({ tasks: snapshot });
        }
        if (url.includes('/api/daily-chat-memory/pending')) {
          if (window.failGet) throw new Error('synthetic offline');
          return reply({ items: window.items });
        }
        throw new Error('unexpected synthetic API: ' + url);
      };
    }, tasks);
    await page.addScriptTag({ content: asset });
    await page.evaluate(() => window.loadDailyChatMemoryPending());
    await page.waitForFunction(() => Boolean(document.querySelector('#memory-source-tasks [data-source-task-list]')));
    return page;
  }

  async function test(name, run) {
    await run();
    passed++;
    console.log('PASS', name);
  }

  try {
    let panelPage;
    await test('source task GET, retry POST, and refresh keep task state honest', async () => {
      const task = { task_id: 'action/42', action_id: 'act-42', phase: 'failed', issue: { code: 'synthetic_failure' }, candidate_ids: ['candidate-42'] };
      panelPage = await sourceFixture(1280, [task]);
      const list = panelPage.locator('#memory-source-tasks [data-source-task-list]');
      await panelPage.waitForFunction(() => document.querySelector('#memory-source-tasks')?.textContent.includes('整理失败'));
      assert.match(await list.textContent(), /synthetic_failure/);
      assert.match(await list.textContent(), /candidate-42/);
      await panelPage.locator('#memory-source-tasks > summary').click();
      const retry = panelPage.locator('#memory-source-tasks button', { hasText: '重试整理' });
      assert.equal(await retry.count(), 1);
      await retry.click();
      await panelPage.waitForFunction(() => document.querySelector('#memory-source-tasks')?.textContent.includes('等待整理'));
      const post = await panelPage.evaluate(() => sourceTaskRetries[0]);
      assert.equal(post.url, '/api/memory-authority/source-tasks/action%2F42/retry');
      assert.deepEqual(post.body, { expected_phase: 'failed' });
      assert.equal(await panelPage.evaluate(() => sourceTasks[0].phase), 'queued');

      await panelPage.locator('[data-candidate-id="draft-one"] button', { hasText: '编辑' }).click();
      const draft = panelPage.locator('[data-candidate-id="draft-one"] [data-field="content"]');
      await draft.fill('主人尚未保存的候选草稿');
      await panelPage.evaluate(() => { sourceTasks[0].phase = 'prepared'; });
      await panelPage.locator('#memory-source-tasks button', { hasText: '刷新准备状态' }).click();
      await panelPage.waitForFunction(() => document.querySelector('[data-source-task-notice]')?.textContent.includes('已刷新'));
      await panelPage.waitForFunction(() => document.querySelector('#memory-source-tasks')?.textContent.includes('候选已生成'));
      assert.equal(await draft.inputValue(), '主人尚未保存的候选草稿', 'source refresh leaves the dirty candidate editor untouched');

      await panelPage.locator('#memory-source-tasks').evaluate(node => { node.open = true; });
      if (stage) await panelPage.screenshot({ path: path.join(stage, 'memory-p2c-source-tasks-1280.png'), fullPage: true });
      await panelPage.setViewportSize({ width: 390, height: 844 });
      const sourceViewport = await panelPage.evaluate(() => ({ width: innerWidth, scrollWidth: document.documentElement.scrollWidth, overflow: [...document.querySelectorAll('*')].filter(node => node.getBoundingClientRect().right > innerWidth + 1).slice(0, 6).map(node => ({ tag: node.tagName, id: node.id, cls: String(node.className || ''), right: Math.round(node.getBoundingClientRect().right) })) }));
      assert.equal(sourceViewport.scrollWidth <= sourceViewport.width + 1, true, 'source task panel fits 390px: ' + JSON.stringify(sourceViewport));
      if (stage) await panelPage.screenshot({ path: path.join(stage, 'memory-p2c-source-tasks-390.png'), fullPage: true });

      await panelPage.evaluate(() => { failSourceTaskGet = true; });
      await panelPage.locator('#memory-source-tasks button', { hasText: '刷新准备状态' }).click();
      await panelPage.waitForFunction(() => document.querySelector('#memory-source-tasks [role="status"]')?.textContent.includes('可能已过期'));
      assert.match(await list.textContent(), /候选已生成/, 'failed refresh retains the last task panel receipt');
      assert.equal(await draft.inputValue(), '主人尚未保存的候选草稿', 'failed source refresh preserves the candidate draft');
      assert.ok(await panelPage.evaluate(() => sourceTaskGets >= 4), 'initial load, retry receipt, manual refresh, and failed refresh all reached synthetic GET');

      // A slower older receipt must not overwrite a later refresh response.
      await panelPage.evaluate(() => {
        failSourceTaskGet = false;
        sourceTasks[0].phase = 'failed';
        sourceTaskGetDelayQueue.push(100, 0);
      });
      const raceStart = await panelPage.evaluate(() => sourceTaskGets);
      await panelPage.locator('#memory-source-tasks button', { hasText: '刷新准备状态' }).click();
      await panelPage.waitForFunction(start => sourceTaskGets >= start + 1, raceStart);
      await panelPage.evaluate(() => { sourceTasks[0].phase = 'prepared'; });
      await panelPage.locator('#memory-source-tasks button', { hasText: '刷新准备状态' }).click();
      await panelPage.waitForFunction(start => sourceTaskGets >= start + 2, raceStart);
      await panelPage.waitForFunction(() => document.querySelector('#memory-source-tasks [role="status"]')?.textContent.includes('已刷新'));
      await panelPage.waitForTimeout(150);
      assert.match(await list.textContent(), /候选已生成/, 'late older source response cannot replace the newer receipt');
      assert.equal(await draft.inputValue(), '主人尚未保存的候选草稿');
    });
    await panelPage.close();
    panelPage = null;

    await test('failed source retry keeps the failure receipt and re-enables the action', async () => {
      const page = await sourceFixture(390, [{ task_id: 'task-fail', action_id: 'act-fail', phase: 'failed', issue: { code: 'synthetic_failure' }, candidate_ids: [] }]);
      try {
        await page.evaluate(() => { failSourceTaskRetry = true; });
        await page.locator('#memory-source-tasks > summary').click();
        const retry = page.locator('#memory-source-tasks button', { hasText: '重试整理' });
        await retry.click();
        await page.waitForFunction(() => document.querySelector('#memory-source-tasks')?.textContent.includes('未重新排队'));
        assert.match(await page.locator('#memory-source-tasks').textContent(), /synthetic_failure/, 'source failure code remains visible after a rejected retry');
        assert.equal(await retry.isEnabled(), true);
        assert.deepEqual(await page.evaluate(() => sourceTaskRetries[0].body), { expected_phase: 'failed' });
      } finally { await page.close(); }
    });

    const api = { gets: 0, posts: [] };
    const config = configFixture();
    const configPage = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const pageErrors = [];
    configPage.on('pageerror', error => pageErrors.push(error.message));
    await configPage.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      if (url.pathname === '/dashboard') {
        return route.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: dashboardMarkup });
      }
      if (url.pathname === '/api/config' && request.method() === 'GET') {
        api.gets++;
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(config) });
      }
      if (url.pathname === '/api/config' && request.method() === 'POST') {
        api.posts.push(request.postDataJSON());
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
          ok: false, error: 'runtime_persist_failed', persistence: { runtime_yaml: 'failed', credentials: 'not_requested' },
        }) });
      }
      return route.abort();
    });
    try {
      await configPage.goto('http://p2c.test/dashboard');
      await configPage.addScriptTag({ content: dashboardScript });
      await configPage.evaluate(() => { document.getElementById('auth-overlay').style.display = 'none'; });
      await configPage.locator('.tab[data-tab="config"]').click();
      await configPage.waitForFunction(() => document.querySelector('#cfg-reflection-diary-l1-input-window-chars')?.value === '4096', null, { timeout: 5000 }).catch(async () => {
        const debug = await configPage.evaluate(() => ({ active: document.querySelector('.tab.active')?.dataset.tab, base: BASE, status: document.querySelector('#config-status')?.textContent, value: document.querySelector('#cfg-reflection-diary-l1-input-window-chars')?.value, loadConfigType: typeof loadConfig }));
        throw new Error('synthetic config GET did not populate controls: ' + JSON.stringify({ debug, apiGets: api.gets, pageErrors }));
      });
      assert.equal(await configPage.locator('#cfg-reflection-diary-l1-input-window-chars').getAttribute('min'), '256');
      assert.equal(await configPage.locator('#cfg-reflection-action-l1-jobs-per-run').getAttribute('max'), '10000');
      await configPage.locator('#cfg-reflection-diary-l1-input-window-chars').fill('24000');
      await configPage.locator('#cfg-reflection-diary-l1-source-kind').selectOption('private');
      await configPage.locator('#cfg-reflection-action-l1-jobs-per-run').fill('17');
      await configPage.locator('#config-view button[onclick="saveConfig(true)"]').click();
      await configPage.waitForFunction(() => document.querySelector('#config-status')?.textContent.includes('配置落盘失败'), null, { timeout: 5000 }).catch(async () => {
        const debug = await configPage.evaluate(() => ({ active: document.querySelector('.tab.active')?.dataset.tab, status: document.querySelector('#config-status')?.textContent, source: document.querySelector('#cfg-reflection-diary-l1-source-kind')?.value, actionJobs: document.querySelector('#cfg-reflection-action-l1-jobs-per-run')?.value }));
        throw new Error('synthetic config save did not reach the expected failure response: ' + JSON.stringify({ debug, apiPosts: api.posts, pageErrors }));
      });
      assert.equal(api.posts.length, 1, 'save submits one synthetic API request');
      assert.equal(api.posts[0].reflection.diary_l1_input_window_chars, 24000);
      assert.equal(api.posts[0].reflection.diary_l1_source_kind, 'private');
      assert.equal(api.posts[0].reflection.action_l1_jobs_per_run, 17);
      assert.match(await configPage.locator('#config-status').textContent(), /输入已保留/);

      await configPage.locator('.tab[data-tab="model-config"]').click();
      await configPage.locator('.tab[data-tab="config"]').click();
      await configPage.waitForFunction(() => document.querySelector('#cfg-reflection-diary-l1-source-kind')?.value === 'private');
      assert.equal(await configPage.locator('#cfg-reflection-diary-l1-input-window-chars').inputValue(), '24000');
      assert.equal(await configPage.locator('#cfg-reflection-action-l1-jobs-per-run').inputValue(), '17');
      assert.ok(api.gets >= 3, 'configuration tab re-read synthetic server config');

      if (stage) await configPage.screenshot({ path: path.join(stage, 'memory-p2c-config-1280.png'), fullPage: true });
      await configPage.setViewportSize({ width: 390, height: 844 });
      assert.equal(await configPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true, 'config settings fit 390px');
      if (stage) await configPage.screenshot({ path: path.join(stage, 'memory-p2c-config-390.png'), fullPage: true });
      assert.deepEqual(pageErrors, [], 'dashboard inline script has no browser exceptions');
      passed++;
      console.log('PASS live dashboard config GET/POST wiring and failed-save draft retention');
    } finally { await configPage.close(); }

    console.log(JSON.stringify({
      passed,
      browser: 'Microsoft Edge headless',
      real_requests: 0,
      production_writes: 0,
      synthetic_config_gets: api.gets,
      synthetic_config_posts: api.posts.length,
      screenshots: stage ? ['memory-p2c-source-tasks-1280.png', 'memory-p2c-source-tasks-390.png', 'memory-p2c-config-1280.png', 'memory-p2c-config-390.png'] : [],
    }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
