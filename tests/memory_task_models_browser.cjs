// Real headless browser, synthetic configuration fixtures and mocked HTTP only.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const script = fs.readFileSync(path.join(root, 'dashboard_assets/memory-task-models.js'), 'utf8');
const stageIds = ['auto_l1', 'chat_l1', 'action_l1', 'diary_l1', 'presentation_l2', 'chat_summary', 'activity_summary'];
const OUTPUT_SCHEMA = {
  type: 'object',
  required: ['full_injection_text', 'coherent_sections', 'optional_use_variants', 'review'],
  properties: {
    full_injection_text: { type: 'string' },
    coherent_sections: { type: 'array' },
    optional_use_variants: { type: 'array' },
    review: { type: 'object' },
  },
};

function makeRoutes() {
  return Object.fromEntries(stageIds.map((stage) => [stage, {
    connection: 'legacy',
    model: '',
    thinking_mode: 'inherit',
    reasoning_effort: '',
    max_tokens: null,
  }]));
}

function makeEffective(routes) {
  return Object.fromEntries(stageIds.map((stage) => {
    const route = routes[stage];
    const unavailable = route.connection === 'daily';
    return [stage, {
      stage,
      connection: route.connection,
      model: unavailable ? '' : (route.model || `resolved-${stage}`),
      thinking_mode: route.thinking_mode === 'inherit' ? 'enabled' : (route.thinking_mode === 'default' ? '' : route.thinking_mode),
      reasoning_effort: route.reasoning_effort,
      max_tokens: route.max_tokens,
      configured: !unavailable,
      unavailable_reason: unavailable ? 'daily connection is not configured' : '',
      model_source: route.model ? 'route.model' : 'connection default',
      thinking_source: route.thinking_mode,
      max_tokens_source: route.max_tokens === null ? 'provider default' : 'route.max_tokens',
    }];
  }));
}

function makeConfig(routes = makeRoutes(), revision = 'revision-1', schema = OUTPUT_SCHEMA, schemaRevision = 'schema-revision-1') {
  return {
    reflection: {
      api_key: 'SYNTHETIC_SECRET_SHOULD_NOT_RENDER',
      memory_task_routes: structuredClone(routes),
      memory_task_effective: makeEffective(routes),
      memory_task_revision: revision,
      memory_l2_output_schema: structuredClone(schema),
      memory_l2_schema_revision: schemaRevision,
      memory_l2_jobs_per_run: 1,
      memory_l2_concurrency: 1,
      memory_l2_prepare_interval_seconds: 5,
    },
  };
}

async function makePage(browser, width = 390) {
  const page = await browser.newPage({ viewport: { width, height: 920 } });
  page.on('pageerror', (error) => console.error('PAGEERROR', error));
  await page.route('http://memory-task.test/**', (route) => route.fulfill({
    status: 200,
    contentType: 'text/html',
    body: '<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><style>:root{--bg:#FDFCF0;--bg-gradient:radial-gradient(ellipse at 30% 20%,#FDFCF0 0%,#EDE4D3 60%,#E2D1B3 100%);--surface:rgba(255,255,255,.45);--surface-solid:#F7F2E8;--border:rgba(180,165,140,.3);--border-strong:rgba(160,140,110,.5);--text:#3A3530;--text-dim:#8A8070;--text-light:#B0A590;--accent:#2F4F4F;--accent-light:#3D6363;--positive:#4A7C59;--negative:#8B4A4A;--warning:#9A7B4F}body{margin:0;background:var(--bg-gradient);color:var(--text)}</style></head><body><main id="memory-task-models"></main></body></html>',
  }));
  const initialConfigJson = JSON.stringify(makeConfig()).replace(/</g, '\\u003c');
  await page.addInitScript(`(() => {
    const stages = ${JSON.stringify(stageIds)};
    const buildEffective = (routes) => Object.fromEntries(stages.map((stage) => {
      const route = routes[stage];
      const unavailable = route.connection === 'daily';
      return [stage, {
        stage,
        connection: route.connection,
        model: unavailable ? '' : (route.model || 'resolved-' + stage),
        thinking_mode: route.thinking_mode === 'inherit' ? 'enabled' : (route.thinking_mode === 'default' ? '' : route.thinking_mode),
        reasoning_effort: route.reasoning_effort,
        max_tokens: route.max_tokens,
        configured: !unavailable,
        unavailable_reason: unavailable ? 'daily connection is not configured' : '',
        model_source: route.model ? 'route.model' : 'connection default',
        thinking_source: route.thinking_mode,
        max_tokens_source: route.max_tokens === null ? 'provider default' : 'route.max_tokens',
      }];
    }));
    window.BASE = '/dashboard';
    window.callLog = [];
    window.serverConfig = ${initialConfigJson};
    window.failPost = false;
    window.failNextGet = false;
    window.keepRoutesOnPost = false;
    window.keepSchemaOnPost = false;
    window.conflictNextPost = false;
    window.schemaConflictNextPost = false;
    window.schemaRevisionCounter = 1;
    window.xssTriggered = false;
    window.revisionCounter = 1;
    window.buildEffective = buildEffective;
    const reply = (data, status = 200) => ({ ok: status >= 200 && status < 400, status, json: async () => structuredClone(data) });
    window.authFetch = async (url, options) => {
      window.callLog.push({ url, method: options?.method || 'GET', body: options?.body || null });
      if (url !== '/dashboard/api/config') throw new Error('unexpected endpoint');
      if (!options) {
        if (window.failNextGet) {
          window.failNextGet = false;
          throw new Error('synthetic network failure');
        }
        return reply(window.serverConfig);
      }
      if (options.method !== 'POST') throw new Error('unexpected method');
      const body = JSON.parse(options.body);
      window.lastPostedBody = body;
      if (window.failPost) return reply({ ok: false, error: 'synthetic failure' }, 503);
      const reflectionPatch = body.reflection || {};
      const taskOptionKeys = ['memory_l2_jobs_per_run', 'memory_l2_concurrency', 'memory_l2_prepare_interval_seconds'];
      const hasTaskPatch = Object.prototype.hasOwnProperty.call(reflectionPatch, 'memory_task_routes')
        || taskOptionKeys.some((key) => Object.prototype.hasOwnProperty.call(reflectionPatch, key));
      const taskConflict = Object.prototype.hasOwnProperty.call(body, 'expected_memory_task_revision')
        && body.expected_memory_task_revision !== window.serverConfig.reflection.memory_task_revision;
      const schemaConflict = Object.prototype.hasOwnProperty.call(body, 'expected_memory_l2_schema_revision')
        && body.expected_memory_l2_schema_revision !== window.serverConfig.reflection.memory_l2_schema_revision;
      if (window.schemaConflictNextPost) {
        window.schemaConflictNextPost = false;
        window.serverConfig.reflection.memory_l2_output_schema = structuredClone(window.conflictingOutputSchema || {
          ...window.serverConfig.reflection.memory_l2_output_schema,
          description: '服务器并发更新',
        });
        window.schemaRevisionCounter += 1;
        window.serverConfig.reflection.memory_l2_schema_revision = 'schema-revision-' + window.schemaRevisionCounter;
        return reply({ ok: false, error: 'memory_l2_schema_revision_conflict' }, 409);
      }
      if (window.conflictNextPost || taskConflict || schemaConflict) {
        window.conflictNextPost = false;
        return reply({ ok: false, error: 'memory_task_revision_conflict' }, 409);
      }
      if (hasTaskPatch && !window.keepRoutesOnPost) {
        if (Object.prototype.hasOwnProperty.call(reflectionPatch, 'memory_task_routes')) {
          window.serverConfig.reflection.memory_task_routes = structuredClone(reflectionPatch.memory_task_routes);
          window.serverConfig.reflection.memory_task_effective = buildEffective(reflectionPatch.memory_task_routes);
        }
        if (Object.prototype.hasOwnProperty.call(reflectionPatch, 'memory_l2_jobs_per_run')) {
          window.serverConfig.reflection.memory_l2_jobs_per_run = reflectionPatch.memory_l2_jobs_per_run;
        }
        if (Object.prototype.hasOwnProperty.call(reflectionPatch, 'memory_l2_concurrency')) {
          window.serverConfig.reflection.memory_l2_concurrency = reflectionPatch.memory_l2_concurrency;
        }
        if (Object.prototype.hasOwnProperty.call(reflectionPatch, 'memory_l2_prepare_interval_seconds')) {
          window.serverConfig.reflection.memory_l2_prepare_interval_seconds = reflectionPatch.memory_l2_prepare_interval_seconds;
        }
        window.revisionCounter += 1;
        window.serverConfig.reflection.memory_task_revision = 'revision-' + window.revisionCounter;
      }
      if (Object.prototype.hasOwnProperty.call(reflectionPatch, 'memory_l2_output_schema')) {
        if (!window.keepSchemaOnPost) {
          window.serverConfig.reflection.memory_l2_output_schema = structuredClone(reflectionPatch.memory_l2_output_schema);
          window.schemaRevisionCounter += 1;
          window.serverConfig.reflection.memory_l2_schema_revision = 'schema-revision-' + window.schemaRevisionCounter;
        }
      }
      return reply({ ok: true });
    };
  })();`);
  await page.goto('http://memory-task.test/');
  await page.addScriptTag({ content: script });
  await page.evaluate(() => window.renderMemoryTaskModels());
  await page.waitForFunction(() => document.querySelector('#memory-task-models .mtm-status')?.textContent.includes('已从服务器读取'));
  return page;
}

async function expectNoOverflow(page, expectedColumns) {
  const result = await page.evaluate(() => ({
    width: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
    columns: getComputedStyle(document.querySelector('.mtm-grid')).gridTemplateColumns,
  }));
  assert.ok(result.scrollWidth <= result.width + 1, `horizontal overflow at ${result.width}px: ${result.scrollWidth}px`);
  if (expectedColumns === 1) assert.equal(result.columns.trim().split(/\s+/).length, 1);
  if (expectedColumns === 2) assert.equal(result.columns.trim().split(/\s+/).length, 2);
}

async function expectDashboardTheme(page) {
  const colors = await page.evaluate(() => ({
    cardBackground: getComputedStyle(document.querySelector('.mtm-stage')).backgroundColor,
    cardText: getComputedStyle(document.querySelector('.mtm-stage')).color,
    summaryWhiteSpace: getComputedStyle(document.querySelector('.mtm-effective-summary')).whiteSpace,
    documentHeight: document.documentElement.scrollHeight,
  }));
  assert.equal(colors.cardBackground, 'rgb(247, 242, 232)');
  assert.equal(colors.cardText, 'rgb(58, 53, 48)');
  assert.equal(colors.summaryWhiteSpace, 'nowrap');
  assert.ok(colors.documentHeight < 3000, `default seven-stage 390px layout should remain bounded: ${colors.documentHeight}px`);
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await makePage(browser, 390);
    await expectNoOverflow(page, 1);
    await expectDashboardTheme(page);
    assert.equal(await page.locator('.mtm-stage').count(), 7);
    assert.equal(await page.locator('input[data-route-field="model"]').count(), 7);
    assert.equal(await page.locator('#mtm-presentation_l2-model').count(), 1, 'presentation_l2 stays visible as its own adjustable stage');
    assert.equal(await page.locator('#mtm-memory_l2_jobs_per_run').inputValue(), '1');
    assert.equal(await page.locator('#mtm-memory_l2_concurrency').inputValue(), '1');
    assert.equal(await page.locator('#mtm-memory_l2_prepare_interval_seconds').inputValue(), '5');
    assert.equal(await page.locator('.mtm-more-settings[open]').count(), 0);
    assert.equal(await page.locator('.mtm-effective-details[open]').count(), 0);
    assert.equal(await page.locator('#memory-task-models').getByText('SYNTHETIC_SECRET_SHOULD_NOT_RENDER').count(), 0);
    assert.equal(await page.locator('#memory-task-models').getByText('独立 L2').count(), 0);

    const screenshotDir = path.resolve(root, '..', '..', 'staging', 'memory-l2-p4-ui-20261018');
    fs.mkdirSync(screenshotDir, { recursive: true });
    const mobileScreenshot = path.join(screenshotDir, 'memory-task-models-390.png');
    const desktopScreenshot = path.join(screenshotDir, 'memory-task-models-1280.png');
    await page.screenshot({ path: mobileScreenshot, fullPage: true });

    await page.setViewportSize({ width: 1280, height: 900 });
    await expectNoOverflow(page, 2);
    await page.screenshot({ path: desktopScreenshot, fullPage: true });

    // A dashboard refresh while editing may update server state, but cannot replace the local draft.
    const modelInput = page.locator('#mtm-chat_l1-model');
    await modelInput.fill('草稿-必须保留');
    const savedDraft = await page.evaluate(() => {
      const draft = JSON.parse(sessionStorage.getItem('ombre.memory-task-models.draft.v1'));
      return { keys: Object.keys(draft).sort(), baseRevision: draft.base_revision, json: JSON.stringify(draft) };
    });
    assert.deepEqual(savedDraft.keys, ['base_output_schema', 'base_output_schema_revision', 'base_revision', 'base_routes', 'base_task_options', 'output_schema_text', 'routes', 'task_options', 'version']);
    assert.equal(savedDraft.baseRevision, 'revision-1');
    assert.equal(JSON.parse(savedDraft.json).output_schema_text, null, 'route-only drafts do not claim an unedited schema');
    assert.doesNotMatch(savedDraft.json, /api_key|SYNTHETIC_SECRET|memory_task_effective/);
    await page.reload();
    await page.addScriptTag({ content: script });
    await page.evaluate(() => window.renderMemoryTaskModels());
    await page.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('已恢复此标签页的未保存草稿'));
    assert.equal(await modelInput.inputValue(), '草稿-必须保留');
    await page.evaluate(() => window.renderMemoryTaskModels(structuredClone(window.serverConfig)));
    assert.equal(await modelInput.inputValue(), '草稿-必须保留');
    assert.match(await page.locator('.mtm-draft-state').textContent(), /有未保存更改/);

    // Failed POST and failed/mismatched readback all keep the entered value and never report success.
    await page.evaluate(() => { window.failPost = true; });
    await page.locator('[data-action="save-memory-task-models"]').click();
    await page.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('保存失败'));
    assert.equal(await modelInput.inputValue(), '草稿-必须保留');
    assert.match(await page.locator('.mtm-draft-state').textContent(), /有未保存更改/);

    await page.evaluate(() => {
      window.failPost = false;
      window.failNextGet = true;
    });
    await page.locator('[data-action="save-memory-task-models"]').click();
    await page.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('读回失败'));
    assert.equal(await modelInput.inputValue(), '草稿-必须保留');

    await page.evaluate(() => window.renderMemoryTaskModels(structuredClone(window.serverConfig)));
    assert.equal(await page.locator('[data-action="rebase-memory-task-models"]').isVisible(), true);
    await page.locator('[data-action="rebase-memory-task-models"]').click();
    await page.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('已核对服务器新版'));
    assert.equal(await modelInput.inputValue(), '草稿-必须保留');

    await page.evaluate(() => { window.keepRoutesOnPost = true; });
    await modelInput.fill('读回不一致草稿');
    await page.locator('[data-action="save-memory-task-models"]').click();
    await page.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('读回与草稿不一致'));
    assert.equal(await modelInput.inputValue(), '读回不一致草稿');

    // Successful save posts only the named section, then renders actual readback as text.
    await page.evaluate(() => {
      window.keepRoutesOnPost = false;
      window.failPost = false;
    });
    await page.locator('#mtm-chat_l1-connection').evaluate((el) => { el.closest('details').open = true; });
    await page.locator('#mtm-chat_l1-connection').selectOption('daily');
    await page.locator('#mtm-action_l1-model').fill('<img src=x onerror="window.xssTriggered=true">demo-model');
    await page.locator('#mtm-chat_l1-thinking_mode').selectOption('default');
    await page.locator('#mtm-chat_l1-reasoning_effort').selectOption('high');
    await page.locator('#mtm-chat_l1-max_tokens').fill('777');
    await page.locator('#mtm-action_l1-reasoning_effort').evaluate((el) => { el.closest('details').open = true; });
    await page.locator('#mtm-action_l1-reasoning_effort').selectOption('medium');
    await page.locator('[data-action="save-memory-task-models"]').click();
    await page.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('部分任务连接未就绪'));
    await page.locator('#mtm-chat_l1-connection').evaluate((el) => { el.closest('.mtm-stage').querySelector('.mtm-effective-details').open = true; });
    await page.locator('#mtm-action_l1-model').evaluate((el) => { el.closest('.mtm-stage').querySelector('.mtm-effective-details').open = true; });

    const posted = await page.evaluate(() => ({
      body: window.lastPostedBody,
      methods: window.callLog.map((call) => call.method),
      postCount: window.callLog.filter((call) => call.method === 'POST').length,
      xssTriggered: window.xssTriggered,
      hasInjectedImage: !!document.querySelector('#memory-task-models img'),
      text: document.querySelector('#memory-task-models').textContent,
    }));
    assert.deepEqual(Object.keys(posted.body).sort(), ['expected_memory_task_revision', 'persist', 'reflection']);
    assert.equal(posted.body.persist, true);
    assert.equal(posted.body.expected_memory_task_revision, 'revision-2');
    assert.deepEqual(Object.keys(posted.body.reflection), ['memory_task_routes']);
    assert.deepEqual(Object.keys(posted.body.reflection.memory_task_routes).sort(), stageIds.slice().sort());
    assert.equal(posted.body.reflection.memory_task_routes.chat_l1.connection, 'daily');
    assert.equal(posted.body.reflection.memory_task_routes.action_l1.model, '<img src=x onerror="window.xssTriggered=true">demo-model');
    assert.equal(posted.body.reflection.memory_task_routes.action_l1.reasoning_effort, 'medium');
    assert.equal(posted.body.reflection.memory_task_routes.chat_l1.thinking_mode, 'default');
    assert.equal(posted.body.reflection.memory_task_routes.chat_l1.reasoning_effort, 'high');
    assert.equal(posted.body.reflection.memory_task_routes.chat_l1.max_tokens, 777);
    assert.equal(posted.postCount, 4);
    assert.ok(posted.methods.includes('GET'), 'initial config uses the existing GET endpoint');
    assert.equal(posted.xssTriggered, false);
    assert.equal(posted.hasInjectedImage, false);
    assert.match(posted.text, /daily connection is not configured/);
    assert.match(posted.text, /未发送 thinking 参数/);
    assert.match(posted.text, /强度 medium/);
    assert.match(posted.text, /<img src=x onerror=/);
    assert.doesNotMatch(posted.text, /SYNTHETIC_SECRET_SHOULD_NOT_RENDER/);
    assert.match(await page.locator('.mtm-status').textContent(), /部分任务连接未就绪/);
    await page.close();

    const conflictPage = await makePage(browser, 390);
    await conflictPage.locator('#mtm-chat_l1-model').fill('本地草稿');
    await conflictPage.evaluate(() => {
      const reflection = window.serverConfig.reflection;
      reflection.memory_task_routes.chat_summary.model = '服务器并发更新';
      reflection.memory_task_effective = window.buildEffective(reflection.memory_task_routes);
      reflection.memory_task_revision = 'revision-2';
      window.revisionCounter = 2;
      window.conflictNextPost = true;
    });
    await conflictPage.locator('[data-action="save-memory-task-models"]').click();
    await conflictPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('配置有新版本'));
    const conflictBody = await conflictPage.evaluate(() => window.lastPostedBody);
    assert.equal(conflictBody.expected_memory_task_revision, 'revision-1');
    assert.equal(await conflictPage.locator('#mtm-chat_l1-model').inputValue(), '本地草稿');
    assert.equal(await conflictPage.locator('[data-action="rebase-memory-task-models"]').isVisible(), true);
    await conflictPage.locator('[data-action="rebase-memory-task-models"]').click();
    await conflictPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('已核对服务器新版'));
    assert.equal(await conflictPage.locator('#mtm-chat_l1-model').inputValue(), '本地草稿');
    assert.equal(await conflictPage.locator('#mtm-chat_summary-model').inputValue(), '服务器并发更新');
    await conflictPage.locator('[data-action="save-memory-task-models"]').click();
    await conflictPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('已保存并应用'));
    const rebasedPost = await conflictPage.evaluate(() => ({
      expectedRevision: window.lastPostedBody.expected_memory_task_revision,
      serverRoutes: window.serverConfig.reflection.memory_task_routes,
    }));
    assert.equal(rebasedPost.expectedRevision, 'revision-2');
    assert.equal(rebasedPost.serverRoutes.chat_l1.model, '本地草稿');
    assert.equal(rebasedPost.serverRoutes.chat_summary.model, '服务器并发更新');
    await conflictPage.close();

    console.log('PASS memory-task-models: 390px and 1280px layout, seven visible stages, GET initialization, target-section-only CAS save, exact readback, refresh/error/conflict draft preservation and rebase, L2 job/interval/concurrency settings, daily unavailable reason, XSS rendered as text');

    const optionsPage = await makePage(browser, 390);
    const jobsInput = optionsPage.locator('#mtm-memory_l2_jobs_per_run');
    const concurrencyInput = optionsPage.locator('#mtm-memory_l2_concurrency');
    const intervalInput = optionsPage.locator('#mtm-memory_l2_prepare_interval_seconds');
    await jobsInput.fill('201');
    await optionsPage.locator('[data-action="save-memory-task-models"]').click();
    await optionsPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('0 到 200 之间的整数'));
    assert.equal(await optionsPage.evaluate(() => window.callLog.filter((call) => call.method === 'POST').length), 0);
    assert.equal(await jobsInput.inputValue(), '201', 'out-of-range values stay visible for correction');

    await jobsInput.fill('1');
    await concurrencyInput.fill('0');
    await optionsPage.locator('[data-action="save-memory-task-models"]').click();
    await optionsPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('1 到 200 之间的整数'));
    assert.equal(await optionsPage.evaluate(() => window.callLog.filter((call) => call.method === 'POST').length), 0);
    assert.equal(await concurrencyInput.inputValue(), '0', 'out-of-range concurrency stays visible for correction');

    await jobsInput.fill('0');
    await concurrencyInput.fill('200');
    await intervalInput.fill('3600');
    await optionsPage.locator('[data-action="save-memory-task-models"]').click();
    await optionsPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('已保存并应用'));
    const optionsPost = await optionsPage.evaluate(() => ({
      body: window.lastPostedBody,
      reflection: window.serverConfig.reflection,
      calls: window.callLog,
    }));
    assert.deepEqual(optionsPost.body.reflection, {
      memory_l2_jobs_per_run: 0,
      memory_l2_concurrency: 200,
      memory_l2_prepare_interval_seconds: 3600,
    });
    assert.equal(optionsPost.reflection.memory_l2_jobs_per_run, 0);
    assert.equal(optionsPost.reflection.memory_l2_concurrency, 200);
    assert.equal(optionsPost.reflection.memory_l2_prepare_interval_seconds, 3600);
    assert.ok(optionsPost.calls.filter((call) => call.method === 'GET').length >= 2,
      'editing L2 frequency checks the current reflection values before the CAS-backed save');
    await optionsPage.close();

    const optionsConflictPage = await makePage(browser, 390);
    const conflictJobsInput = optionsConflictPage.locator('#mtm-memory_l2_jobs_per_run');
    await conflictJobsInput.fill('7');
    await optionsConflictPage.evaluate(() => { window.serverConfig.reflection.memory_l2_jobs_per_run = 2; });
    await optionsConflictPage.locator('[data-action="save-memory-task-models"]').click();
    await optionsConflictPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('服务器任务配置或第二遍输出 Schema 已有更新'));
    assert.equal(await optionsConflictPage.evaluate(() => window.callLog.filter((call) => call.method === 'POST').length), 0,
      'a preflight conflict does not overwrite a newer scheduler value');
    assert.equal(await conflictJobsInput.inputValue(), '7');
    await optionsConflictPage.locator('[data-action="rebase-memory-task-models"]').click();
    await optionsConflictPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('已核对服务器新版'));
    assert.equal(await conflictJobsInput.inputValue(), '7', 'explicit rebase preserves the owner edit');
    await optionsConflictPage.locator('[data-action="save-memory-task-models"]').click();
    await optionsConflictPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('已保存并应用'));
    assert.equal(await optionsConflictPage.evaluate(() => window.serverConfig.reflection.memory_l2_jobs_per_run), 7);
    await optionsConflictPage.close();

    const invalidSchemaPage = await makePage(browser, 390);
    const invalidSchemaInput = invalidSchemaPage.locator('#mtm-memory_l2_output_schema');
    assert.equal(await invalidSchemaInput.inputValue(), JSON.stringify(OUTPUT_SCHEMA, null, 2),
      'the schema editor is initialized from the GET payload, not a JavaScript copy');
    assert.equal(await invalidSchemaPage.locator('.mtm-schema[open]').count(), 0,
      'the long schema editor starts collapsed');
    await invalidSchemaPage.locator('.mtm-schema').evaluate((node) => { node.open = true; });
    await invalidSchemaInput.fill('{"type":');
    await invalidSchemaPage.locator('[data-action="save-memory-task-models"]').click();
    await invalidSchemaPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('JSON 语法无效'));
    assert.equal(await invalidSchemaPage.evaluate(() => window.callLog.filter((call) => call.method === 'POST').length), 0,
      'invalid JSON blocks every save request');
    const invalidDraft = await invalidSchemaPage.evaluate(() => JSON.parse(sessionStorage.getItem('ombre.memory-task-models.draft.v1')));
    assert.equal(invalidDraft.output_schema_text, '{"type":');
    await invalidSchemaPage.reload();
    await invalidSchemaPage.addScriptTag({ content: script });
    await invalidSchemaPage.evaluate(() => window.renderMemoryTaskModels());
    await invalidSchemaPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('已恢复此标签页的未保存草稿'));
    assert.equal(await invalidSchemaInput.inputValue(), '{"type":', 'invalid schema draft survives a page reload');
    await invalidSchemaPage.close();

    const schemaOnlyPage = await makePage(browser, 390);
    const schemaOnlyInput = schemaOnlyPage.locator('#mtm-memory_l2_output_schema');
    await schemaOnlyPage.locator('.mtm-schema').evaluate((node) => { node.open = true; });
    const ownerSchema = { ...OUTPUT_SCHEMA, description: 'owner schema-only edit' };
    await schemaOnlyInput.fill(JSON.stringify(ownerSchema, null, 2));
    await schemaOnlyPage.locator('[data-action="save-memory-task-models"]').click();
    await schemaOnlyPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('schema-revision-2'));
    const schemaOnlyPost = await schemaOnlyPage.evaluate(() => ({
      body: window.lastPostedBody,
      schema: window.serverConfig.reflection.memory_l2_output_schema,
      schemaRevision: window.serverConfig.reflection.memory_l2_schema_revision,
      taskRevision: window.serverConfig.reflection.memory_task_revision,
    }));
    assert.deepEqual(Object.keys(schemaOnlyPost.body).sort(), [
      'expected_memory_l2_schema_revision', 'persist', 'reflection',
    ]);
    assert.deepEqual(Object.keys(schemaOnlyPost.body.reflection), ['memory_l2_output_schema'],
      'schema-only saves do not submit routes or task options');
    assert.equal(schemaOnlyPost.body.expected_memory_l2_schema_revision, 'schema-revision-1');
    assert.equal(Object.prototype.hasOwnProperty.call(schemaOnlyPost.body, 'expected_memory_task_revision'), false,
      'schema-only saves use only the schema CAS');
    assert.deepEqual(schemaOnlyPost.schema, ownerSchema);
    assert.equal(schemaOnlyPost.schemaRevision, 'schema-revision-2');
    assert.equal(schemaOnlyPost.taskRevision, 'revision-1');
    assert.match(await schemaOnlyPage.locator('.mtm-status').textContent(), /已保存并应用/);
    await schemaOnlyPage.close();

    const combinedPage = await makePage(browser, 390);
    await combinedPage.locator('.mtm-schema').evaluate((node) => { node.open = true; });
    await combinedPage.locator('#mtm-memory_l2_output_schema').fill(JSON.stringify(ownerSchema, null, 2));
    await combinedPage.locator('#mtm-memory_l2_jobs_per_run').fill('4');
    await combinedPage.locator('[data-action="save-memory-task-models"]').click();
    await combinedPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('schema-revision-2'));
    const combinedPost = await combinedPage.evaluate(() => ({
      body: window.lastPostedBody,
      schema: window.serverConfig.reflection.memory_l2_output_schema,
      taskRevision: window.serverConfig.reflection.memory_task_revision,
      schemaRevision: window.serverConfig.reflection.memory_l2_schema_revision,
    }));
    assert.deepEqual(Object.keys(combinedPost.body).sort(), [
      'expected_memory_l2_schema_revision', 'expected_memory_task_revision', 'persist', 'reflection',
    ]);
    assert.deepEqual(Object.keys(combinedPost.body.reflection).sort(), [
      'memory_l2_jobs_per_run', 'memory_l2_output_schema',
    ], 'combined saves submit only dirty schema and scheduler fields');
    assert.equal(combinedPost.body.expected_memory_task_revision, 'revision-1');
    assert.equal(combinedPost.body.expected_memory_l2_schema_revision, 'schema-revision-1');
    assert.deepEqual(combinedPost.schema, ownerSchema);
    assert.equal(combinedPost.taskRevision, 'revision-2');
    assert.equal(combinedPost.schemaRevision, 'schema-revision-2');
    await combinedPage.close();

    const schemaConflictPage = await makePage(browser, 390);
    await schemaConflictPage.locator('.mtm-schema').evaluate((node) => { node.open = true; });
    const localSchema = { ...OUTPUT_SCHEMA, description: 'local schema draft' };
    const serverSchema = { ...OUTPUT_SCHEMA, description: 'server schema revision' };
    const conflictSchemaInput = schemaConflictPage.locator('#mtm-memory_l2_output_schema');
    await conflictSchemaInput.fill(JSON.stringify(localSchema, null, 2));
    await schemaConflictPage.evaluate((newerSchema) => {
      window.conflictingOutputSchema = newerSchema;
      window.schemaConflictNextPost = true;
    }, serverSchema);
    await schemaConflictPage.locator('[data-action="save-memory-task-models"]').click();
    await schemaConflictPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('配置有新版本'));
    const conflictedBody = await schemaConflictPage.evaluate(() => window.lastPostedBody);
    assert.equal(conflictedBody.expected_memory_l2_schema_revision, 'schema-revision-1');
    assert.equal(Object.prototype.hasOwnProperty.call(conflictedBody, 'expected_memory_task_revision'), false);
    assert.equal(await conflictSchemaInput.inputValue(), JSON.stringify(localSchema, null, 2),
      'a schema CAS conflict does not replace the local draft');
    assert.equal(await schemaConflictPage.locator('[data-action="rebase-memory-task-models"]').isVisible(), true);
    await schemaConflictPage.locator('[data-action="rebase-memory-task-models"]').click();
    await schemaConflictPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('已核对服务器新版'));
    assert.equal(await conflictSchemaInput.inputValue(), JSON.stringify(localSchema, null, 2),
      'explicit rebase preserves the owner schema edit');
    await schemaConflictPage.locator('[data-action="save-memory-task-models"]').click();
    await schemaConflictPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('schema-revision-3'));
    const rebasedSchemaPost = await schemaConflictPage.evaluate(() => ({
      body: window.lastPostedBody,
      schema: window.serverConfig.reflection.memory_l2_output_schema,
      schemaRevision: window.serverConfig.reflection.memory_l2_schema_revision,
    }));
    assert.equal(rebasedSchemaPost.body.expected_memory_l2_schema_revision, 'schema-revision-2');
    assert.equal(Object.prototype.hasOwnProperty.call(rebasedSchemaPost.body, 'expected_memory_task_revision'), false);
    assert.deepEqual(rebasedSchemaPost.schema, localSchema);
    assert.equal(rebasedSchemaPost.schemaRevision, 'schema-revision-3');
    await schemaConflictPage.close();

    const schemaMismatchPage = await makePage(browser, 390);
    await schemaMismatchPage.locator('.mtm-schema').evaluate((node) => { node.open = true; });
    const mismatchSchema = { ...OUTPUT_SCHEMA, description: 'server readback mismatch' };
    const mismatchSchemaInput = schemaMismatchPage.locator('#mtm-memory_l2_output_schema');
    await mismatchSchemaInput.fill(JSON.stringify(mismatchSchema, null, 2));
    await schemaMismatchPage.evaluate(() => { window.keepSchemaOnPost = true; });
    await schemaMismatchPage.locator('[data-action="save-memory-task-models"]').click();
    await schemaMismatchPage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('Schema 读回与草稿不一致'));
    assert.equal(await mismatchSchemaInput.inputValue(), JSON.stringify(mismatchSchema, null, 2));
    assert.equal(await schemaMismatchPage.evaluate(() => JSON.parse(sessionStorage.getItem('ombre.memory-task-models.draft.v1')).output_schema_text),
      JSON.stringify(mismatchSchema, null, 2), 'a successful HTTP response without exact schema readback does not clear the draft');
    await schemaMismatchPage.close();

    const schemaFailurePage = await makePage(browser, 390);
    await schemaFailurePage.locator('.mtm-schema').evaluate((node) => { node.open = true; });
    const failedSchemaDraft = { ...OUTPUT_SCHEMA, description: 'keep after network failure' };
    await schemaFailurePage.locator('#mtm-memory_l2_output_schema').fill(JSON.stringify(failedSchemaDraft, null, 2));
    await schemaFailurePage.evaluate(() => { window.failPost = true; });
    await schemaFailurePage.locator('[data-action="save-memory-task-models"]').click();
    await schemaFailurePage.waitForFunction(() => document.querySelector('.mtm-status')?.textContent.includes('保存失败'));
    assert.equal(await schemaFailurePage.locator('#mtm-memory_l2_output_schema').inputValue(), JSON.stringify(failedSchemaDraft, null, 2));
    assert.equal(await schemaFailurePage.evaluate(() => JSON.parse(sessionStorage.getItem('ombre.memory-task-models.draft.v1')).output_schema_text),
      JSON.stringify(failedSchemaDraft, null, 2), 'schema text stays in the session draft after a failed save');
    await schemaFailurePage.close();
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
