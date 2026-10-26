// Synthetic browser coverage for persistent chat-submission tasks.
// Uses a local in-process HTTP fixture only: no provider, model, or production write.
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const asset = fs.readFileSync(path.join(root, 'dashboard_assets/chat-memory.js'), 'utf8');
const dashboard = fs.readFileSync(path.join(root, 'dashboard.html'), 'utf8');
const styles = (dashboard.match(/<style[^>]*>[\s\S]*?<\/style>/gi) || []).join('\n');
const CHAT_A = 'chat_memory_' + 'a'.repeat(32);
const CHAT_B = 'chat_memory_' + 'b'.repeat(32);
const CHAT_C = 'chat_memory_' + 'c'.repeat(32);
const CHAT_D = 'chat_memory_' + 'd'.repeat(32);
const BROWSE_A = 'memory_' + 'e'.repeat(16);
const BROWSE_B = 'memory_' + 'f'.repeat(16);

function copy(value) { return JSON.parse(JSON.stringify(value)); }

function makeTask(taskId, phase, revision, draftRevision, more = {}) {
  return Object.assign({ source_kind: 'chat_submission', task_id: taskId, phase,
    revision, draft_revision: draftRevision, candidate_ids: [], issue: null }, more);
}

function makeDetail(task, body, title = '合成标题', hints = []) {
  return { task: copy(task), draft: { narrative: { title, body }, semantic_hints: copy(hints) },
    history: [{ phase: task.phase, revision: task.revision, draft_revision: task.draft_revision,
      created_at: '2026-10-16T10:00:00Z' }],
    receipt: { memory_status: task.phase === 'prepared' ? 'pending' : 'absent',
      preparation_status: task.phase } };
}

function makeBrowseDetail(task, body, title) {
  const detail = makeDetail(task, body, title);
  detail.draft = { content: body, metadata: { name: title, tags: ['browse_note'] },
    descriptor: { schema: 'aiz.memory-source-ref-v1', namespace: 'snapshot.aiz.browse.summary',
      source_id: 'synthetic-browse-item', source_revision: 'synthetic-source-revision',
      source_scope: { source_kind: 'web_browse', source_event_id: 'synthetic-browse-event' } } };
  return detail;
}

function fixtureState() {
  const action = { task_id: 'action/42', action_id: 'act-42', phase: 'failed',
    issue: { code: 'synthetic_action_failure' }, candidate_ids: ['action-candidate-42'] };
  const tasks = [action,
    makeTask(CHAT_A, 'queued', 1, 1),
    makeTask(CHAT_B, 'generating', 2, 1),
    makeTask(CHAT_C, 'failed', 3, 2, { issue: { code: 'source_capture_failed', field_path: 'source.capture' } }),
    makeTask(CHAT_D, 'prepared', 4, 2, { candidate_ids: ['review-candidate-42'] }),
    makeTask(BROWSE_A, 'failed', 1, 1, { source_kind: 'web_browse', issue: { code: 'synthetic_browse_failure' } }),
    makeTask(BROWSE_B, 'prepared', 5, 2, { source_kind: 'web_browse', candidate_ids: [BROWSE_B] })];
  const details = new Map([
    [CHAT_A, makeDetail(tasks[1], '原始聊天提交正文', '原标题', [{ text: '原始提示' }])],
    [CHAT_B, makeDetail(tasks[2], '处理中提交正文')],
    [CHAT_C, makeDetail(tasks[3], '失败任务仍保留的提交正文')],
    [CHAT_D, makeDetail(tasks[4], '已经生成候选的提交正文')],
    [BROWSE_A, makeBrowseDetail(tasks[5], '浏览后保留的完整提交正文', '浏览原标题')],
    [BROWSE_B, makeBrowseDetail(tasks[6], '已经完成准备的浏览提交正文', '浏览已准备标题')],
  ]);
  details.get(BROWSE_B).receipt.memory_status = 'committed';
  return { tasks, details, puts: [], retries: [], gets: [], failList: false, failDetails: new Set(),
    conflictNext: false, dropNextPutReply: false, prepareNextPut: false, reordered: false };
}

function syncTaskFromDetail(state, taskId) {
  const detail = state.details.get(taskId);
  const row = state.tasks.find(task => task.task_id === taskId);
  if (detail && row) Object.assign(row, copy(detail.task));
}

function sendJson(response, status, payload) {
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  response.end(JSON.stringify(payload));
}

function startFixtureServer(state) {
  const html = `<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"></head><body>
    ${styles}<main class="content"><div class="bucket-bulk-message" id="daily-chat-memory-message"></div>
    <div id="daily-chat-memory-pending"></div></main>
    <script>
      var BASE = location.origin;
      window.getActiveTab = function() { return 'other'; };
      window.confirm = function() { return true; };
      window.esc = function(value) { return String(value == null ? '' : value).replace(/[&<>\"']/g, function(ch) { return ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[ch]); }); };
      window.escAttr = window.esc;
      window.jsString = function(value) { return String(value).replace(/'/g, "\\\\'"); };
      window.authRequests = [];
      window.authFetch = async function(url, options) {
        window.authRequests.push({url: String(url), method: options && options.method || 'GET'});
        return fetch(new URL(url, BASE).toString(), options);
      };
    </script><script src="/dashboard-assets/chat-memory.js"></script></body></html>`;
  const server = http.createServer(async (request, response) => {
    const url = new URL(request.url, 'http://localhost');
    if (request.method === 'GET' && url.pathname === '/') {
      response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }); response.end(html); return;
    }
    if (request.method === 'GET' && url.pathname === '/dashboard-assets/chat-memory.js') {
      response.writeHead(200, { 'Content-Type': 'text/javascript; charset=utf-8' }); response.end(asset); return;
    }
    if (request.method === 'GET' && url.pathname === '/api/daily-chat-memory/pending') {
      sendJson(response, 200, { items: [] }); return;
    }
    if (url.pathname === '/api/memory-authority/source-tasks' && request.method === 'GET') {
      if (state.failList) { sendJson(response, 503, { error: 'synthetic_list_offline' }); return; }
      state.gets.push({ kind: 'list' });
      let tasks = state.tasks.map(copy);
      if (state.reordered) tasks = [tasks.find(task => task.task_id === CHAT_A), ...tasks.filter(task => task.task_id !== CHAT_A)];
      sendJson(response, 200, { tasks }); return;
    }
    const retryMatch = url.pathname.match(/^\/api\/memory-authority\/source-tasks\/(.+)\/retry$/);
    if (retryMatch && request.method === 'POST') {
      const taskId = decodeURIComponent(retryMatch[1]);
      const body = await readJson(request);
      state.retries.push({ taskId, body });
      const task = state.tasks.find(entry => entry.task_id === taskId);
      if (!task || task.phase !== body.expected_phase) { sendJson(response, 409, { error: 'task_phase_changed' }); return; }
      task.phase = 'queued'; task.revision += 1; task.issue = null;
      const detail = state.details.get(taskId);
      if (detail) { detail.task = copy(task); detail.receipt.preparation_status = 'queued'; }
      sendJson(response, 200, { task: copy(task), status: 'queued', action_reexecuted: false }); return;
    }
    const detailMatch = url.pathname.match(/^\/api\/memory-authority\/source-tasks\/(.+)$/);
    if (detailMatch) {
      const taskId = decodeURIComponent(detailMatch[1]);
      if (state.failDetails.has(taskId)) { sendJson(response, 503, { error: 'synthetic_detail_offline' }); return; }
      const detail = state.details.get(taskId);
      if (!detail) { sendJson(response, 404, { error: 'source_task_not_found' }); return; }
      if (request.method === 'GET') {
        state.gets.push({ kind: 'detail', taskId }); sendJson(response, 200, copy(detail)); return;
      }
      if (request.method === 'PUT') {
        const body = await readJson(request);
        state.puts.push({ taskId, body: copy(body) });
        if (detail.task.source_kind === 'web_browse' &&
            (Object.keys(body.draft).join(',') !== 'narrative' || !body.draft.narrative?.body)) {
          sendJson(response, 400, { error: 'invalid_browse_submission_edit' }); return;
        }
        if (state.prepareNextPut) {
          state.prepareNextPut = false;
          detail.task.revision += 1; detail.task.phase = 'prepared'; detail.task.issue = null;
          detail.task.candidate_ids = ['review-candidate-race'];
          detail.receipt.preparation_status = 'prepared'; detail.receipt.memory_status = 'pending';
          syncTaskFromDetail(state, taskId);
          sendJson(response, 409, { error: 'edit_prepared_candidate_in_review', task: copy(detail.task) }); return;
        }
        if (state.conflictNext) {
          state.conflictNext = false;
          detail.task.revision += 1; detail.task.draft_revision += 1;
          if (detail.task.source_kind === 'web_browse') {
            detail.draft.content = '409 时服务器已保存的浏览版本';
            detail.draft.metadata.name = '服务器浏览冲突标题';
          } else detail.draft = { narrative: { title: '服务器冲突标题', body: '409 时服务器已保存的版本' } };
          detail.receipt.preparation_status = 'queued'; detail.receipt.memory_status = 'absent';
          syncTaskFromDetail(state, taskId);
          sendJson(response, 409, { error: 'revision_conflict', task: copy(detail.task) }); return;
        }
        if (String(body.expected_revision) !== String(detail.task.revision)) {
          sendJson(response, 409, { error: 'revision_conflict', task: copy(detail.task) }); return;
        }
        detail.task.revision += 1; detail.task.draft_revision += 1; detail.task.phase = 'queued';
        detail.task.issue = null; detail.task.candidate_ids = [];
        if (detail.task.source_kind === 'web_browse') {
          detail.draft.content = body.draft.narrative.body;
          detail.draft.metadata.name = body.draft.narrative.title || '';
          detail.draft.draft_arguments = copy(body.draft);
        } else detail.draft = copy(body.draft);
        detail.receipt.preparation_status = 'queued'; detail.receipt.memory_status = 'absent';
        detail.history.push({ phase: 'queued', revision: detail.task.revision,
          draft_revision: detail.task.draft_revision, created_at: '2026-10-16T10:01:00Z' });
        syncTaskFromDetail(state, taskId);
        state.reordered = true;
        if (state.dropNextPutReply) {
          state.dropNextPutReply = false;
          sendJson(response, 503, { error: 'synthetic_lost_save_ack' }); return;
        }
        sendJson(response, 200, copy(detail)); return;
      }
    }
    sendJson(response, 404, { error: 'unexpected_synthetic_route' });
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve({ server,
    url: `http://127.0.0.1:${server.address().port}/` })));
}

function readJson(request) {
  return new Promise((resolve, reject) => {
    let raw = '';
    request.setEncoding('utf8');
    request.on('data', chunk => { raw += chunk; });
    request.on('end', () => { try { resolve(raw ? JSON.parse(raw) : {}); } catch (error) { reject(error); } });
    request.on('error', reject);
  });
}

(async () => {
  const state = fixtureState();
  const { server, url } = await startFixtureServer(state);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  let passed = 0;
  const screenshotDir = process.env.CHAT_SUBMISSION_SCREENSHOT_DIR;
  if (screenshotDir) fs.mkdirSync(screenshotDir, { recursive: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('dialog', dialog => dialog.accept());

  async function openPanel() {
    await page.locator('#memory-source-tasks').evaluate(node => { node.open = true; });
  }
  async function openTask(taskId) {
    await openPanel();
    const row = page.locator(`[data-chat-submission-task-id="${taskId}"]`);
    await row.locator('button', { hasText: /查看|收起/ }).first().click();
    await page.waitForFunction(id => Boolean(document.querySelector(`[data-chat-submission-detail="${id}"]`)), taskId);
    return row;
  }
  async function test(name, run) { await run(); passed++; console.log('PASS', name); }

  try {
    await page.goto(url);
    await page.evaluate(() => window.loadDailyChatMemoryPending());
    await page.waitForFunction(() => Boolean(document.querySelector('#memory-source-tasks [data-source-task-list]')));
    await page.waitForFunction(() => document.querySelector('#memory-source-tasks')?.textContent.includes('准备中'));
    await openPanel();

    await test('mixed source tasks keep action rows while exposing chat phases, reasons, and Review status', async () => {
      const text = await page.locator('#memory-source-tasks').textContent();
      assert.match(text, /等待准备/);
      assert.match(text, /准备中/);
      assert.match(text, /准备失败/);
      assert.match(text, /source_capture_failed/);
      assert.match(text, /source\.capture/);
      assert.match(text, /候选已准备 · 待 Review；尚未正式入库/);
      assert.match(text, /act-42/);
      assert.match(text, /synthetic_action_failure/);
      assert.equal(await page.locator('#memory-source-tasks [data-chat-submission-task-id]').count(), 6);
      assert.equal(await page.locator('#memory-source-tasks button', { hasText: '重试整理' }).count(), 1,
        'the existing action retry stays visible');
      assert.equal(await page.locator(`#memory-source-tasks [data-chat-submission-task-id="${CHAT_C}"] button`, { hasText: '重试准备' }).count(), 1);
      if (screenshotDir) await page.screenshot({ path: path.join(screenshotDir, 'chat-submission-1280.png'), fullPage: true });
    });

    await test('prepared chat tasks show the saved draft and point to existing Review L1 editing', async () => {
      const row = await openTask(CHAT_D);
      const detail = row.locator('[data-chat-submission-detail]');
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"]`)?.textContent.includes('正式入库回执：Review 待审（尚未正式入库）'), CHAT_D);
      assert.match(await detail.textContent(), /下方现有 Review 候选卡片/);
      assert.match(await detail.textContent(), /正式记忆只有在 Review 确认后才入库/);
      assert.match(await detail.textContent(), /已经生成候选的提交正文/);
      await detail.locator('summary', { hasText: '提交历史' }).click();
      assert.match(await detail.textContent(), /draft revision 2/);
      assert.equal(await detail.locator('[data-chat-submission-editor]').count(), 0,
        'prepared drafts are read only here');
      assert.equal(await detail.locator('[data-chat-submission-save]').count(), 0);
      if (screenshotDir) await page.screenshot({ path: path.join(screenshotDir, 'chat-submission-prepared-1280.png'), fullPage: true });
      await row.locator('button', { hasText: '收起提交稿' }).click();
    });

    await test('browse envelope opens in the shared editor, saves narrative only, and reads back without changing provenance', async () => {
      const row = await openTask(BROWSE_A);
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"] [data-chat-submission-body]`)?.value === '浏览后保留的完整提交正文', BROWSE_A);
      const detail = row.locator('[data-chat-submission-detail]');
      assert.match(await row.textContent(), /浏览记忆提交/);
      assert.equal(await detail.locator('[data-chat-submission-title]').inputValue(), '浏览原标题');
      assert.equal(await detail.locator('[data-chat-submission-hints]').count(), 0);
      assert.match(await detail.locator('[data-browse-submission-source]').textContent(), /synthetic-browse-event/);
      assert.equal(await detail.locator('[data-browse-submission-source] input, [data-browse-submission-source] textarea').count(), 0);
      const descriptor = copy(state.details.get(BROWSE_A).draft.descriptor);
      await detail.locator('[data-chat-submission-title]').fill('编辑后的浏览标题');
      await detail.locator('[data-chat-submission-body]').fill('编辑后的浏览完整正文，保留主体和未知。');
      await detail.locator('button', { hasText: '读取服务器最新草稿' }).click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"] [data-chat-submission-notice]`)?.textContent.includes('当前未保存输入保持不变'), BROWSE_A);
      assert.equal(await detail.locator('[data-chat-submission-body]').inputValue(), '编辑后的浏览完整正文，保留主体和未知。');
      await detail.locator('[data-chat-submission-save]').click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"] [data-chat-submission-notice]`)?.textContent.includes('已保存并从服务器读回'), BROWSE_A);
      assert.equal(state.puts.at(-1).taskId, BROWSE_A);
      assert.deepEqual(state.puts.at(-1).body.draft, { narrative: {
        title: '编辑后的浏览标题', body: '编辑后的浏览完整正文，保留主体和未知。' } });
      assert.equal(state.puts.at(-1).body.expected_revision, 1);
      assert.equal(state.details.get(BROWSE_A).task.draft_revision, 2);
      assert.equal(state.details.get(BROWSE_A).draft.content, '编辑后的浏览完整正文，保留主体和未知。');
      assert.deepEqual(state.details.get(BROWSE_A).draft.descriptor, descriptor);
      assert.deepEqual(state.details.get(BROWSE_A).draft.metadata.tags, ['browse_note']);
      assert.ok(state.gets.some(entry => entry.kind === 'detail' && entry.taskId === BROWSE_A));
    });

    await test('browse conflict, session reload, and lost acknowledgement preserve the same editable draft', async () => {
      let row = page.locator(`[data-chat-submission-task-id="${BROWSE_A}"]`);
      await row.locator('[data-chat-submission-body]').fill('浏览本地未保存稿，跨刷新与冲突保留。');
      await page.locator('#memory-source-tasks button', { hasText: '刷新准备状态' }).click();
      await page.waitForFunction(() => !document.querySelector('#memory-source-tasks [role="status"]')?.textContent.includes('正在刷新'));
      assert.equal(await row.locator('[data-chat-submission-body]').inputValue(), '浏览本地未保存稿，跨刷新与冲突保留。');
      await page.reload();
      await page.evaluate(() => window.loadDailyChatMemoryPending());
      await page.waitForFunction(() => Boolean(document.querySelector('#memory-source-tasks [data-source-task-list]')));
      row = await openTask(BROWSE_A);
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"] [data-chat-submission-body]`)?.value === '浏览本地未保存稿，跨刷新与冲突保留。', BROWSE_A);
      assert.equal(await row.locator('[data-chat-submission-hints]').count(), 0);
      state.conflictNext = true;
      await row.locator('[data-chat-submission-save]').click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"]`)?.textContent.includes('409 时服务器已保存的浏览版本'), BROWSE_A);
      assert.equal(await row.locator('[data-chat-submission-body]').inputValue(), '浏览本地未保存稿，跨刷新与冲突保留。');
      assert.equal(await row.locator('[data-chat-submission-save]').isDisabled(), true);
      await row.locator('button', { hasText: '保留当前编辑并以服务器版本为基线' }).click();
      state.dropNextPutReply = true;
      await row.locator('[data-chat-submission-save]').click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"] [data-chat-submission-notice]`)?.textContent.includes('保存响应未确认，但服务器读回已证明'), BROWSE_A);
      assert.equal(state.puts.at(-1).body.expected_revision, 3);
      assert.deepEqual(Object.keys(state.puts.at(-1).body.draft), ['narrative']);
      assert.equal(state.details.get(BROWSE_A).draft.content, '浏览本地未保存稿，跨刷新与冲突保留。');
      assert.equal(await row.locator('[data-chat-submission-save]').isDisabled(), true);
    });

    await test('prepared browse drafts retain their body and reflect committed receipt without inventing Review pending', async () => {
      const row = await openTask(BROWSE_B);
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"]`)?.textContent.includes('正式入库回执：已正式入库'), BROWSE_B);
      assert.match(await row.textContent(), /已经完成准备的浏览提交正文/);
      assert.doesNotMatch(await row.textContent(), /待 Review；尚未正式入库|正式记忆只有在 Review 确认后才入库/);
      assert.equal(await row.locator('[data-chat-submission-editor]').count(), 0);
      assert.equal(await row.locator('[data-chat-submission-hints]').count(), 0);
      assert.match(await row.locator('[data-browse-submission-source]').textContent(), /synthetic-browse-item/);
      await row.locator('button', { hasText: '收起提交稿' }).click();
    });

    await test('detail refresh preserves edits; CAS save reads back the new persistent revision and reorders the task', async () => {
      const row = await openTask(CHAT_A);
      const detail = row.locator('[data-chat-submission-detail]');
      const title = detail.locator('[data-chat-submission-title]');
      const body = detail.locator('[data-chat-submission-body]');
      const hints = detail.locator('[data-chat-submission-hints]');
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"] [data-chat-submission-body]`)?.value === '原始聊天提交正文', CHAT_A);
      await title.fill('编辑后的标题');
      await body.fill('编辑后的完整提交正文，包含保留下来的上下文。');
      await hints.fill('[{"text":"主人明确补充的语义"}]');
      await detail.locator('button', { hasText: '读取服务器最新草稿' }).click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"] [data-chat-submission-notice]`)?.textContent.includes('当前未保存输入保持不变'), CHAT_A);
      assert.equal(await title.inputValue(), '编辑后的标题');
      assert.equal(await body.inputValue(), '编辑后的完整提交正文，包含保留下来的上下文。');
      assert.equal(await hints.inputValue(), '[{"text":"主人明确补充的语义"}]');

      await detail.locator('[data-chat-submission-save]').click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"] [data-chat-submission-notice]`)?.textContent.includes('已保存并从服务器读回'), CHAT_A);
      const put = state.puts.at(-1);
      assert.equal(put.taskId, CHAT_A);
      assert.equal(put.body.expected_revision, 1);
      assert.match(put.body.request_id, /^rq_/);
      assert.deepEqual(put.body.draft, { narrative: { body: '编辑后的完整提交正文，包含保留下来的上下文。', title: '编辑后的标题' },
        semantic_hints: [{ text: '主人明确补充的语义' }] });
      assert.equal(state.details.get(CHAT_A).task.draft_revision, 2);
      await page.waitForFunction(() => document.querySelector('#memory-source-tasks [data-chat-submission-task-id]')?.getAttribute('data-chat-submission-task-id') === 'chat_memory_' + 'a'.repeat(32));
      assert.ok(state.gets.some(entry => entry.kind === 'detail' && entry.taskId === CHAT_A), 'saved draft is read back through the detail API');
    });

    await test('a newer server draft and a 409 preserve local text until the owner explicitly rebases', async () => {
      let row = page.locator(`[data-chat-submission-task-id="${CHAT_A}"]`);
      let detail = row.locator('[data-chat-submission-detail]');
      let body = detail.locator('[data-chat-submission-body]');
      await body.fill('本地草稿，在列表刷新与版本冲突时保留。');
      const serverDetail = state.details.get(CHAT_A);
      serverDetail.task.revision += 1; serverDetail.task.draft_revision += 1;
      serverDetail.draft = { narrative: { title: '服务器新标题', body: '服务器较新的完整提交稿' } };
      syncTaskFromDetail(state, CHAT_A);
      await detail.locator('button', { hasText: '读取服务器最新草稿' }).click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"]`)?.textContent.includes('服务器较新的完整提交稿'), CHAT_A);
      row = page.locator(`[data-chat-submission-task-id="${CHAT_A}"]`);
      detail = row.locator('[data-chat-submission-detail]'); body = detail.locator('[data-chat-submission-body]');
      assert.equal(await body.inputValue(), '本地草稿，在列表刷新与版本冲突时保留。');
      assert.equal(await detail.locator('[data-chat-submission-save]').isDisabled(), true, 'stale CAS is blocked until explicit rebase');
      await detail.locator('button', { hasText: '保留当前编辑并以服务器版本为基线' }).click();
      assert.equal(await detail.locator('[data-chat-submission-body]').inputValue(), '本地草稿，在列表刷新与版本冲突时保留。');
      await detail.locator('[data-chat-submission-save]').click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"] [data-chat-submission-notice]`)?.textContent.includes('已保存并从服务器读回'), CHAT_A);
      assert.equal(state.puts.at(-1).body.expected_revision, 3);

      await detail.locator('[data-chat-submission-body]').fill('本地草稿，服务端返回 409 后保留。');
      state.conflictNext = true;
      await detail.locator('[data-chat-submission-save]').click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"] [data-chat-submission-body]`)?.value === '本地草稿，服务端返回 409 后保留。' && document.querySelector(`[data-chat-submission-task-id="${id}"]`)?.textContent.includes('409 时服务器已保存的版本'), CHAT_A);
      row = page.locator(`[data-chat-submission-task-id="${CHAT_A}"]`);
      detail = row.locator('[data-chat-submission-detail]');
      assert.equal(await detail.locator('[data-chat-submission-save]').isDisabled(), true, '409 does not permit resubmitting the stale revision');
      await detail.locator('button', { hasText: '保留当前编辑并以服务器版本为基线' }).click();
      await detail.locator('[data-chat-submission-save]').click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"] [data-chat-submission-notice]`)?.textContent.includes('已保存并从服务器读回'), CHAT_A);
      assert.equal(state.puts.at(-1).body.expected_revision, 5);
      assert.equal(state.details.get(CHAT_A).draft.narrative.body, '本地草稿，服务端返回 409 后保留。');
    });

    await test('a lost save acknowledgement is resolved by reading the new persistent draft revision', async () => {
      const row = page.locator(`[data-chat-submission-task-id="${CHAT_A}"]`);
      const detail = row.locator('[data-chat-submission-detail]');
      await detail.locator('[data-chat-submission-title]').fill('回执丢失但已读回');
      await detail.locator('[data-chat-submission-body]').fill('服务器已持久保存的完整聊天正文。');
      state.dropNextPutReply = true;
      await detail.locator('[data-chat-submission-save]').click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"] [data-chat-submission-notice]`)?.textContent.includes('保存响应未确认，但服务器读回已证明'), CHAT_A);
      assert.equal(await row.locator('[data-chat-submission-body]').inputValue(), '服务器已持久保存的完整聊天正文。');
      assert.equal(await row.locator('[data-chat-submission-title]').inputValue(), '回执丢失但已读回');
      assert.equal(state.details.get(CHAT_A).draft.narrative.body, '服务器已持久保存的完整聊天正文。');
      assert.equal(await row.locator('[data-chat-submission-save]').isDisabled(), true, 'confirmed server readback clears the local dirty state');
    });

    await test('prepared-candidate 409 keeps local edits visible and directs complete editing to Review', async () => {
      const row = await openTask(CHAT_B);
      const detail = row.locator('[data-chat-submission-detail]');
      await detail.locator('[data-chat-submission-body]').fill('准备完成竞态时仍要保留的本地输入。');
      await detail.locator('[data-chat-submission-hints]').fill('{"text":"不是数组"}');
      assert.equal(await detail.locator('[data-chat-submission-save]').isDisabled(), true, 'semantic hints require a JSON array');
      assert.match(await detail.locator('[data-chat-submission-error]').textContent(), /必须是 JSON 数组/);
      await detail.locator('[data-chat-submission-hints]').fill('[{"text":"有效提示"}]');
      assert.equal(await detail.locator('[data-chat-submission-save]').isDisabled(), false);
      state.prepareNextPut = true;
      await detail.locator('[data-chat-submission-save]').click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"]`)?.textContent.includes('可复制以下修改内容，再到 Review 候选卡片编辑'), CHAT_B);
      const refreshed = page.locator(`[data-chat-submission-task-id="${CHAT_B}"] [data-chat-submission-detail]`);
      assert.match(await refreshed.textContent(), /处理中提交正文/);
      assert.match(await refreshed.textContent(), /准备完成竞态时仍要保留的本地输入/);
      assert.match(await refreshed.textContent(), /下方现有 Review 候选卡片/);
      assert.equal(await refreshed.locator('[data-chat-submission-editor]').count(), 0);
      assert.equal(state.details.get(CHAT_B).task.phase, 'prepared');
    });

    await test('session refresh, detail errors, and list errors do not replace unsaved chat text', async () => {
      const row = page.locator(`[data-chat-submission-task-id="${CHAT_A}"]`);
      await row.locator('[data-chat-submission-body]').fill('跨页面刷新仍需保留的本地输入。');
      await page.reload();
      await page.evaluate(() => window.loadDailyChatMemoryPending());
      await page.waitForFunction(() => Boolean(document.querySelector('#memory-source-tasks [data-source-task-list]')));
      const refreshedRow = await openTask(CHAT_A);
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"] [data-chat-submission-body]`)?.value === '跨页面刷新仍需保留的本地输入。', CHAT_A);
      assert.equal(await refreshedRow.locator('[data-chat-submission-body]').inputValue(), '跨页面刷新仍需保留的本地输入。');

      state.failDetails.add(CHAT_A);
      await refreshedRow.locator('button', { hasText: '读取服务器最新草稿' }).click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"]`)?.textContent.includes('synthetic_detail_offline'), CHAT_A);
      assert.equal(await page.locator(`[data-chat-submission-task-id="${CHAT_A}"] [data-chat-submission-body]`).inputValue(), '跨页面刷新仍需保留的本地输入。');
      state.failDetails.delete(CHAT_A);

      state.failList = true;
      await page.locator('#memory-source-tasks button', { hasText: '刷新准备状态' }).click();
      await page.waitForFunction(() => document.querySelector('#memory-source-tasks [role="status"]')?.textContent.includes('可能已过期'));
      assert.equal(await page.locator(`[data-chat-submission-task-id="${CHAT_A}"] [data-chat-submission-body]`).inputValue(), '跨页面刷新仍需保留的本地输入。');
      assert.equal(await page.evaluate(() => authRequests.some(call => call.url.endsWith('/' + 'chat_memory_' + 'a'.repeat(32)) && call.method === 'GET')), true,
        'detail reads continue through the Dashboard authFetch session after reload');
      state.failList = false;

      await page.setViewportSize({ width: 390, height: 844 });
      await page.locator('#memory-source-tasks').evaluate(node => { node.open = true; });
      const overflow = await page.evaluate(() => ({ width: innerWidth, scrollWidth: document.documentElement.scrollWidth }));
      assert.ok(overflow.scrollWidth <= overflow.width + 1, '390px task and editor layout fits: ' + JSON.stringify(overflow));
      if (screenshotDir) await page.screenshot({ path: path.join(screenshotDir, 'chat-submission-390.png'), fullPage: true });
    });

    await test('failed chat retry uses expected_phase and does not alter the existing action retry', async () => {
      await openPanel();
      const failed = page.locator(`[data-chat-submission-task-id="${CHAT_C}"]`);
      await failed.locator('button', { hasText: '重试准备' }).click();
      await page.waitForFunction(id => document.querySelector(`[data-chat-submission-task-id="${id}"]`)?.textContent.includes('等待准备'), CHAT_C);
      assert.deepEqual(state.retries.at(-1), { taskId: CHAT_C, body: { expected_phase: 'failed' } });
      assert.equal(await page.locator('#memory-source-tasks button', { hasText: '重试整理' }).count(), 1);
      assert.equal(await page.locator(`[data-chat-submission-task-id="${CHAT_C}"] button`, { hasText: '重试准备' }).count(), 0);
    });

    assert.deepEqual(pageErrors, [], 'browser fixture has no uncaught exceptions');
    console.log(JSON.stringify({ passed, browser: 'Microsoft Edge headless', viewport_widths: [1280, 390],
      synthetic_gets: state.gets.length, synthetic_puts: state.puts.length, synthetic_retries: state.retries.length,
      provider_calls: 0, production_writes: 0,
      screenshots: screenshotDir ? ['chat-submission-1280.png', 'chat-submission-prepared-1280.png', 'chat-submission-390.png'] : [] }));
  } finally {
    await page.close();
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
