const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage();
  try {
    await page.route('**/*', route => route.abort());
    await page.setContent('<div id="daily-chat-memory-message"></div><div id="daily-chat-memory-pending"></div><div id="daily-chat-memory-runs"></div>');
    await page.evaluate(() => {
      window.BASE = ''; window.getActiveTab = () => 'other'; window.confirm = () => true;
      window.esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
      window.escAttr = window.esc; window.jsString = value => String(value).replace(/'/g, "\\'");
      window.requests = [];
      const items = Array.from({length: 101}, (_, index) => ({id: 'page-' + index, status: 'pending', authority_revision: 1,
        preparation_status: index === 0 ? 'waiting_budget' : null,
        preparation_counts: {generated: 3, queued: 1, processed: 2, adopted: 0},
        candidate: {title: '候选' + index, content: '完整的建议正文' + index, original_excerpt: '真实原文依据', kind: 'key_event', confidence: 0.9}, display: {}}));
      const tasks = Array.from({length: 101}, (_, index) => ({task_id: 'action_source_' + index, phase: 'queued', action_id: 'action' + index, candidate_ids: []}));
      window.authFetch = async raw => {
        window.requests.push(raw);
        const url = new URL(raw, 'http://fixture.invalid'); const offset = Number(url.searchParams.get('offset') || 0);
        const source = url.pathname.includes('source-tasks');
        const rows = (source ? tasks : items).slice(offset, offset + 100);
        return {ok: true, status: 200, json: async () => ({status: 'ok', [source ? 'tasks' : 'items']: rows,
          total: 101, saved_total: 1007, has_more: offset === 0, next_offset: offset === 0 ? 100 : null})};
      };
    });
    await page.addScriptTag({content: fs.readFileSync(path.join(__dirname, '..', 'dashboard_assets', 'chat-memory.js'), 'utf8')});
    await page.evaluate(() => window.loadDailyChatMemoryPending());
    assert.equal(await page.locator('[data-candidate-id]').count(), 100);
    assert.match(await page.locator('[data-memory-pagination]').textContent(), /已保存候选共 1007 条/);
    const first = page.locator('[data-candidate-id="page-0"]');
    assert.match(await first.textContent(), /已保留，等待后续处理/);
    await first.getByRole('button', {name: '编辑', exact: true}).click();
    await first.locator('[data-field="content"]').fill('尚未保存的完整草稿，加载后不能丢。');
    await page.getByRole('button', {name: '加载更多候选', exact: true}).click();
    await page.waitForFunction(() => document.querySelectorAll('[data-candidate-id]').length === 101);
    assert.equal(await page.locator('[data-candidate-id="page-0"] [data-field="content"]').inputValue(), '尚未保存的完整草稿，加载后不能丢。');
    assert.equal(await page.locator('[data-memory-load-more]').count(), 0);
    await page.locator('#memory-source-tasks > summary').click();
    await page.getByRole('button', {name: '加载更多准备任务', exact: true}).click();
    await page.waitForFunction(() => document.querySelector('[data-source-task-notice]').textContent.includes('已加载 101 条'));
    assert.equal(await page.locator('[data-source-task-load-more]').count(), 0);
    assert(await page.evaluate(() => requests.some(url => url.includes('status=review_visible') && url.includes('offset=100'))));
    assert(await page.evaluate(() => requests.some(url => url.includes('source-tasks') && url.includes('offset=100'))));
    await page.evaluate(() => {
      const original = window.authFetch;
      window.retryBodies = [];
      window.authFetch = async (url, options) => {
        if (url.includes('/daily-chat-memory/runs')) return {ok: true, json: async () => ({runs: [
          {date: '2026-10-16', status: 'failed'}, {date: '2026-10-16', status: 'pending'},
          {date: '2026-10-17', status: 'partial', error_category: 'invalid_l1_document', progress: {state: 'stopped'}}]})};
        if (options && options.method === 'POST') {
          retryBodies.push(JSON.parse(options.body));
          return {ok: true, json: async () => ({status: 'pending', candidates: []})};
        }
        return original(url, options);
      };
    });
    await page.evaluate(() => window.loadDailyChatMemoryRuns());
    assert.equal(await page.locator('[data-memory-retry-date]').count(), 1);
    await page.locator('[data-memory-retry-date="2026-10-17"]').click();
    await page.waitForFunction(() => retryBodies.length === 1);
    const retry = await page.evaluate(() => retryBodies[0]);
    assert.equal(retry.date, '2026-10-17');
    assert.equal(retry.retry_failed_windows, true);
    assert.equal(retry.force, undefined);
    await page.locator('.chat-memory-runs-actions button').click();
    await page.waitForFunction(() => retryBodies.length === 2);
    assert.equal(await page.evaluate(() => retryBodies[1].retry_failed_windows), undefined);
    console.log('PASS candidate/source-task pagination, saved totals, retained draft badge and unsaved edit preservation');
    console.log('PASS explicit failed-date retry parameters; completed dates and normal scan preserved');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
