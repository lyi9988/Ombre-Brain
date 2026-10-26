// One isolated Review-card DOM check; no server, model, or production writes.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'dashboard_assets/chat-memory.js'), 'utf8');
const dashboard = fs.readFileSync(path.join(root, 'dashboard.html'), 'utf8');
const styles = (dashboard.match(/<style[^>]*>[\s\S]*?<\/style>/g) || []).join('\n');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => route.abort());
    await page.setContent(styles + '<div class="content"><div id="daily-chat-memory-message"></div><div id="daily-chat-memory-pending"></div></div>');
    await page.evaluate(() => {
      window.BASE = ''; window.getActiveTab = () => 'other'; window.confirm = () => true;
      window.esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
      window.escAttr = window.esc; window.jsString = value => String(value).replace(/'/g, "\\'");
      window.calls = [];
      const base = id => ({ id, status: 'pending', authority_revision: 1, candidate: {
        title: id, content: '待审核的后续建议，不是 L0 原文。', original_excerpt: '主人补充的真实依据。',
        kind: 'reflection', domain: [], tags: [], source_status: 'readable' }, display: {} });
      const prepared = base('followup-candidate');
      prepared.candidate.l3_followup = { memory_id: 'target-memory-7', target_title: '原有的那次经历', expected_revision: 7,
        kind: 'understanding', preparation_state: 'prepared', reason: '本次解释补上了具体适用条件。',
        changes: [{ field: 'statements[0].content', before: '当时的理解', after: '补充依据后的理解' },
          { field: 'statements[0].cognition', before: null, after: { epistemic_status: 'tentative', question: '<img src=x onerror="window.xss=true">' } }] };
      const preparing = base('followup-preparing');
      preparing.candidate.l3_followup = { memory_id: 'target-memory-8', target_title: '另一段原记忆', expected_revision: 2,
        kind: 'correction', preparation_state: 'queued', changes: [], reason: '' };
      const ordinary = base('ordinary'); ordinary.candidate.content = '普通候选原正文';
      window.items = [prepared, preparing, ordinary];
      window.authFetch = async (url, options) => {
        calls.push({ url: String(url), method: options?.method || 'GET' });
        if (options?.method && options.method !== 'GET') throw Error('unexpected write');
        const result = String(url).includes('model-io') ? { items: [], supported_stages: ['chat_l1', 'presentation_l2'], total: 0 }
          : String(url).includes('source-tasks') ? { tasks: [] } : { items: structuredClone(items) };
        return { ok: true, status: 200, json: async () => result };
      };
    });
    await page.addScriptTag({ content: source });
    await page.evaluate(() => loadDailyChatMemoryPending());
    const card = page.locator('[data-candidate-id="followup-candidate"]');
    const preview = card.locator('.chat-memory-followup-review');
    assert.match(await preview.textContent(), /修订原记忆：原有的那次经历 \/ 原版本 7/);
    assert.equal(await preview.locator('details').evaluate(element => element.open), true);
    assert.match(await preview.textContent(), /修订前当时的理解修订后补充依据后的理解/);
    assert.match(await preview.textContent(), /epistemic_status/);
    assert.match(await preview.textContent(), /tentative/);
    assert.equal(await preview.locator('input, textarea, img').count(), 0);
    assert.equal(await page.evaluate(() => Boolean(window.xss)), false);
    assert.match(await page.locator('[data-candidate-id="followup-preparing"] .chat-memory-followup-review').textContent(), /旧计划尚未采用/);
    const ordinary = page.locator('[data-candidate-id="ordinary"]');
    assert.equal(await ordinary.locator('.chat-memory-followup-review').count(), 0);
    await ordinary.getByRole('button', { name: '编辑', exact: true }).click();
    await ordinary.locator('[data-field="content"]').fill('普通候选仍然可编辑');
    assert.equal(await ordinary.locator('[data-field="content"]').inputValue(), '普通候选仍然可编辑');
    assert.equal(await card.locator('[data-model-io-toggle]').getAttribute('data-model-io-id'), 'target-memory-7');
    assert.equal(await ordinary.locator('[data-model-io-toggle]').getAttribute('data-model-io-id'), 'ordinary');
    await card.locator('[data-model-io-toggle]').click();
    const calls = await page.evaluate(() => window.calls);
    assert.ok(calls.some(call => call.url.includes('target-memory-7') && call.url.includes('model-io')));
    assert.ok(calls.every(call => call.method === 'GET'));
    assert.deepEqual(errors, []);
    assert.match(dashboard, /chat-memory\.js\?v=20261026-l3-followup-review-r1/);
    console.log(JSON.stringify({ passed: 1, browser: 'Edge', followupTargetAndRevision: true, localBeforeAfterReadOnly: true,
      objectValuesEscaped: true, preparingPlanNotAdopted: true, ordinaryEditingPreserved: true,
      modelIoUsesTargetMemory: true, actualWrites: 0, realModelCalls: 0 }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
