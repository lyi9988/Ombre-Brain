const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const script = fs.readFileSync(path.join(root, 'dashboard_assets/chat-memory.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'dashboard.html'), 'utf8');
const styles = (html.match(/<style[^>]*>[\s\S]*?<\/style>/g) || []).join('\n');
const memoryId = 'chat_memory_' + 'a'.repeat(32);
const hostileText = '<img src=x onerror="window.xss=true">';

function makeL1() {
  return {
    schema_version: 'memory-l1-v1',
    narrative: { title: '合成 L1', body: '待审核的完整 L1 正文。' },
    entities: [],
    scenes: [],
    statements: [{
      statement_id: 'statement-a',
      content: '保留的结构化语义。',
      subject_ref: { state: 'not_stated', value: null, evidence_refs: [] },
      classification: {},
      time: {},
      conditions: {},
      evidence_refs: [],
    }],
    retrieval: { keywords: ['合成'] },
    evidence_bindings: [],
    relations: [],
    attributes: [],
    generation: { origin: 'chat_l1', task_id: memoryId },
  };
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  try {
    await page.route('**/*', route => route.abort());
    await page.setContent(styles + '<div class="content"><div class="bucket-bulk-message" id="daily-chat-memory-message"></div><div id="daily-chat-memory-pending"></div></div>');
    await page.evaluate(({ memoryId, hostileText, l1Document }) => {
      window.BASE = '';
      window.getActiveTab = () => 'other';
      window.confirm = () => true;
      window.loadBuckets = () => {};
      window.esc = value => String(value ?? '').replace(/[&<>"']/g, char =>
        ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
      window.escAttr = window.esc;
      window.jsString = value => String(value).replace(/'/g, "\\'");
      window.modelIoRequests = [];
      window.modelIoReply = {
        status: 'ok',
        memory_id: memoryId,
        items: [
          {
            io_id: 'io-l1',
            stage: 'chat_l1',
            task_id: memoryId,
            attempt: 1,
            attempt_id: 'claim-lease-1',
            draft_revision: 3,
            source_revisions: { canonical: 'rev-17' },
            generation: { model: 'synthetic-model', route: 'local-fixture' },
            messages: [{ role: 'system', content: hostileText + '\n' + '完整输入'.repeat(80) }],
            completion_parameters: { max_tokens: 3200, reasoning_effort: 'low' },
            started_at: '2026-10-18T10:00:00Z',
            ended_at: '2026-10-18T10:00:01Z',
            status: 'failed',
            usage: { prompt_tokens: 12, completion_tokens: 9, reasoning_tokens: 4, total_tokens: 21 },
            finish_reason: 'length',
            raw_output: '{"html":"<script>window.xss=true</script>"}',
            validation: { valid: false, error: 'missing_applicability' },
          },
          {
            io_id: 'io-l2',
            stage: 'presentation_l2',
            task_id: memoryId,
            attempt: 'lease-token-2',
            attempt_id: 'claim-lease-2',
            status: 'completed',
            usage: { input_tokens: 30, output_tokens: 12, total_tokens: 42 },
            finish_reason: 'stop',
            messages: [{ role: 'user', content: '完整 L2 输入' }],
            raw_output: '完整 L2 可见输出。',
            validation: { valid: true, state: 'ready' },
          },
        ],
        has_more: true,
        next_offset: 2,
        capture_state: 'recorded',
        supported_stages: ['chat_l1', 'presentation_l2'],
      };
      window.sourceTasks = [{
        source_kind: 'chat_submission', task_id: memoryId, phase: 'prepared', candidate_ids: [memoryId],
      }];
      window.items = [
        {
          id: memoryId, status: 'pending', authority_revision: 1,
          candidate: { title: '合成 L1', kind: 'key_event', confidence: 0.7, l1_document: l1Document },
          display: {},
        },
        {
          id: '', status: 'pending', authority_revision: 1,
          candidate: { title: '旧候选', content: '没有模型记录入口' },
          display: {},
        },
      ];
      const reply = (data, status = 200) => ({
        ok: status < 400, status, json: async () => structuredClone(data),
      });
      window.authFetch = async url => {
        if (url.includes('/model-io')) {
          window.modelIoRequests.push(url);
          return reply(window.modelIoReply);
        }
        if (url.includes('/api/memory-authority/source-tasks')) return reply({ tasks: window.sourceTasks });
        if (url.includes('/api/daily-chat-memory/pending')) return reply({ items: window.items });
        return reply({ error: 'unexpected_request' }, 404);
      };
    }, { memoryId, hostileText, l1Document: makeL1() });
    await page.addScriptTag({ content: script });
    await page.evaluate(() => window.loadDailyChatMemoryPending());

    const reviewCard = page.locator('[data-candidate-id="' + memoryId + '"]');
    const reviewToggle = reviewCard.getByRole('button', { name: '模型输入与输出', exact: true });
    assert.equal(await reviewToggle.count(), 1, 'canonical chat memory Review item has an on-demand model I/O entry');
    assert.equal(await page.locator('[data-candidate-id=""] [data-model-io-toggle]').count(), 0,
      'items without an explicit authority ID do not get a guessed memory ID');
    assert.equal(await page.evaluate(() => modelIoRequests.length), 0, 'model I/O is not fetched on initial page load');

    await reviewToggle.click();
    const viewer = reviewCard.locator('.chat-memory-model-io-panel');
    await viewer.waitFor({ state: 'visible' });
    await page.waitForFunction(() => document.querySelectorAll('.chat-memory-model-io-record').length === 2);
    assert.match(await page.evaluate(() => modelIoRequests[0]), new RegExp('/memories/' + memoryId + '/model-io\\?limit=20&offset=0$'));
    assert.equal(await viewer.locator('[data-model-io-stage-section]').count(), 2);
    assert.equal(await viewer.locator('.chat-memory-model-io-record[open]').count(), 0,
      'large input and output remain collapsed until the attempt is opened');
    const l1Record = viewer.locator('[data-model-io-stage-section="chat_l1"] .chat-memory-model-io-record');
    const l2Record = viewer.locator('[data-model-io-stage-section="presentation_l2"] .chat-memory-model-io-record');
    assert.match(await l1Record.locator('summary').first().textContent(), /第 1 次尝试.*finish_reason length.*prompt 12.*completion 9/);
    assert.match(await l2Record.locator('summary').first().textContent(), /本轮调用/);
    assert.doesNotMatch(await l2Record.locator('summary').first().textContent(), /第 lease-token-2 次尝试/);
    await l1Record.locator('summary').first().click();
    assert.equal(await l1Record.locator('dd').filter({ hasText: 'claim-lease-1' }).count(), 1,
      'attempt_id is identified as a task call marker, not a physical provider attempt');
    const input = l1Record.locator('.chat-memory-model-io-message .chat-memory-model-io-content');
    assert.match(await input.textContent(), /<img src=x onerror="window\.xss=true">/);
    const output = l1Record.locator('.chat-memory-model-io-output');
    assert.equal(await output.textContent(), '{"html":"<script>window.xss=true</script>"}');
    assert.equal(await viewer.locator('img, script').count(), 0, 'untrusted input/output is rendered as text');
    assert.equal(await page.evaluate(() => Boolean(window.xss)), false);
    assert.match(await l1Record.locator('.chat-memory-model-io-validation').textContent(), /valid: false.*error: missing_applicability/);
    assert.equal(await l1Record.getByRole('button', { name: '复制完整输出' }).count(), 1);

    await reviewCard.getByRole('button', { name: '编辑', exact: true }).click();
    await reviewCard.locator('[data-l1-body]').fill('主人尚未保存的完整 L1 草稿。');
    await page.evaluate(() => {
      window.modelIoPanelBeforeRefresh = document.querySelector('.chat-memory-model-io-panel');
      window.modelIoRecordBeforeRefresh = document.querySelector('.chat-memory-model-io-record');
      return window.loadDailyChatMemoryPending();
    });
    const refreshedCard = page.locator('[data-candidate-id="' + memoryId + '"]');
    assert.equal(await refreshedCard.locator('[data-l1-body]').inputValue(), '主人尚未保存的完整 L1 草稿。',
      'review refresh preserves the complete L1 draft');
    assert.equal(await page.evaluate(() => window.modelIoPanelBeforeRefresh === document.querySelector('.chat-memory-model-io-panel')), true,
      'polling reuses the expanded model I/O DOM instead of rebuilding long text');
    assert.equal(await page.evaluate(() => window.modelIoRecordBeforeRefresh === document.querySelector('.chat-memory-model-io-record')), true);
    assert.equal(await page.evaluate(() => modelIoRequests.length), 1, 'polling does not re-fetch stored model input/output');
    assert.equal(await refreshedCard.locator('.chat-memory-model-io-panel').isVisible(), true);
    assert.equal(await refreshedCard.locator('.chat-memory-model-io-record[open]').count(), 1);

    await page.evaluate(() => {
      modelIoReply = {
        status: 'ok', memory_id: 'chat_memory_' + 'a'.repeat(32), items: [{
          io_id: 'io-l1-page-2', stage: 'chat_l1', task_id: 'chat_memory_' + 'a'.repeat(32),
          attempt: 0, status: 'completed', usage: { total_tokens: 1 }, messages: [], raw_output: '较早一轮输出。',
        }],
        has_more: false, next_offset: 3, capture_state: 'recorded',
        supported_stages: ['chat_l1', 'presentation_l2'],
      };
    });
    await refreshedCard.locator('[data-model-io-more]').click();
    await page.waitForFunction(() => document.querySelectorAll('.chat-memory-model-io-record').length === 3);
    assert.match(await page.evaluate(() => modelIoRequests[1]), /limit=20&offset=2$/);
    assert.equal(await page.evaluate(() => window.modelIoRecordBeforeRefresh === document.querySelector('.chat-memory-model-io-record')), true,
      'continuing older records appends without rebuilding already opened text');

    await page.evaluate(() => {
      modelIoReply = {
        status: 'ok', memory_id: 'chat_memory_' + 'a'.repeat(32), items: [],
        has_more: false, next_offset: 0, capture_state: 'missing',
        supported_stages: ['chat_l1', 'presentation_l2'],
      };
    });
    await refreshedCard.locator('[data-model-io-refresh]').click();
    await page.waitForFunction(() => document.querySelectorAll('.chat-memory-model-io-record').length === 0);
    assert.equal(await page.evaluate(() => modelIoRequests.length), 3);
    assert.equal(await refreshedCard.locator('.chat-memory-model-io-empty').count(), 2);
    assert.match(await refreshedCard.locator('.chat-memory-model-io-empty').first().textContent(),
      /此阶段尚无已留存的调用记录；旧输出无法补回/);
    assert.doesNotMatch(await viewer.locator('[data-model-io-status]').textContent(), /失败/,
      'missing historical capture is not reported as a generation failure');

    await page.locator('#memory-source-tasks > summary').click();
    const taskCard = page.locator('[data-chat-submission-task-id="' + memoryId + '"]');
    await taskCard.waitFor({ state: 'visible' });
    const taskToggle = taskCard.getByRole('button', { name: '模型输入与输出', exact: true });
    await taskToggle.click();
    await page.waitForFunction(() => document.querySelectorAll('[data-chat-submission-task-id] .chat-memory-model-io-panel').length === 1);
    assert.match(await page.evaluate(() => modelIoRequests[3]), new RegExp('/memories/' + memoryId + '/model-io\\?limit=20&offset=0$'),
      'source preparation task uses its canonical chat memory task ID');
    assert.equal(await taskCard.locator('.chat-memory-model-io-panel').isVisible(), true);

    console.log('PASS memory model I/O viewer contract, safe text rendering, pagination, and draft preservation');
  } finally {
    await page.close();
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
