// Actual dashboard edit functions, headless Edge, synthetic HTTP responses.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const html = fs.readFileSync(path.join(__dirname, '../dashboard.html'), 'utf8');
const names = ['bucketEditArguments', 'bucketEditError', 'submitBucketDateEdit', 'submitBucketTitleEdit', 'submitBucketContentEdit'];
const code = names.map(name => {
  const match = html.match(new RegExp('^(?:async )?function ' + name + '\\([^]*?^}', 'm'));
  assert(match, name);
  return match[0];
}).join('\n');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    for (const width of [390, 1280]) {
      const page = await browser.newPage({ viewport: { width, height: 844 } });
      await page.setContent('<div id="detail-panel" data-bucket-id="memory-a" data-bucket-revision="1"></div>' +
        '<form id="edit"><textarea name="content">仍在编辑的完整草稿</textarea><input name="title" value="新标题">' +
        '<input name="date" value="2026-09-01"><button type="submit">保存</button></form>');
      await page.evaluate(() => {
        window.BASE = '';
        window.calls = [];
        window.alerts = [];
        window.alert = text => alerts.push(text);
        window.loadBuckets = async () => {};
        window.showDetail = async () => { window.reloads = (window.reloads || 0) + 1; };
        window.statusCode = 409;
        window.authFetch = async (url, options) => {
          calls.push({ url, body: JSON.parse(options.body) });
          if (window.disconnect) throw new Error('断线');
          return { ok: statusCode === 200, status: statusCode,
            json: async () => statusCode === 409 ? { error: 'revision_conflict', current_revision: 2 } : { revision: 2 } };
        };
      });
      await page.addScriptTag({ content: code });
      await page.evaluate(async () => {
        const form = document.getElementById('edit');
        await submitBucketContentEdit({ preventDefault() {}, target: form }, 'memory-a');
      });
      let result = await page.evaluate(() => ({ calls, alerts, draft: document.querySelector('textarea').value,
        disabled: document.querySelector('button').disabled, reloads: window.reloads || 0 }));
      assert.equal(result.calls[0].body.expected_revision, 1);
      assert.equal(result.draft, '仍在编辑的完整草稿');
      assert.equal(result.reloads, 0);
      assert.equal(result.disabled, false);
      assert.match(result.alerts[0], /当前修改已保留/);
      await page.evaluate(async () => {
        window.disconnect = true;
        await submitBucketContentEdit({ preventDefault() {}, target: document.getElementById('edit') }, 'memory-a');
      });
      result = await page.evaluate(() => ({ calls, draft: document.querySelector('textarea').value,
        disabled: document.querySelector('button').disabled }));
      assert.equal(result.calls[1].body.idempotency_key, result.calls[0].body.idempotency_key);
      assert.equal(result.draft, '仍在编辑的完整草稿');
      assert.equal(result.disabled, false);
      await page.evaluate(async () => {
        window.disconnect = false;
        window.statusCode = 200;
        document.getElementById('detail-panel').dataset.bucketRevision = '2';
        await submitBucketTitleEdit({ preventDefault() {}, target: document.getElementById('edit') }, 'memory-a');
        await submitBucketDateEdit({ preventDefault() {}, target: document.getElementById('edit') }, 'memory-a');
      });
      result = await page.evaluate(() => ({ calls, reloads: window.reloads }));
      assert.equal(result.calls[2].body.expected_revision, 2);
      assert.equal(result.calls[2].body.name, '新标题');
      assert.equal(result.calls[3].body.date, '2026-09-01');
      assert.equal(result.reloads, 2);
      await page.close();
      process.stdout.write('PASS legacy edit version/draft/retry/title/date ' + width + '\n');
    }
  } finally { await browser.close(); }
})().catch(error => { process.stderr.write(error.stack + '\n'); process.exitCode = 1; });
