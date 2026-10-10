// Real headless browser, synthetic fixtures and mocked HTTP only.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
const script = fs.readFileSync(path.join(root, 'dashboard_assets/chat-memory.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'dashboard.html'), 'utf8');
const styles = (html.match(/<style[^>]*>[\s\S]*?<\/style>/g) || []).join('\n');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  let passed = 0;
  async function fixture(width = 390) {
    const page = await browser.newPage({ viewport: { width, height: 844 } });
    await page.route('**/*', route => route.abort());
    await page.setContent(styles + '<div class="content"><div class="bucket-bulk-message" id="daily-chat-memory-message"></div><div id="daily-chat-memory-pending"></div></div>');
    await page.evaluate(() => {
      window.BASE = ''; window.getActiveTab = () => 'other'; window.confirm = () => true;
      window.esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
      window.escAttr = window.esc; window.jsString = v => String(v).replace(/'/g, "\\'");
      window.calls = []; window.failEdit = 0; window.failGet = false; window.editDelay = 0;
      window.confirmStatus = 'created'; window.loadBuckets = () => {};
      window.items = ['one', 'two'].map(id => ({ id, status: 'pending', authority_revision: 1,
        candidate: { title: id, content: '原始合成正文', kind: 'preference', tags: ['保留'], domain: ['日常'],
          importance: 5, confidence: 0, original_excerpt: '合成来源',
          narrative: { event_time: { precision: 'day', value: '2026-10-09' } } }, display: {} }));
      window.savedRequests = {}; window.confirmRequests = {}; window.failConfirm = 0; window.rejectConfirm = false;
      window.authFetch = async (url, options) => {
        const reply = (data, status = 200) => ({ok: status < 400, status, json: async () => structuredClone(data)});
        if (!options) { if (window.failGet) throw Error('offline'); return reply({items: window.items}); }
        const body = JSON.parse(options.body); window.calls.push({url, body});
        if (url.endsWith('/edit')) {
          if (window.editDelay) await new Promise(r => setTimeout(r, window.editDelay));
          if (window.savedRequests[body.request_id]) return reply(window.savedRequests[body.request_id]);
          const item = window.items.find(i => i.id === body.candidate_id);
          if (!item || item.authority_revision !== body.expected_revision) return reply({error:'revision_conflict'},409);
          const changed = body.edit.content && body.edit.content !== item.candidate.content;
          Object.assign(item.candidate, body.edit);
          if (body.edit.content) item.candidate.proposed_memory = body.edit.content;
          if (changed) item.candidate.narrative.event_time = {precision:'unknown'};
          if ('event_date' in body.edit) item.candidate.narrative.event_time = {precision:'day',value:body.edit.event_date};
          item.authority_revision++;
          const result = {status:'saved',adopted:false,item:structuredClone(item)};
          window.savedRequests[body.request_id] = result;
          if (window.failEdit-- > 0) throw Error('lost response');
          return reply(result);
        }
        if (window.rejectConfirm) { window.rejectConfirm = false; return reply({error:'invalid_request'},400); }
        if (window.confirmRequests[body.request_id]) return reply(window.confirmRequests[body.request_id]);
        const results = body.candidate_ids.map(id => ({ id,
          status: window.items.find(i => i.id === id)?.authority_revision !== body.expected_revisions[id]
            ? 'revision_conflict' : window.confirmStatus }));
        window.items = window.items.filter(i => !results.some(r => r.id === i.id && r.status === 'created'));
        const result = {status:'ok',results}; window.confirmRequests[body.request_id] = result;
        if (window.failConfirm-- > 0) throw Error('lost confirm response');
        return reply(result);
      };
    });
    await page.addScriptTag({content:script});
    await page.evaluate(() => window.loadDailyChatMemoryPending());
    return page;
  }
  const card = (page, id = 'one') => page.locator('[data-candidate-id="' + id + '"]');
  const open = async (page, id) => card(page,id).getByRole('button',{name:'编辑',exact:true}).click();
  const edit = async (page, text, id) => { await open(page,id); await card(page,id).locator('[data-field="content"]').fill(text); };
  const save = async (page,id) => card(page,id).getByRole('button',{name:'保存修改（仍待审核）',exact:true}).click();
  const result = async page => page.locator('#daily-chat-memory-message').textContent();
  async function test(name, fn) {
    const page = await fixture();
    try { await fn(page); passed++; console.log('PASS', name); }
    finally { await page.close(); }
  }
  try {
    await test('save remains pending and clears stale date', async p => {
      await edit(p,'改后的正文'); await save(p);
      assert.match(await result(p),/仍待审核/);
      assert.equal(await p.evaluate(() => items[0].status),'pending');
      assert.equal(await card(p).locator('[data-field="event_date"]').inputValue(),'');
      assert.equal(await p.evaluate(() => calls.filter(c => c.url.endsWith('/confirm')).length),0);
    });
    await test('collapsed edits saved before single adoption', async p => {
      await edit(p,'收起仍保留');
      await card(p).getByRole('button',{name:'收起编辑',exact:true}).click();
      await card(p).getByRole('button',{name:'采用并入库',exact:true}).click();
      assert.equal(await p.evaluate(() => calls[0].body.edit.content),'收起仍保留');
      assert.equal(await p.evaluate(() => calls[1].body.expected_revisions.one),2);
    });
    await test('batch adopts edited versions of both cards', async p => {
      await edit(p,'第一条'); await edit(p,'第二条','two');
      await p.locator('#daily-chat-memory-select-all').check();
      await p.getByRole('button',{name:'批量写入选中',exact:true}).click();
      assert.deepEqual(await p.evaluate(() => calls.map(c=>c.url.split('/').pop())),['edit','edit','confirm']);
      assert.deepEqual(await p.evaluate(() => calls[2].body.expected_revisions),{one:2,two:2});
    });
    await test('list refresh preserves dirty fields and expanded panel', async p => {
      await edit(p,'列表重载保留'); await p.evaluate(() => loadDailyChatMemoryPending());
      assert.equal(await card(p).locator('textarea').inputValue(),'列表重载保留');
      assert.equal(await card(p).locator('textarea').isVisible(),true);
    });
    await test('failed refresh does not discard draft', async p => {
      await edit(p,'断线草稿'); await p.evaluate(async () => {failGet=true; await loadDailyChatMemoryPending();});
      assert.equal(await card(p).locator('textarea').inputValue(),'断线草稿');
      assert.match(await result(p),/草稿保留/);
    });
    await test('stale draft conflicts and cannot adopt another version', async p => {
      await edit(p,'旧页面草稿');
      await p.evaluate(async () => {items[0].authority_revision++; await loadDailyChatMemoryPending();});
      await card(p).getByRole('button',{name:'采用并入库',exact:true}).click();
      assert.equal(await p.evaluate(() => calls.filter(c=>c.url.endsWith('/confirm')).length),0);
      assert.equal(await card(p).locator('textarea').inputValue(),'旧页面草稿');
    });
    await test('lost save response retries the same request', async p => {
      await edit(p,'回执丢失'); await p.evaluate(() => failEdit=1);
      await save(p);
      assert.equal(await card(p).locator('textarea').isDisabled(),true);
      await p.evaluate(() => loadDailyChatMemoryPending());
      assert.equal(await card(p).locator('textarea').isDisabled(),true);
      assert.equal(await card(p).getByRole('button',{name:'保存修改（仍待审核）',exact:true}).isDisabled(),false);
      await save(p);
      assert.equal(await p.evaluate(() => calls[0].body.request_id===calls[1].body.request_id),true);
      assert.equal(await p.evaluate(() => items[0].authority_revision),2);
      assert.match(await result(p),/仍待审核/);
      assert.equal(await card(p).locator('textarea').isDisabled(),false);
    });
    await test('lost adoption receipt survives refresh and retries original body', async p => {
      await p.evaluate(() => failConfirm=1);
      await card(p).getByRole('button',{name:'采用并入库',exact:true}).click();
      await p.evaluate(() => loadDailyChatMemoryPending());
      assert.equal(await card(p).count(),1);
      assert.equal(await card(p).locator('textarea').isDisabled(),true);
      await card(p).getByRole('button',{name:'采用并入库',exact:true}).click();
      assert.deepEqual(await p.evaluate(() => calls[0].body),await p.evaluate(() => calls[1].body));
      assert.match(await result(p),/成功 1／1/);
    });
    await test('definite rejection unlocks fields and permits a corrected action', async p => {
      await p.evaluate(() => rejectConfirm=true);
      await card(p).getByRole('button',{name:'采用并入库',exact:true}).click();
      await edit(p,'修正后重试'); await save(p);
      assert.match(await result(p),/仍待审核/);
      await card(p).getByRole('button',{name:'采用并入库',exact:true}).click();
      assert.match(await result(p),/成功 1／1/);
    });
    await test('title-only edit sends no body or time changes', async p => {
      await open(p); await card(p).locator('[data-field="title"]').fill('新标题'); await save(p);
      assert.deepEqual(await p.evaluate(() => Object.keys(calls[0].body.edit)),['title']);
      assert.equal(await card(p).locator('[data-field="event_date"]').inputValue(),'2026-10-09');
      assert.equal(await card(p).locator('.chat-memory-card-head strong').textContent(),'新标题');
    });
    await test('tag edits are arrays and zero confidence is preserved', async p => {
      await open(p); await card(p).locator('[data-field="tags"]').fill('甲，乙'); await save(p);
      assert.deepEqual(await p.evaluate(() => calls[0].body.edit.tags),['甲','乙']);
      assert.equal(await card(p).locator('[data-field="confidence"]').inputValue(),'0');
    });
    await test('unknown commit outcome does not show successful adoption', async p => {
      await p.evaluate(() => confirmStatus='commit_failed');
      await card(p).getByRole('button',{name:'采用并入库',exact:true}).click();
      assert.match(await result(p),/成功 0／1/);
      assert.equal(await card(p).count(),1);
    });
    await test('typing during pending fetch survives the late response', async p => {
      await edit(p,'起始草稿');
      await p.evaluate(() => {
        const fetch = authFetch;
        authFetch = async (...args) => { await new Promise(r=>setTimeout(r,100)); return fetch(...args); };
        window.pendingLoad = loadDailyChatMemoryPending();
      });
      await card(p).locator('textarea').fill('请求过程中继续写');
      await p.evaluate(() => pendingLoad);
      assert.equal(await card(p).locator('textarea').inputValue(),'请求过程中继续写');
    });
    for (const width of [390,1280]) {
      const p = await fixture(width); await edit(p,'可视合成草稿');
      const overflow = await p.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      assert.equal(overflow,false);
      if (process.env.EDITOR_SCREENSHOT_DIR) await p.screenshot({path:path.join(process.env.EDITOR_SCREENSHOT_DIR,`memory-editor-${width}.png`),fullPage:true});
      passed++; console.log('PASS viewport',width); await p.close();
    }
    console.log(JSON.stringify({passed,real_requests:0,production_writes:0}));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
