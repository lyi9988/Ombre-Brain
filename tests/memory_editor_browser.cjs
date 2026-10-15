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
  async function fixture(width = 390, withL1 = false) {
    const page = await browser.newPage({ viewport: { width, height: 844 } });
    page.on('pageerror', error => console.error('PAGEERROR', error));
    await page.route('**/*', route => route.abort());
    await page.setContent(styles + '<div class="content"><div class="bucket-bulk-message" id="daily-chat-memory-message"></div><div id="daily-chat-memory-pending"></div></div>');
    await page.evaluate((includeL1) => {
      window.BASE = ''; window.getActiveTab = () => 'other'; window.confirm = () => true;
      window.esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
      window.escAttr = window.esc; window.jsString = v => String(v).replace(/'/g, "\\'");
      window.calls = []; window.failEdit = 0; window.failGet = false; window.editDelay = 0;
      window.confirmStatus = 'created'; window.loadBuckets = () => {};
      window.items = ['one', 'two'].map(id => ({ id, status: 'pending', authority_revision: 1,
        candidate: { title: id, content: '原始合成正文', kind: 'preference', tags: ['保留'], domain: ['日常'],
          importance: 5, confidence: 0, original_excerpt: '合成来源',
          narrative: { event_time: { precision: 'day', value: '2026-10-09' } } }, display: {} }));
      window.syntheticL1 = (body = '日记整篇第一遍正文。') => ({
        schema_version: 'memory-l1-v1',
        entry: { form: 'diary_full', parent_ref: null, document_group_id: 'diary-root', derived_from_revision: 4,
          scene_ref: null, shared_statement_refs: [] },
        narrative: { title: '合成日记整篇', body },
        entities: [],
        scenes: [{ scene_id: 'scene-a', scene_revision_id: 'scene-rev-a', title: '场景A', anchors: [],
          statement_refs: [], attributes: [], generation: { origin: 'first_pass_model', task_id: 'scene-task' } }],
        statements: [{ statement_id: 'statement-a', statement_revision_id: 'statement-rev-a', content: '一次完整陈述',
          subject_ref: { state: 'not_stated', value: null, evidence_refs: [] },
          classification: { kind: { state: 'known', value: 'experience', evidence_refs: [] }, topics: ['daily_life'],
            stability: { state: 'not_stated', value: null, evidence_refs: [] },
            modality: { state: 'known', value: 'assertion', evidence_refs: [] },
            basis: { state: 'known', value: 'record', evidence_refs: [] } },
          time: { event: { start: { state: 'not_stated', value: null, evidence_refs: [] },
              end: { state: 'not_stated', value: null, evidence_refs: [] }, precision: 'unknown' },
            applicability: { temporal_scope: 'unknown', scene_ref: null } },
          conditions: {}, property: null, evidence_refs: [], role_hint: { state: 'not_stated', value: null, evidence_refs: [] }, attributes: [] }],
        retrieval: { keywords: ['初始词'], tags: ['日记'] },
        evidence_bindings: [{ binding_id: 'binding-a', source_namespace: 'snapshot.ombre.diary_mcp', source_id: 'diary-a',
          source_revision: 'source-rev-a', text_view_id: 'diary.text-v1', content_sha256: 'source-hash-a' }],
        relations: [], attributes: [],
        generation: { origin: 'first_pass_model', prompt_revision: 'prompt-r7', catalog_revision: 'catalog-r4',
          model_route_revision: 'route-r3', task_id: 'diary-task', actor: 'model', catalog_snapshot: { kinds: ['experience'] } },
      });
      if (includeL1) {
        const document = syntheticL1();
        items[0].candidate = { source_type: 'diary', title: document.narrative.title, content: document.narrative.body,
          proposed_memory: document.narrative.body, original_excerpt: '完整原文摘录', source_status: 'readable',
          source_refs: ['diary-a@source-rev-a'], l1_document: document,
          diary_l1_bundle: { memory_id: 'diary-root', expected_revision: 4,
            coverage: { state: 'complete', source_length: 200 }, scene_bodies: { 'scene-a': '场景A的完整正文。' } } };
      }
      window.savedRequests = {}; window.confirmRequests = {}; window.failConfirm = 0; window.rejectConfirm = false;
      window.authFetch = async (url, options) => {
        const reply = (data, status = 200) => ({ok: status < 400, status, json: async () => structuredClone(data)});
        if (url.includes('/api/memory-authority/source-tasks') && !options) return reply({tasks: []});
        if (!options) { if (window.failGet) throw Error('offline'); return reply({items: window.items}); }
        const body = JSON.parse(options.body); window.calls.push({url, body});
        if (url.endsWith('/edit')) {
          if (window.editDelay) await new Promise(r => setTimeout(r, window.editDelay));
          if (window.savedRequests[body.request_id]) return reply(window.savedRequests[body.request_id]);
          const item = window.items.find(i => i.id === body.candidate_id);
          if (!item || item.authority_revision !== body.expected_revision) return reply({error:'revision_conflict'},409);
          if (body.edit.document) {
            const document = structuredClone(body.edit.document);
            document.generation = { ...document.generation, origin: 'owner_edit', actor: 'owner' };
            item.candidate.l1_document = document;
            item.candidate.content = document.narrative.body;
            item.candidate.proposed_memory = document.narrative.body;
            item.candidate.title = document.narrative.title;
            if (body.edit.scene_bodies) item.candidate.diary_l1_bundle.scene_bodies = structuredClone(body.edit.scene_bodies);
            item.candidate.memory_l1 = structuredClone(document);
            delete item.candidate.memory_l1.narrative.body;
            item.authority_revision++;
            const result = {status:'saved',adopted:false,item:structuredClone(item)};
            window.savedRequests[body.request_id] = result;
            if (window.failEdit-- > 0) throw Error('lost response');
            return reply(result);
          }
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
    }, withL1);
    await page.addScriptTag({content:script});
    await page.evaluate(() => window.loadDailyChatMemoryPending());
    return page;
  }
  const card = (page, id = 'one') => page.locator('[data-candidate-id="' + id + '"]');
  const open = async (page, id) => card(page,id).getByRole('button',{name:'编辑',exact:true}).click();
  const edit = async (page, text, id) => { await open(page,id); await card(page,id).locator('[data-field="content"]').fill(text); };
  const save = async (page,id) => card(page,id).getByRole('button',{name:'保存修改（仍待审核）',exact:true}).click();
  const result = async page => page.locator('#daily-chat-memory-message').textContent();
  async function test(name, fn, withL1 = false) {
    const page = await fixture(390, withL1);
    try { await fn(page); passed++; console.log('PASS', name); }
    finally { await page.close(); }
  }
  {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    try {
      await page.setContent(html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '').replace(/<link\b[^>]*>/gi, ''));
      const fields = [
        ['cfg-reflection-diary-l1-input-window-chars', 'diary_l1_input_window_chars'],
        ['cfg-reflection-diary-l1-max-windows', 'diary_l1_max_windows'],
        ['cfg-reflection-diary-l1-scene-limit', 'diary_l1_scene_limit'],
        ['cfg-reflection-diary-l1-max-tokens', 'diary_l1_max_tokens'],
        ['cfg-reflection-diary-l1-scene-min-value', 'diary_l1_scene_min_value'],
        ['cfg-reflection-chat-memory-candidate-max-tokens', 'daily_chat_memory_candidate_max_tokens'],
        ['cfg-reflection-chat-memory-max-windows', 'daily_chat_memory_max_windows_per_run'],
        ['cfg-reflection-chat-memory-window-max-input-chars', 'daily_chat_memory_window_max_input_chars'],
        ['cfg-reflection-chat-memory-max-per-day', 'daily_chat_memory_max_per_day'],
        ['cfg-reflection-chat-memory-review-max-per-day', 'daily_chat_memory_review_max_per_day'],
      ];
      for (const [id, key] of fields) {
        const input = page.locator('#' + id);
        assert.equal(await input.count(), 1, 'dashboard has one control for ' + key);
        assert.equal(await input.getAttribute('min'), null, 'min bound comes from GET for ' + key);
        assert.equal(await input.getAttribute('max'), null, 'max bound comes from GET for ' + key);
        assert.equal(await input.getAttribute('placeholder'), '从服务器配置读取');
        assert.equal(await input.getAttribute('value'), null, 'UI does not invent a config value for ' + key);
        assert.ok(html.includes("'" + id + "': '" + key + "'"), 'GET mapping exists for ' + key);
        assert.ok(html.includes(key + ': generationConfig.' + key), 'POST mapping exists for ' + key);
      }
      assert.ok(html.includes('generation_parameter_bounds'), 'input bounds come from the server GET');
      assert.ok(html.includes('input.min = String(bounds.minimum)') && html.includes('input.max = String(bounds.maximum)'), 'server bounds are applied to controls');
      assert.ok(html.includes('generationIntegerValue') && html.includes('generationDecimalValue'), 'save validates with server-provided bounds');
      assert.ok(html.includes('reflectionConfigDraftSnapshot'), 'reflection setting drafts retain a GET baseline');
      assert.ok(html.includes('请求失败，生效未确认；输入已保留'), 'failed config saves keep the visible draft');
      assert.ok(html.includes('最终记忆召回或 Prompt 注入条数'), 'background budgets are distinguished from final recall/injection limits');
      passed++; console.log('PASS dashboard exposes GET-backed generation controls and bounded save wiring');
    } finally { await page.close(); }
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
    await test('complete L1 JSON save reads back body and semantics while remaining pending', async p => {
      await open(p, 'one');
      await card(p, 'one').getByRole('button', {name:'完整L1 JSON'}).click();
      const json = card(p, 'one').locator('[data-l1-json]');
      const document = JSON.parse(await json.inputValue());
      document.retrieval.keywords = ['主人新增完整JSON字段'];
      await json.fill(JSON.stringify(document, null, 2));
      await save(p, 'one');
      assert.equal(await p.evaluate(() => calls.filter(c => c.url.endsWith('/confirm')).length), 0);
      assert.equal(await p.evaluate(() => items[0].status), 'pending');
      assert.deepEqual(await p.evaluate(() => calls[0].body.edit.document.retrieval.keywords), ['主人新增完整JSON字段']);
      assert.equal(await p.evaluate(() => calls[0].body.edit.document.narrative.body), '日记整篇第一遍正文。');
      assert.match(await card(p, 'one').locator('.chat-memory-l1-details').textContent(), /readable|owner_edit|source-rev-a/);
      assert.equal(JSON.parse(await json.inputValue()).retrieval.keywords[0], '主人新增完整JSON字段');
      assert.match(await result(p), /仍待审核/);
    }, true);
    await test('L1 body and JSON views share a single dirty document', async p => {
      await open(p, 'one');
      const body = card(p, 'one').locator('[data-l1-body]');
      await body.fill('主人编辑后的整篇L1正文。');
      await card(p, 'one').getByRole('button', {name:'完整L1 JSON'}).click();
      const json = card(p, 'one').locator('[data-l1-json]');
      const document = JSON.parse(await json.inputValue());
      assert.equal(document.narrative.body, '主人编辑后的整篇L1正文。');
      document.retrieval.tags = ['JSON端修改'];
      await json.fill(JSON.stringify(document, null, 2));
      await card(p, 'one').getByRole('button', {name:'正文与场景'}).click();
      assert.equal(await body.inputValue(), '主人编辑后的整篇L1正文。');
      await save(p, 'one');
      assert.equal(await p.evaluate(() => calls[0].body.edit.document.narrative.body), '主人编辑后的整篇L1正文。');
      assert.deepEqual(await p.evaluate(() => calls[0].body.edit.document.retrieval.tags), ['JSON端修改']);
    }, true);
    await test('diary scene body is an explicit part of the save-only candidate edit', async p => {
      await open(p, 'one');
      await card(p, 'one').locator('[data-scene-body="scene-a"]').fill('主人编辑后的场景正文。');
      await save(p, 'one');
      assert.deepEqual(await p.evaluate(() => calls[0].body.edit.scene_bodies), {'scene-a':'主人编辑后的场景正文。'});
      assert.deepEqual(await p.evaluate(() => items[0].candidate.diary_l1_bundle.scene_bodies), {'scene-a':'主人编辑后的场景正文。'});
      assert.equal(await p.evaluate(() => items[0].status), 'pending');
    }, true);
    await test('split scene edit keeps old child text visible until owner maps or removes it', async p => {
      await open(p, 'one');
      await card(p, 'one').getByRole('button', {name:'完整L1 JSON'}).click();
      const json = card(p, 'one').locator('[data-l1-json]');
      const document = JSON.parse(await json.inputValue());
      document.scenes = [{ ...document.scenes[0], scene_id: 'new:scene', scene_revision_id: null, title: '调整后的场景' }];
      await json.fill(JSON.stringify(document, null, 2));
      const orphan = card(p, 'one').locator('[data-orphan-scene-body="scene-a"]');
      assert.equal(await orphan.inputValue(), '场景A的完整正文。');
      await card(p, 'one').locator('[data-orphan-scene-target]').selectOption('new:scene');
      await card(p, 'one').getByRole('button', {name:'复制到所选场景并移除旧项'}).click();
      assert.equal(await card(p, 'one').locator('[data-scene-body="new:scene"]').inputValue(), '场景A的完整正文。');
      assert.equal(await card(p, 'one').locator('[data-orphan-scene-body="scene-a"]').count(), 0);
      await save(p, 'one');
      assert.deepEqual(await p.evaluate(() => calls[0].body.edit.scene_bodies), {'new:scene':'场景A的完整正文。'});
    }, true);
    await test('invalid L1 JSON stays dirty and cannot be saved or adopted', async p => {
      await open(p, 'one');
      await card(p, 'one').getByRole('button', {name:'完整L1 JSON'}).click();
      await card(p, 'one').locator('[data-l1-json]').fill('{');
      await save(p, 'one');
      assert.match(await card(p, 'one').locator('[data-l1-json-error]').textContent(), /尚未解析成功/);
      await card(p, 'one').getByRole('button', {name:'采用并入库'}).click();
      assert.equal(await p.evaluate(() => calls.length), 0);
      assert.match(await result(p), /JSON/);
    }, true);
    await test('new L1 input typed during a save remains dirty and uses the next CAS revision', async p => {
      await open(p, 'one');
      const body = card(p, 'one').locator('[data-l1-body]');
      await body.fill('第一次提交正文。');
      await p.evaluate(() => editDelay = 100);
      const pendingSave = save(p, 'one');
      await p.waitForTimeout(20);
      await body.fill('提交期间新增的正文。');
      await pendingSave;
      await p.waitForFunction(() => items[0].authority_revision === 2);
      assert.equal(await body.inputValue(), '提交期间新增的正文。');
      assert.match(await result(p), /仍留在草稿/);
      await p.evaluate(() => editDelay = 0);
      await save(p, 'one');
      assert.equal(await p.evaluate(() => calls[1].body.expected_revision), 2);
      assert.equal(await p.evaluate(() => calls[1].body.edit.document.narrative.body), '提交期间新增的正文。');
    }, true);
    await test('L1 save and adoption remain two distinct requests', async p => {
      await open(p, 'one');
      await card(p, 'one').locator('[data-l1-body]').fill('准备采用的完整正文。');
      await card(p, 'one').getByRole('button', {name:'采用并入库'}).click();
      assert.deepEqual(await p.evaluate(() => calls.map(c => c.url.split('/').pop())), ['edit','confirm']);
      assert.equal(await p.evaluate(() => calls[0].body.edit.document.narrative.body), '准备采用的完整正文。');
      assert.equal(await p.evaluate(() => calls[1].body.expected_revisions.one), 2);
    }, true);
    for (const width of [390,1280]) {
      const p = await fixture(width, true);
      await open(p, 'one');
      await card(p, 'one').locator('[data-l1-body]').fill('可视合成的整篇L1正文。');
      await card(p, 'one').locator('[data-scene-body="scene-a"]').fill('可视合成的场景正文。');
      const overflow = await p.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      assert.equal(overflow,false);
      if (process.env.EDITOR_SCREENSHOT_DIR) await p.screenshot({path:path.join(process.env.EDITOR_SCREENSHOT_DIR,`memory-editor-${width}.png`),fullPage:true});
      passed++; console.log('PASS viewport',width); await p.close();
    }
    console.log(JSON.stringify({passed,real_requests:0,production_writes:0}));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
