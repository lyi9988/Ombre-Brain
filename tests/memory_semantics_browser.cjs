// Headless browser, isolated synthetic candidate API; no real requests.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
const script = fs.readFileSync(path.join(root, 'dashboard_assets/chat-memory.js'), 'utf8');
const styles = (fs.readFileSync(path.join(root, 'dashboard.html'), 'utf8').match(/<style[^>]*>[\s\S]*?<\/style>/g) || []).join('\n');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  let passed = 0, blocked = 0;
  try {
    for (const width of [390, 1280]) {
      const page = await browser.newPage({ viewport: { width, height: 844 } });
      await page.route('**/*', route => { blocked++; return route.abort(); });
      await page.setContent(styles + '<div class="content"><div id="daily-chat-memory-message"></div><div id="daily-chat-memory-pending"></div></div>');
      await page.evaluate(() => {
        window.BASE = ''; window.getActiveTab = () => 'other'; window.confirm = () => true;
        window.esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
        window.escAttr = window.esc; window.jsString = v => String(v).replace(/'/g, "\\'");
        const semantics = { version:'memory-semantics-v1', state:'current',
          items:[{semantic_kind:'preference',subject_ref:{role:'user'},assertion_basis:'owner_statement',
            assertion_state:'stated',conditions_status:'not_stated', qualifiers:{topic:'茶',value:'<img src=x onerror="window.xss=true">'},
            evidence_refs:[{namespace:'canonical_event',event_id:'evt-synthetic',version_id:'version-synthetic'}]}],
          links:[{target_candidate_id:'chat-memory-synthetic',target_status_at_link:'pending'}],
          source_coverage:[{bridge_status:'exact_runtime_bridge'}], issues:[] };
        window.items = ['new','legacy','invalid'].map(id => ({id,status:'pending',authority_revision:1,
          candidate:{title:'合成记忆 '+id,content:'她告诉我自己喜欢喝茶，我保留她当时的原话。',kind:'preference',
            original_excerpt:'我喜欢喝茶。',semantic_annotations:id==='legacy'?null:structuredClone(semantics)},
          display:{semantics:{state:id==='invalid'?'invalidated':'current',annotation_count:1,source_link_count:1,unresolved_count:0}}}));
        window.authFetch = async (_url, options) => {
          if (options) throw Error('unexpected write');
          return {ok:true,json:async()=>({items:structuredClone(items)})};
        };
      });
      await page.addScriptTag({content:script});
      await page.evaluate(() => loadDailyChatMemoryPending());
      const card = page.locator('[data-candidate-id="new"]');
      assert.equal(await card.locator('details').last().getAttribute('open'), null);
      assert.equal(await page.locator('[data-candidate-id="legacy"] .chat-memory-semantics').textContent(), '');
      assert.match(await page.locator('[data-candidate-id="invalid"] .chat-memory-semantics').textContent(), /待复核/);
      await card.locator('.chat-memory-semantics summary').click();
      assert.match(await card.locator('.chat-memory-semantics').textContent(), /条件未说明.*永久/);
      assert.match(await card.locator('.chat-memory-semantics').textContent(), /不代表同一事实/);
      assert.equal(await page.locator('.chat-memory-semantics img').count(), 0);
      assert.equal(await page.evaluate(() => Boolean(window.xss)), false);
      passed += 6;
      await card.getByRole('button',{name:'编辑',exact:true}).click();
      const body = card.locator('[data-field="content"]');
      await body.fill('修改正文后，旧标注不能再当作当前解释。');
      assert.match(await card.locator('.chat-memory-semantics').textContent(), /待复核/);
      assert.equal(await card.locator('.chat-memory-semantics li').count(), 0);
      await page.evaluate(() => loadDailyChatMemoryPending());
      assert.match(await card.locator('.chat-memory-semantics').textContent(), /待复核/);
      await body.fill('她告诉我自己喜欢喝茶，我保留她当时的原话。');
      await card.locator('.chat-memory-semantics summary').click();
      assert.match(await card.locator('.chat-memory-semantics').textContent(), /主人原话/);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      passed += 5;
      if (process.env.SEMANTICS_SCREENSHOT_DIR) await page.screenshot({path:path.join(process.env.SEMANTICS_SCREENSHOT_DIR,`memory-semantics-${width}.png`),fullPage:true});
      await page.close();
    }
    console.log(JSON.stringify({passed,blocked_requests:blocked,real_requests:0,production_writes:0}));
  } finally { await browser.close(); }
})().catch(e => {console.error(e); process.exitCode=1;});
