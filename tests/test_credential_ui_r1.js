const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const html = fs.readFileSync(path.join(__dirname, '..', 'dashboard.html'), 'utf8').replace(/\r\n/g, '\n');
function extract(name) {
  const start = html.indexOf(`function ${name}(`);
  const end = html.indexOf('\n}\n', start);
  assert.ok(start >= 0 && end > start, name);
  return html.slice(start, end + 2);
}
const context = vm.createContext({normalizeGatewayModelEntry: value => value});
vm.runInContext(['splitEnvLines', 'splitSecretLines', 'normalizeGatewayUpstream', 'configSavePresentation']
  .map(extract).join('\n'), context);

test('two key slots keep a blank first slot through normalization', () => {
  const slots = context.splitSecretLines('\nTEST_ONLY_second\n');
  assert.deepEqual(Array.from(slots), ['', 'TEST_ONLY_second']);
  const normalized = context.normalizeGatewayUpstream({api_key_envs:['FIRST','SECOND'], api_key_values:slots});
  assert.deepEqual(Array.from(normalized.api_key_values), ['', 'TEST_ONLY_second']);
  assert.equal(normalized.credential_mode, 'replace');
  assert.equal(context.splitSecretLines('\n\n').length, 0);
});

test('append mode survives normalization and two populated slots keep order', () => {
  const normalized = context.normalizeGatewayUpstream({
    credential_mode:'append', api_key_values:['TEST_ONLY_one','TEST_ONLY_two']});
  assert.equal(normalized.credential_mode, 'append');
  assert.deepEqual(Array.from(normalized.api_key_values), ['TEST_ONLY_one','TEST_ONLY_two']);
});

test('host persistence not confirmed is visible and retains inputs', () => {
  const result = context.configSavePresentation({
    ok:true, local_update:'applied', attention_required:true,
    persistence:{credentials:'saved', credential_scope:'brain_visible_file', host_target_confirmed:false},
    gateway_activation:{state:'confirmed', provider_verified:false}});
  assert.equal(result.refresh, false);
  assert.equal(result.tone, 'warning');
  assert.match(result.text, /宿主持久目标尚未独立确认/);
});

test('conflict review advances revision only after explicit review and never reloads fields', async () => {
  const start = html.indexOf('async function reviewCredentialConflict(');
  const end = html.indexOf('\n}\n', start);
  let accepted = false;
  const review = vm.createContext({
    BASE:'', configSnapshot:{credential_revision:'before'},
    authFetch:async()=>({ok:true,json:async()=>({credential_revision:'after',gateway:{upstreams:[]}})}),
    confirm:()=>accepted, setConfigStatus:()=>{},
    document:{getElementById:()=>{throw Error('review must preserve existing field values');}},
  });
  vm.runInContext(html.slice(start,end+2), review);
  await review.reviewCredentialConflict();
  assert.equal(review.configSnapshot.credential_revision, 'before');
  accepted = true;
  await review.reviewCredentialConflict();
  assert.equal(review.configSnapshot.credential_revision, 'after');
});

test('every inline script remains syntactically valid', () => {
  for (const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g))
    new vm.Script(match[1]);
});
