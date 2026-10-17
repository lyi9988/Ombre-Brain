const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const assert = require('node:assert/strict');

const source = fs.readFileSync(path.join(__dirname, '..', 'dashboard_assets', 'chat-memory.js'), 'utf8');

function extractFunction(name) {
  const marker = `function ${name}(`;
  const start = source.indexOf(marker);
  assert.notEqual(start, -1, `${name} exists in chat-memory.js`);
  const open = source.indexOf('{', start);
  assert.notEqual(open, -1, `${name} has a function body`);
  let depth = 0;
  let quote = '';
  let escaped = false;
  for (let index = open; index < source.length; index += 1) {
    const char = source[index];
    if (quote) {
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === quote) quote = '';
      continue;
    }
    if (char === "'" || char === '"' || char === '`') { quote = char; continue; }
    if (char === '{') depth += 1;
    else if (char === '}' && --depth === 0) return source.slice(start, index + 1);
  }
  assert.fail(`${name} has an unterminated body`);
}

const dom = { document: null };
const sandbox = { window: dom, displayValue: (value) => String(value) };
vm.createContext(sandbox);
vm.runInContext([
  extractFunction('cognitionInputText'),
  extractFunction('captureL1CognitionForm'),
  extractFunction('syncL1CognitionForm'),
  'globalThis.capture = captureL1CognitionForm;',
  'globalThis.sync = syncL1CognitionForm;',
].join('\n'), sandbox, { filename: 'chat-memory-cognition-functions.js' });

function makeDom() {
  const document = {
    activeElement: null,
    created: [],
    createElement(tag) {
      assert.equal(tag, 'option');
      const option = { value: '', textContent: '', disabled: false };
      this.created.push(option);
      return option;
    },
  };
  return document;
}

function makeInput(field, value = '') {
  return {
    value,
    disabled: false,
    getAttribute(name) { return name === 'data-cognition-field' ? field : null; },
  };
}

function makeCard(document, fieldValues = {}, activeRoot = null) {
  const select = {
    value: '', disabled: false, children: [], attributes: {},
    replaceChildren() { this.children = []; },
    appendChild(child) { this.children.push(child); },
    setAttribute(name, value) { this.attributes[name] = value; },
  };
  const inputs = Object.entries(fieldValues).map(([field, value]) => makeInput(field, value));
  const byField = new Map(inputs.map((input) => [input.getAttribute('data-cognition-field'), input]));
  const card = {
    ownerDocument: document,
    getRootNode() { return activeRoot || document; },
    querySelector(selector) {
      if (selector === '[data-cognition-statement]') return select;
      const match = selector.match(/^\[data-cognition-field="([^"]+)"\]$/);
      return match ? byField.get(match[1]) || null : null;
    },
    querySelectorAll(selector) {
      return selector === '[data-cognition-field]' ? inputs : [];
    },
  };
  return { card, select, inputs: byField };
}

function l1(document) {
  return JSON.parse(JSON.stringify(document));
}

test('sync uses DOM owner document with an L1 document and preserves focus through open shadow roots', () => {
  const document = makeDom();
  const focused = makeInput('role.value', '主人正在编辑的新角色');
  const nestedHost = { shadowRoot: { activeElement: focused } };
  document.activeElement = nestedHost;
  const shadowRoot = { activeElement: nestedHost };
  const form = makeCard(document, {
    'role.state': '草稿旧值', 'role.value': focused.value,
    'epistemic_status.state': '旧状态', 'epistemic_status.value': '旧取值',
    about_refs: '旧对象', open_questions: '旧问题',
  }, shadowRoot);
  form.inputs.set('role.value', focused);
  const l1Document = {
    schema_version: 'memory-l1-v1',
    narrative: { title: '合成记忆', body: '合成正文' },
    statements: [{ statement_id: 'statement-a', content: '一条完整陈述', cognition: {
      role: { state: 'unresolved', value: null, evidence_refs: ['source-a'] },
      epistemic_status: { state: 'known', value: 'tentative', evidence_refs: [] },
      about_refs: ['entity-a'], open_questions: ['还待核对什么？'],
    } }],
  };
  dom.document = document;

  sandbox.sync(form.card, l1Document, 0);

  assert.equal(document.created.length, 1);
  assert.equal(form.select.value, '0');
  assert.match(form.select.children[0].textContent, /statement-a.*一条完整陈述/);
  assert.equal(form.inputs.get('role.state').value, 'unresolved');
  assert.equal(form.inputs.get('role.value').value, '主人正在编辑的新角色', 'active draft in a nested shadow root is not overwritten');
  assert.equal(form.inputs.get('epistemic_status.value').value, 'tentative');
  assert.equal(form.inputs.get('about_refs').value, 'entity-a');
});

test('sync handles a real L1 object without statements and disables cognition controls', () => {
  const document = makeDom();
  const form = makeCard(document, {
    'role.state': 'stale', 'role.value': 'stale',
    'epistemic_status.state': 'stale', 'epistemic_status.value': 'stale',
    about_refs: 'stale', open_questions: 'stale',
  });
  dom.document = document;

  sandbox.sync(form.card, { schema_version: 'memory-l1-v1', narrative: { title: '', body: '' }, statements: [] }, 0);

  assert.equal(form.select.children.length, 1);
  assert.equal(form.select.children[0].textContent, '当前没有可编辑的现有陈述');
  assert.equal(form.select.disabled, true);
  for (const input of form.inputs.values()) {
    assert.equal(input.value, '');
    assert.equal(input.disabled, true);
  }
});

test('capture creates complete cognition containers while retaining explicit unknown and extension data', () => {
  const document = makeDom();
  const form = makeCard(document, {
    'role.state': '', 'role.value': 'question',
    'epistemic_status.state': '', 'epistemic_status.value': '',
    about_refs: '', open_questions: '',
  });
  const l1Document = { schema_version: 'memory-l1-v1', narrative: { title: '', body: '' }, statements: [
    { statement_id: 'statement-a', content: '正文保留', subject_ref: { state: 'not_stated', value: null, evidence_refs: [] } },
  ] };
  dom.document = document;

  sandbox.capture(form.card, l1Document, 0);

  const cognition = JSON.parse(JSON.stringify(l1Document.statements[0].cognition));
  assert.deepEqual(cognition, {
    role: { state: 'known', value: 'question', evidence_refs: [] },
    epistemic_status: { state: 'not_stated', value: null, evidence_refs: [] },
    about_refs: [], open_questions: [],
  });
  assert.equal(l1Document.statements[0].content, '正文保留');
  assert.equal(l1Document.statements[0].subject_ref.state, 'not_stated');

  const untouched = { statements: [{ statement_id: 'statement-b', content: '未变陈述' }] };
  sandbox.capture(makeCard(document, {
    'role.state': '', 'role.value': '', 'epistemic_status.state': '', 'epistemic_status.value': '',
    about_refs: '', open_questions: '',
  }).card, untouched, 0);
  assert.equal(Object.hasOwn(untouched.statements[0], 'cognition'), false, 'unmodified legacy statement stays without cognition');

  const explicitUnknown = makeCard(document, {
    'role.state': 'unresolved', 'role.value': '',
    'epistemic_status.state': '', 'epistemic_status.value': '',
    about_refs: '', open_questions: '先留一个问题',
  });
  const unknownDocument = { statements: [{ statement_id: 'statement-c', content: '待核对' }] };
  sandbox.capture(explicitUnknown.card, unknownDocument, 0);
  assert.deepEqual(JSON.parse(JSON.stringify(unknownDocument.statements[0].cognition.role)), {
    state: 'unresolved', value: null, evidence_refs: [],
  }, 'explicit uncertainty does not become known');
  assert.deepEqual(JSON.parse(JSON.stringify(unknownDocument.statements[0].cognition.epistemic_status)), {
    state: 'not_stated', value: null, evidence_refs: [],
  });

  const existing = { statements: [{ statement_id: 'statement-d', content: '保留原文', cognition: {
    role: { state: 'unresolved', value: null, evidence_refs: ['evidence-a'], extension: 'keep' },
    epistemic_status: { state: 'not_stated', value: null, evidence_refs: [] },
    about_refs: ['entity-a'], open_questions: ['旧问题'], extension: { keep: true },
  } }] };
  const edit = makeCard(document, {
    'role.state': 'unresolved', 'role.value': '',
    'epistemic_status.state': 'not_stated', 'epistemic_status.value': '',
    about_refs: 'entity-a', open_questions: '新问题',
  });
  sandbox.capture(edit.card, existing, 0);
  assert.deepEqual(JSON.parse(JSON.stringify(existing.statements[0].cognition)), {
    role: { state: 'unresolved', value: null, evidence_refs: ['evidence-a'], extension: 'keep' },
    epistemic_status: { state: 'not_stated', value: null, evidence_refs: [] },
    about_refs: ['entity-a'], open_questions: ['新问题'], extension: { keep: true },
  });
});
