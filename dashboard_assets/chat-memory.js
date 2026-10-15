(function () {
  var memoryItems = new Map();
  var memoryDrafts = new Map();
  var memoryRequests = new Map();
  var memoryLoadEpoch = 0;
  var memorySourceLoadEpoch = 0;
  var memoryBusy = false;
  var memoryStateRestored = false;
  var MEMORY_STATE_KEY = 'guyan.ombre.daily-chat-memory-review.v1';

  function cloneValue(value) {
    return value == null ? value : JSON.parse(JSON.stringify(value));
  }

  function persistMemoryState() {
    try {
      if (!window.sessionStorage) return;
      var drafts = Array.from(memoryDrafts.entries());
      var requests = Array.from(memoryRequests.entries());
      if (!drafts.length && !requests.length) {
        window.sessionStorage.removeItem(MEMORY_STATE_KEY);
        return;
      }
      window.sessionStorage.setItem(MEMORY_STATE_KEY, JSON.stringify({ version: 1, drafts: drafts, requests: requests }));
    } catch (_) { /* Session storage may be unavailable in a restricted browser context. */ }
  }

  function restoreMemoryState() {
    if (memoryStateRestored) return;
    memoryStateRestored = true;
    try {
      if (!window.sessionStorage) return;
      var raw = window.sessionStorage.getItem(MEMORY_STATE_KEY);
      if (!raw) return;
      var saved = JSON.parse(raw);
      if (!saved || saved.version !== 1) return;
      (Array.isArray(saved.drafts) ? saved.drafts : []).forEach(function (pair) {
        if (Array.isArray(pair) && typeof pair[0] === 'string' && pair[1] && typeof pair[1] === 'object') {
          memoryDrafts.set(pair[0], pair[1]);
        }
      });
      (Array.isArray(saved.requests) ? saved.requests : []).forEach(function (pair) {
        if (Array.isArray(pair) && typeof pair[0] === 'string' && pair[1] && pair[1].body) {
          memoryRequests.set(pair[0], pair[1]);
        }
      });
    } catch (_) { /* Ignore malformed session-only draft state; server data remains authoritative. */ }
  }

  function splitTerms(value) {
    return Array.from(new Set(String(value || '').split(/[,，\n]/).map(function (v) { return v.trim(); }).filter(Boolean)));
  }

  function itemFields(item) {
    var c = item.candidate || {};
    var time = (c.narrative || {}).event_time || {};
    return { title: c.title || item.id || '', content: c.proposed_memory || c.content || '',
      kind: c.kind || 'memory', domain: splitTerms(listText(c.domain)), tags: splitTerms(listText(c.tags)),
      importance: Number(c.importance || 5), confidence: Number(c.confidence == null ? 0.7 : c.confidence),
      event_date: time.precision === 'day' ? (time.value || '') : '' };
  }

  function itemL1Document(item) {
    var candidate = item && item.candidate && typeof item.candidate === 'object' ? item.candidate : {};
    var document = candidate.l1_document || candidate.memory_l1_document ||
      item && item.l1_document || candidate.memory_l1 || item && item.memory_l1;
    if (!document || typeof document !== 'object' || Array.isArray(document)) return null;
    try { document = cloneValue(document); } catch (_) { return null; }
    if (!document.narrative || typeof document.narrative !== 'object' || Array.isArray(document.narrative)) return null;
    if (typeof document.narrative.body !== 'string') {
      document.narrative.body = String(candidate.proposed_memory || candidate.content || item.body || '');
    }
    return document;
  }

  function itemSceneBodies(item) {
    var candidate = item && item.candidate && typeof item.candidate === 'object' ? item.candidate : {};
    var metadata = candidate.metadata && typeof candidate.metadata === 'object' ? candidate.metadata : {};
    var bundle = candidate.diary_l1_bundle || metadata.diary_l1_bundle || item && item.diary_l1_bundle || {};
    var value = candidate.scene_bodies || (bundle && bundle.scene_bodies) || item && item.scene_bodies || {};
    return value && typeof value === 'object' && !Array.isArray(value) ? cloneValue(value) : {};
  }

  function itemIsDiary(item) {
    var candidate = item && item.candidate && typeof item.candidate === 'object' ? item.candidate : {};
    var metadata = candidate.metadata && typeof candidate.metadata === 'object' ? candidate.metadata : {};
    return candidate.source_type === 'diary' || item && item.source_type === 'diary' ||
      Boolean(candidate.diary_l1_bundle || metadata.diary_l1_bundle || item && item.diary_l1_bundle);
  }

  function sameValue(left, right) {
    return JSON.stringify(left) === JSON.stringify(right);
  }

  function l1DraftChanged(draft) {
    return Boolean(draft && (draft.invalidJson || !sameValue(draft.document, draft.baseDocument) ||
      !sameValue(draft.sceneBodies || {}, draft.baseSceneBodies || {})));
  }

  function sameDraft(left, right) {
    if (!left || !right || left.type !== right.type) return left === right;
    if (left.type === 'l1') return !l1DraftChanged({
      invalidJson: left.invalidJson,
      document: left.document,
      baseDocument: right.document,
      sceneBodies: left.sceneBodies,
      baseSceneBodies: right.sceneBodies,
    });
    return sameValue(left.fields, right.fields);
  }

  function captureMemoryDraft(card) {
    if (!card) return;
    var id = card.getAttribute('data-candidate-id');
    var item = memoryItems.get(id);
    if (!item) return;
    var prior = memoryDrafts.get(id);
    var l1Document = itemL1Document(item);
    if (l1Document) {
      var baseDocument = prior && prior.type === 'l1' ? prior.baseDocument : l1Document;
      var document = prior && prior.type === 'l1' ? cloneValue(prior.document) : cloneValue(l1Document);
      var baseSceneBodies = prior && prior.type === 'l1' ? prior.baseSceneBodies : itemSceneBodies(item);
      var sceneBodies = prior && prior.type === 'l1' ? cloneValue(prior.sceneBodies || {}) : cloneValue(baseSceneBodies || {});
      var mode = card.getAttribute('data-l1-editor-mode') || (prior && prior.mode) || 'body';
      var invalidJson = false;
      var rawJson = '';
      if (mode === 'json') {
        var jsonInput = card.querySelector('[data-l1-json]');
        rawJson = jsonInput ? String(jsonInput.value || '') : JSON.stringify(document, null, 2);
        try {
          var parsed = JSON.parse(rawJson);
          if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed) ||
              !parsed.narrative || typeof parsed.narrative !== 'object' || typeof parsed.narrative.body !== 'string') {
            throw new Error('invalid_l1_document');
          }
          document = parsed;
        } catch (_) { invalidJson = true; }
      } else {
        var titleInput = card.querySelector('[data-l1-title]');
        var bodyInput = card.querySelector('[data-l1-body]');
        if (titleInput) document.narrative.title = String(titleInput.value || '');
        if (bodyInput) document.narrative.body = String(bodyInput.value || '');
        rawJson = JSON.stringify(document, null, 2);
      }
      card.querySelectorAll('[data-scene-body], [data-orphan-scene-body]').forEach(function (input) {
        var field = input.hasAttribute('data-scene-body') ? 'data-scene-body' : 'data-orphan-scene-body';
        sceneBodies[input.getAttribute(field)] = String(input.value || '');
      });
      if (mode === 'json' && !invalidJson) refreshDiarySceneBodies(card, document, sceneBodies);
      var changed = invalidJson || !sameValue(document, baseDocument) || !sameValue(sceneBodies, baseSceneBodies || {});
      if (changed) {
        memoryDrafts.set(id, { type: 'l1', baseDocument: cloneValue(baseDocument), document: document,
          rawJson: rawJson, sceneBodies: sceneBodies, baseSceneBodies: cloneValue(baseSceneBodies || {}),
          invalidJson: invalidJson, mode: mode, revision: prior ? prior.revision : Number(card.getAttribute('data-revision')),
          expanded: card.getAttribute('data-editing') === 'true' });
      } else memoryDrafts.delete(id);
      var error = card.querySelector('[data-l1-json-error]');
      if (error) error.textContent = invalidJson ? 'JSON 尚未解析成功；草稿已保留，修复 JSON 后才能保存或采用。' : '';
      persistMemoryState();
      return;
    }
    var base = prior ? prior.base : itemFields(item);
    var fields = readDailyChatMemoryEdits(card);
    var edits = {};
    Object.keys(fields).forEach(function (key) {
      if (JSON.stringify(fields[key]) !== JSON.stringify(base[key])) edits[key] = fields[key];
    });
    if (Object.keys(edits).length) {
      memoryDrafts.set(id, { base: base, fields: fields, edits: edits,
        revision: prior ? prior.revision : Number(card.getAttribute('data-revision')),
        expanded: card.getAttribute('data-editing') === 'true' });
    } else memoryDrafts.delete(id);
    persistMemoryState();
    var semanticPanel = card.querySelector('.chat-memory-semantics');
    if (semanticPanel) {
      var wasOpen = Boolean(semanticPanel.querySelector('details[open]'));
      semanticPanel.innerHTML = renderMemorySemantics(item, memoryDrafts.get(id));
      var details = semanticPanel.querySelector('details');
      if (details && wasOpen) details.open = true;
    }
  }

  function captureMemoryDrafts() {
    document.querySelectorAll('.chat-memory-card').forEach(captureMemoryDraft);
  }

  function stableMemoryRequest(key, body) {
    var signature = JSON.stringify(body);
    var prior = memoryRequests.get(key);
    if (prior && prior.signature !== signature) throw new Error('上次操作结果尚未确认，请先重试原操作；草稿已保留。');
    if (!prior) {
      prior = { signature: signature, body: Object.assign({}, body, { request_id: newDailyChatMemoryRequestId() }) };
      memoryRequests.set(key, prior);
      persistMemoryState();
    }
    return prior.body;
  }

  function pendingMemoryAction(id) {
    return Array.from(memoryRequests.entries()).find(function (entry) {
      return entry[0].indexOf('edit:') !== 0 && (entry[1].body.candidate_ids || []).includes(id);
    });
  }

  function syncUncertainLocks() {
    document.querySelectorAll('.chat-memory-card').forEach(function (card) {
      var id = card.getAttribute('data-candidate-id');
      var locked = memoryRequests.has('edit:' + id) || Boolean(pendingMemoryAction(id));
      card.querySelectorAll('input[data-field], textarea[data-field], [data-l1-title], [data-l1-body], [data-l1-json], [data-scene-body]').forEach(function (el) {
        if (locked) { el.disabled = true; el.setAttribute('data-uncertain-lock', 'true'); }
        else if (el.hasAttribute('data-uncertain-lock')) { el.disabled = false; el.removeAttribute('data-uncertain-lock'); }
      });
    });
  }

  function draftSnapshot(draft) {
    return draft && draft.type === 'l1'
      ? { type: 'l1', document: cloneValue(draft.document), sceneBodies: cloneValue(draft.sceneBodies || {}) }
      : { type: 'legacy', fields: cloneValue(draft && draft.fields || {}) };
  }

  function l1RequestEdit(draft, item) {
    if (draft.invalidJson) throw new Error('L1 JSON 尚未解析成功，修复后才能保存。');
    var edit = { document: cloneValue(draft.document) };
    if (itemIsDiary(item)) edit.scene_bodies = cloneValue(draft.sceneBodies || {});
    return edit;
  }

  function remapL1IdentityReferences(document, mapping) {
    function map(value) { return typeof value === 'string' && mapping[value] ? mapping[value] : value; }
    function visit(node, key) {
      if (Array.isArray(node)) {
        if (key === 'statement_refs' || key === 'shared_statement_refs') {
          return node.map(map);
        }
        return node.map(function (child) { return visit(child, ''); });
      }
      if (!node || typeof node !== 'object') return node;
      Object.keys(node).forEach(function (childKey) {
        var value = node[childKey];
        if (['entity_id', 'statement_id', 'scene_id', 'relation_id'].includes(childKey)) node[childKey] = map(value);
        else if (childKey === 'object_id' && node.object_type) node[childKey] = map(value);
        else if (childKey === 'value' && node.state === 'known') node[childKey] = map(value);
        else node[childKey] = visit(value, childKey);
      });
      return node;
    }
    return visit(document, '');
  }

  function l1ReadbackMatches(submitted, readback) {
    if (!submitted || !readback || !submitted.narrative || !readback.narrative) return false;
    var expected = cloneValue(submitted);
    var mapping = {};
    [['entities', 'entity_id'], ['statements', 'statement_id'], ['scenes', 'scene_id'], ['relations', 'relation_id']].forEach(function (entry) {
      var module = entry[0], field = entry[1];
      var before = submitted[module] || [];
      var after = readback[module] || [];
      for (var index = 0; index < Math.min(before.length, after.length); index++) {
        if (before[index] && after[index] && before[index][field] && after[index][field]) mapping[before[index][field]] = after[index][field];
      }
    });
    expected = remapL1IdentityReferences(expected, mapping);
    if (!sameValue(expected.evidence_bindings, readback.evidence_bindings)) return false;
    var managedEntry = ['parent_ref', 'document_group_id', 'derived_from_revision', 'scene_ref', 'shared_statement_refs'];
    if (!expected.entry || !readback.entry || expected.entry.form !== readback.entry.form ||
        !managedEntry.every(function (key) { return sameValue(expected.entry[key], readback.entry[key]); })) return false;
    expected.generation = cloneValue(readback.generation);
    (expected.scenes || []).forEach(function (scene, index) {
      if (readback.scenes && readback.scenes[index]) scene.generation = cloneValue(readback.scenes[index].generation);
    });
    (expected.relations || []).forEach(function (relation, index) {
      if (readback.relations && readback.relations[index]) relation.origin = readback.relations[index].origin;
    });
    return sameValue(expected, readback);
  }

  function rebaseL1Draft(savedItem, submitted, currentDraft) {
    var savedDocument = itemL1Document(savedItem);
    if (!savedDocument) throw new Error('保存回执未读回完整L1文档，草稿保留并等待重试核对。');
    var nextDocument = cloneValue(savedDocument);
    var mapping = {};
    [['entities', 'entity_id'], ['statements', 'statement_id'], ['scenes', 'scene_id'], ['relations', 'relation_id']].forEach(function (entry) {
      var module = entry[0], field = entry[1];
      var before = submitted.document && submitted.document[module] || [];
      var after = savedDocument[module] || [];
      for (var index = 0; index < Math.min(before.length, after.length); index++) {
        if (before[index] && after[index] && before[index][field] && after[index][field]) mapping[before[index][field]] = after[index][field];
      }
      if (!sameValue(currentDraft.document && currentDraft.document[module], submitted.document && submitted.document[module])) {
        nextDocument[module] = cloneValue(currentDraft.document[module] || []);
      }
    });
    Object.keys(currentDraft.document || {}).forEach(function (key) {
      if (['generation', 'evidence_bindings', 'entry'].includes(key)) return;
      if (!sameValue(currentDraft.document[key], submitted.document && submitted.document[key])) {
        nextDocument[key] = cloneValue(currentDraft.document[key]);
      }
    });
    nextDocument = remapL1IdentityReferences(nextDocument, mapping);
    nextDocument.generation = cloneValue(savedDocument.generation);
    nextDocument.evidence_bindings = cloneValue(savedDocument.evidence_bindings);
    nextDocument.entry = cloneValue(savedDocument.entry);

    var savedSceneBodies = itemSceneBodies(savedItem);
    var nextSceneBodies = cloneValue(savedSceneBodies);
    var oldScenes = submitted.document && submitted.document.scenes || [];
    var newScenes = savedDocument.scenes || [];
    var sceneMap = {};
    for (var i = 0; i < Math.min(oldScenes.length, newScenes.length); i++) {
      if (oldScenes[i] && newScenes[i]) sceneMap[oldScenes[i].scene_id] = newScenes[i].scene_id;
    }
    Object.keys(currentDraft.sceneBodies || {}).forEach(function (sceneId) {
      if (!sameValue(currentDraft.sceneBodies[sceneId], (submitted.sceneBodies || {})[sceneId])) {
        var targetId = sceneMap[sceneId] || sceneId;
        nextSceneBodies[targetId] = currentDraft.sceneBodies[sceneId];
      }
    });
    return {
      type: 'l1', baseDocument: savedDocument, document: nextDocument,
      rawJson: currentDraft.mode === 'json' ? JSON.stringify(nextDocument, null, 2) : JSON.stringify(nextDocument, null, 2),
      sceneBodies: nextSceneBodies, baseSceneBodies: savedSceneBodies,
      invalidJson: false, mode: currentDraft.mode || 'body',
      revision: Number(savedItem.authority_revision), expanded: currentDraft.expanded,
    };
  }

  function updateCardFromSavedItem(card, item, draft) {
    if (!card) return;
    card.setAttribute('data-revision', String(item.authority_revision));
    var l1Document = itemL1Document(item);
    if (l1Document) {
      var bodyEl = card.querySelector('.chat-memory-card-body');
      var titleEl = card.querySelector('.chat-memory-card-head strong');
      if (bodyEl) bodyEl.textContent = l1Document.narrative.body;
      if (titleEl) titleEl.textContent = l1Document.narrative.title;
      var semantics = card.querySelector('.chat-memory-semantics');
      if (semantics) semantics.innerHTML = renderL1ReviewDetails(item, l1Document);
      var editor = card.querySelector('.chat-memory-l1-editor');
      if (editor) {
        var mode = draft && draft.type === 'l1' ? draft.mode : 'body';
        editor.setAttribute('data-l1-editor-mode', mode);
        card.setAttribute('data-l1-editor-mode', mode);
        var bodyView = card.querySelector('[data-l1-body-view]');
        var jsonView = card.querySelector('[data-l1-json-view]');
        if (bodyView) bodyView.hidden = mode !== 'body';
        if (jsonView) jsonView.hidden = mode !== 'json';
        if (draft && draft.type === 'l1') {
          syncL1Form(card, draft.document, draft.sceneBodies);
          var json = card.querySelector('[data-l1-json]');
          if (json) json.value = draft.rawJson || JSON.stringify(draft.document, null, 2);
          var error = card.querySelector('[data-l1-json-error]');
          if (error) error.textContent = draft.invalidJson ? 'JSON 尚未解析成功；草稿已保留，修复 JSON 后才能保存或采用。' : '';
        } else {
          syncL1Form(card, l1Document, itemSceneBodies(item));
          var savedJson = card.querySelector('[data-l1-json]');
          if (savedJson) savedJson.value = JSON.stringify(l1Document, null, 2);
        }
      }
      return;
    }
    var savedFields = itemFields(item);
    Object.keys(savedFields).forEach(function (field) {
      var input = card.querySelector('[data-field="' + field + '"]');
      if (input) input.value = Array.isArray(savedFields[field]) ? savedFields[field].join(', ') : String(savedFields[field]);
    });
    if (draft && draft.type !== 'l1') {
      Object.keys(draft.fields || {}).forEach(function (field) {
        var input = card.querySelector('[data-field="' + field + '"]');
        if (input) input.value = Array.isArray(draft.fields[field]) ? draft.fields[field].join(', ') : String(draft.fields[field]);
      });
    }
    var legacyBody = card.querySelector('.chat-memory-card-body');
    if (legacyBody) legacyBody.textContent = item.candidate.proposed_memory || item.candidate.content;
    var legacyTitle = card.querySelector('.chat-memory-card-head strong');
    if (legacyTitle) legacyTitle.textContent = savedFields.title;
  }

  async function saveMemoryDraft(id) {
    if (pendingMemoryAction(id)) throw new Error('采用或审核结果尚未确认，请先重试原操作；暂不修改候选。');
    var draft = memoryDrafts.get(id);
    if (!draft) return;
    var key = 'edit:' + id;
    var item = memoryItems.get(id);
    var priorRequest = memoryRequests.get(key);
    var snapshot = priorRequest && priorRequest.snapshot ? priorRequest.snapshot : draftSnapshot(draft);
    if (!priorRequest && draft.type === 'l1' && draft.invalidJson) throw new Error('L1 JSON 尚未解析成功，修复后才能保存。');
    var requestEdit = priorRequest ? null : (draft.type === 'l1' ? l1RequestEdit(draft, item) : draft.edits);
    var body = priorRequest ? priorRequest.body : stableMemoryRequest(key, {
      candidate_id: id, edit: requestEdit, expected_revision: draft.revision,
    });
    var request = memoryRequests.get(key);
    if (request && !request.snapshot) { request.snapshot = snapshot; persistMemoryState(); }
    snapshot = request && request.snapshot || snapshot;
    var res = await authFetch(dailyChatMemoryApiBase() + '/api/daily-chat-memory/edit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!res) throw new Error('未确认保存结果，请恢复连接后重试；草稿已保留。');
    var data = await res.json();
    if (!res.ok) {
      if (res.status >= 400 && res.status < 500) { memoryRequests.delete(key); persistMemoryState(); }
      throw new Error(data.error === 'revision_conflict' ? '候选已在其他页面修改，请核对新版本；草稿已保留。' : (data.error || '保存失败'));
    }
    if (data.status !== 'saved' || data.adopted !== false || !data.item) throw new Error('保存回执不完整，请重试核对。');
    if (!['pending', 'deferred'].includes(data.item.status) || Number(data.item.authority_revision) !== Number(body.expected_revision) + 1) {
      throw new Error('保存回执中的候选状态或CAS版本不一致；草稿已保留。');
    }
    if (draft.type === 'l1') {
      var readback = itemL1Document(data.item);
      if (!readback || !l1ReadbackMatches(body.edit.document, readback)) {
        throw new Error('保存回执没有读回一致的完整L1正文与语义；草稿已保留。');
      }
    }
    memoryItems.set(id, data.item);
    memoryRequests.delete(key);
    var card = Array.from(document.querySelectorAll('.chat-memory-card')).find(function (c) { return c.getAttribute('data-candidate-id') === id; });
    var currentDraft = memoryDrafts.get(id);
    if (!currentDraft || sameDraft(currentDraft, snapshot)) {
      memoryDrafts.delete(id);
      currentDraft = null;
    } else if (currentDraft.type === 'l1' && snapshot.type === 'l1') {
      currentDraft = rebaseL1Draft(data.item, snapshot, currentDraft);
      if (l1DraftChanged(currentDraft)) memoryDrafts.set(id, currentDraft);
      else { memoryDrafts.delete(id); currentDraft = null; }
    } else if (currentDraft.type !== 'l1' && snapshot.type !== 'l1') {
      var savedFields = itemFields(data.item);
      var lateFields = {};
      Object.keys(currentDraft.fields || {}).forEach(function (field) {
        lateFields[field] = sameValue(currentDraft.fields[field], snapshot.fields[field])
          ? savedFields[field] : currentDraft.fields[field];
      });
      var lateEdits = {};
      Object.keys(lateFields).forEach(function (field) {
        if (!sameValue(lateFields[field], savedFields[field])) lateEdits[field] = lateFields[field];
      });
      if (Object.keys(lateEdits).length) {
        currentDraft = { type: 'legacy', base: savedFields, fields: lateFields, edits: lateEdits,
          revision: Number(data.item.authority_revision), expanded: currentDraft.expanded };
        memoryDrafts.set(id, currentDraft);
      } else { memoryDrafts.delete(id); currentDraft = null; }
    }
    persistMemoryState();
    updateCardFromSavedItem(card, data.item, currentDraft);
  }

  async function saveDailyChatMemoryEdit(button) {
    if (memoryBusy) return;
    captureMemoryDrafts();
    var card = button.closest('.chat-memory-card');
    var id = card.getAttribute('data-candidate-id');
    if (!memoryDrafts.has(id)) { setDailyChatMemoryMessage('没有未保存的修改。', 'ok'); return; }
    memoryBusy = true;
    setMemoryControlsBusy(true);
    try {
      await saveMemoryDraft(id);
      setDailyChatMemoryMessage(memoryDrafts.has(id)
        ? '已保存提交时的版本；提交期间新增的修改仍留在草稿。候选仍待审核，没有采用入库。'
        : '修改已保存，仍待审核；没有采用入库。', 'ok');
    } catch (e) { setDailyChatMemoryMessage(e.message, 'error'); }
    finally { memoryBusy = false; setMemoryControlsBusy(false); }
  }

  function setMemoryControlsBusy(value) {
    document.querySelectorAll('#daily-chat-memory-pending button, #daily-chat-memory-pending select, #daily-chat-memory-pending input[type="checkbox"]').forEach(function (el) {
      if (value) { el.setAttribute('data-was-disabled', el.disabled && !el.hasAttribute('data-uncertain-lock') ? 'true' : 'false'); el.disabled = true; }
      else {
        el.disabled = el.getAttribute('data-was-disabled') === 'true';
        el.removeAttribute('data-was-disabled');
        el.removeAttribute('data-uncertain-lock');
      }
    });
    if (!value) syncUncertainLocks();
  }

  function discardDailyChatMemoryDraft(button) {
    if (memoryBusy) return;
    var id = button.closest('.chat-memory-card').getAttribute('data-candidate-id');
    if (memoryRequests.has('edit:' + id) || pendingMemoryAction(id)) { setDailyChatMemoryMessage('操作结果尚未确认，请先重试原操作。', 'error'); return; }
    if (!confirm('放弃这条未保存修改，重新读取服务器版本？')) return;
    memoryDrafts.delete(id);
    persistMemoryState();
    var card = button.closest('.chat-memory-card');
    card.remove();
    loadDailyChatMemoryPending();
  }

  if (document.addEventListener) document.addEventListener('input', function (event) {
    var card = event.target.closest && event.target.closest('.chat-memory-card');
    if (card) captureMemoryDraft(card);
  });
  if (window.addEventListener) window.addEventListener('beforeunload', function (event) {
    captureMemoryDrafts();
    if (memoryDrafts.size || memoryBusy || memoryRequests.size) { event.preventDefault(); event.returnValue = ''; }
  });
  function dailyChatMemoryApiBase() {
    return typeof BASE !== 'undefined' ? BASE : '';
  }

  function setDailyChatMemoryMessage(message, tone) {
    var el = document.getElementById('daily-chat-memory-message');
    if (!el) return;
    el.textContent = message || '';
    el.classList.remove('ok', 'error');
    if (tone) el.classList.add(tone);
  }

  function newDailyChatMemoryRequestId() {
    return 'rq_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 10);
  }

  var SOFT_FLAG_LABELS = {
    low_confidence: '低置信度',
    possibly_generic: '可能太泛',
    possibly_transient: '可能只当时有效',
    possible_duplicate: '疑似重复',
    excerpt_overlap: '建议与原文重叠',
    weak_source_support: '来源支持较弱',
    previously_rejected_similar: '曾拒绝过相似内容',
    needs_owner_edit: '需要先编辑再写入',
  };

  var SOFT_FLAG_HINTS = {
    low_confidence: '模型把握不高，主人判断是否值得保存。',
    possibly_generic: '内容可能比较泛泛，建议编辑后再写入。',
    possibly_transient: '可能只在当时有效，不一定是长期记忆。',
    possible_duplicate: '与已有的已确认记忆相似，主人确认是否重复。',
    excerpt_overlap: '建议记忆与来源片段高度重叠，请先改写成可读记忆。',
    weak_source_support: '来源支撑较弱（例如只有助手单方表述），主人判断。',
    previously_rejected_similar: '之前拒绝过类似候选，但这次来源/内容不同，可重新考虑。',
    needs_owner_edit: '建议记忆基本是原文照抄，编辑成可读记忆后才能写入。',
  };

  function legacyNoOriginal(item) {
    var display = item && item.display ? item.display : {};
    return Boolean(display.legacy_no_original);
  }

  function confirmBlocked(item) {
    var display = item && item.display ? item.display : {};
    return Boolean(display.confirm_blocked);
  }

  function needsOwnerEdit(item) {
    var display = item && item.display ? item.display : {};
    return Boolean(display.needs_owner_edit);
  }

  function hasSourcePreview(item) {
    var display = item && item.display ? item.display : {};
    return Boolean(display.has_source_preview);
  }

  function renderSoftFlagChips(flags) {
    if (!flags || !flags.length) return '';
    return '<div class="chat-memory-flags">' + flags.map(function (flag) {
      var label = SOFT_FLAG_LABELS[flag] || flag;
      var hint = SOFT_FLAG_HINTS[flag] || '';
      return '<span class="chat-memory-flag" title="' + escAttr(hint) + '">' + esc(label) + '</span>';
    }).join('') + '</div>';
  }

  function displayValue(value) {
    if (value == null || value === '') return '—';
    if (typeof value === 'string') return value;
    try { return JSON.stringify(value); } catch (_) { return String(value); }
  }

  function renderL1ReviewDetails(item, document) {
    var candidate = item && item.candidate && typeof item.candidate === 'object' ? item.candidate : {};
    var metadata = candidate.metadata && typeof candidate.metadata === 'object' ? candidate.metadata : {};
    var bundle = candidate.diary_l1_bundle || metadata.diary_l1_bundle || item && item.diary_l1_bundle || {};
    var display = item && item.display && typeof item.display === 'object' ? item.display : {};
    var sourceState = candidate.l0_source_state || candidate.source_state || candidate.source_status ||
      item && (item.source_state || item.source_status) || display.source_state || candidate.source_verification || '未提供';
    var bindings = Array.isArray(document.evidence_bindings) ? document.evidence_bindings : [];
    var sourceRefs = candidate.source_refs || item && item.source_refs || [];
    if (!Array.isArray(sourceRefs)) sourceRefs = [sourceRefs];
    var bindingRows = bindings.map(function (binding) {
      if (!binding || typeof binding !== 'object') return '<li>' + esc(displayValue(binding)) + '</li>';
      var label = [binding.source_namespace || binding.namespace || binding.source_type || binding.type,
        binding.source_id || binding.id, binding.source_revision || binding.revision,
        binding.author_ref || binding.speaker_ref, binding.text_view_id,
        binding.content_sha256 || binding.source_content_sha256].filter(Boolean).join(' · ');
      return '<li>' + esc(label || displayValue(binding)) + '</li>';
    }).join('');
    var refRows = sourceRefs.map(function (value) { return '<li>' + esc(displayValue(value)) + '</li>'; }).join('');
    var generation = document.generation || candidate.generation || {};
    var generationRows = Object.keys(generation).sort().map(function (key) {
      return '<li><strong>' + esc(key) + '：</strong>' + esc(displayValue(generation[key])) + '</li>';
    }).join('');
    var coverage = bundle && bundle.coverage || candidate.coverage || item && item.coverage;
    return '<details class="chat-memory-excerpt-details chat-memory-l1-details">' +
      '<summary>来源状态与生成版本</summary>' +
      '<p><strong>来源状态：</strong>' + esc(displayValue(sourceState)) + '</p>' +
      (coverage ? '<p><strong>覆盖：</strong>' + esc(displayValue(coverage)) + '</p>' : '') +
      '<p><strong>L1来源绑定：</strong>' + bindings.length + ' 项</p>' +
      (bindingRows ? '<ul>' + bindingRows + '</ul>' : '<p>当前文档没有来源绑定；不代表来源已验证。</p>') +
      (refRows ? '<p><strong>候选来源引用：</strong></p><ul>' + refRows + '</ul>' : '') +
      '<p><strong>generation：</strong></p>' +
      (generationRows ? '<ul>' + generationRows + '</ul>' : '<p>未提供生成版本信息。</p>') +
      '</details>';
  }

  function renderL1EditPanel(item, document, draft) {
    var current = draft && draft.type === 'l1' ? draft.document : document;
    var mode = draft && draft.type === 'l1' ? draft.mode : 'body';
    var rawJson = draft && draft.type === 'l1' ? draft.rawJson : JSON.stringify(current, null, 2);
    var sceneBodies = draft && draft.type === 'l1' ? draft.sceneBodies : itemSceneBodies(item);
    var bodyFields = '<label class="chat-memory-edit-field">L1标题' +
      '<input type="text" data-l1-title value="' + escAttr((current.narrative || {}).title || '') + '" />' +
      '</label>' +
      '<label class="chat-memory-edit-field">第一遍完整正文' +
      '<textarea data-l1-body rows="9">' + esc((current.narrative || {}).body || '') + '</textarea>' +
      '</label>';
    var scenes = itemIsDiary(item) ? renderDiarySceneBodies(current, sceneBodies) : '';
    return '<div class="chat-memory-l1-editor" data-l1-editor-mode="' + escAttr(mode) + '">' +
      '<div class="chat-memory-card-actions">' +
        '<button type="button" onclick="toggleDailyChatMemoryL1Mode(this, \'body\')">正文与场景</button>' +
        '<button type="button" onclick="toggleDailyChatMemoryL1Mode(this, \'json\')">完整L1 JSON</button>' +
      '</div>' +
      '<p class="chat-memory-edit-needed">完整L1与正文共用一份草稿。来源绑定、身份、父子引用和generation由服务端保护；修改这些字段会返回明确错误。</p>' +
      '<p data-l1-json-error class="chat-memory-stale">' + (draft && draft.invalidJson ? 'JSON 尚未解析成功；草稿已保留，修复 JSON 后才能保存或采用。' : '') + '</p>' +
      '<div data-l1-body-view' + (mode === 'json' ? ' hidden' : '') + '>' + bodyFields + '</div>' +
      '<label class="chat-memory-edit-field" data-l1-json-view' + (mode === 'json' ? '' : ' hidden') + '>完整L1 JSON' +
        '<textarea data-l1-json rows="22" spellcheck="false" style="font-family:ui-monospace,Consolas,monospace;white-space:pre">' + esc(rawJson || '') + '</textarea>' +
        '<small>JSON包含正文、主体、场景、陈述、分类、时间条件、关系、属性与检索字段。保存会完整提交，服务端校验受保护的机器字段。</small>' +
      '</label>' +
      (scenes ? '<div class="chat-memory-l1-scenes"><strong>日记场景正文</strong><div data-l1-scenes>' + scenes + '</div></div>' : '') +
    '</div>';
  }

  function renderDiarySceneBodies(document, sceneBodies, selectedTargets) {
    sceneBodies = sceneBodies && typeof sceneBodies === 'object' ? sceneBodies : {};
    selectedTargets = selectedTargets && typeof selectedTargets === 'object' ? selectedTargets : {};
    var scenes = Array.isArray(document && document.scenes) ? document.scenes : [];
    var known = new Set();
    var html = scenes.map(function (scene) {
      if (!scene || typeof scene !== 'object') return '';
      var id = String(scene.scene_id || '');
      known.add(id);
      return '<label class="chat-memory-edit-field">日记场景正文 · ' + esc(scene.title || id) +
        '<textarea data-scene-body="' + escAttr(id) + '" rows="6">' + esc(Object.prototype.hasOwnProperty.call(sceneBodies, id) ? sceneBodies[id] : '') +
        '</textarea><small>场景正文会随同一候选保存；边界沿用scene身份时保留这段完整正文。</small></label>';
    }).join('');
    Object.keys(sceneBodies).filter(function (id) { return !known.has(id); }).forEach(function (id) {
      var options = scenes.filter(function (scene) { return scene && scene.scene_id; }).map(function (scene) {
        return '<option value="' + escAttr(scene.scene_id) + '"' + (selectedTargets[id] === scene.scene_id ? ' selected' : '') + '>' + esc(scene.title || scene.scene_id) + '</option>';
      }).join('');
      html += '<div class="chat-memory-edit-field chat-memory-edit-needed">未映射的旧场景正文 · ' + esc(id) +
        '<textarea data-orphan-scene-body="' + escAttr(id) + '" rows="6">' + esc(sceneBodies[id]) + '</textarea>' +
        '<div class="chat-memory-card-actions"><select data-orphan-scene-target><option value="">选择要保留正文的当前场景</option>' + options + '</select>' +
        '<button type="button" onclick="mapDailyChatMemorySceneBody(this)">复制到所选场景并移除旧项</button>' +
        '<button type="button" class="danger" onclick="discardDailyChatMemorySceneBody(this)">确认移除旧场景正文</button></div>' +
        '<small>旧正文会一直保留到你明确复制或移除；保存仍会要求每段正文对应当前场景。</small></div>';
    });
    return html;
  }

  function refreshDiarySceneBodies(card, document, sceneBodies) {
    var container = card && card.querySelector('[data-l1-scenes]');
    if (!container) return;
    var selectedTargets = {};
    container.querySelectorAll('[data-orphan-scene-target]').forEach(function (select) {
      var wrapper = select.closest('.chat-memory-edit-field');
      var source = wrapper && wrapper.querySelector('[data-orphan-scene-body]');
      if (source && select.value) selectedTargets[source.getAttribute('data-orphan-scene-body')] = select.value;
    });
    container.innerHTML = renderDiarySceneBodies(document, sceneBodies, selectedTargets);
  }

  function syncL1Form(card, document, sceneBodies) {
    var title = card.querySelector('[data-l1-title]');
    var body = card.querySelector('[data-l1-body]');
    if (title) title.value = String((document.narrative || {}).title || '');
    if (body) body.value = String((document.narrative || {}).body || '');
    refreshDiarySceneBodies(card, document, sceneBodies);
  }

  function ensureL1Draft(card, item) {
    var id = card.getAttribute('data-candidate-id');
    var draft = memoryDrafts.get(id);
    if (!draft || draft.type !== 'l1') {
      var current = itemL1Document(item);
      if (!current) return null;
      var scenes = itemSceneBodies(item);
      draft = { type: 'l1', baseDocument: cloneValue(current), document: cloneValue(current),
        rawJson: JSON.stringify(current, null, 2), sceneBodies: scenes, baseSceneBodies: cloneValue(scenes),
        invalidJson: false, mode: card.getAttribute('data-l1-editor-mode') || 'body',
        revision: Number(card.getAttribute('data-revision')), expanded: true };
      memoryDrafts.set(id, draft);
    }
    return draft;
  }

  function mapDailyChatMemorySceneBody(button) {
    var card = button.closest('.chat-memory-card');
    var item = card && memoryItems.get(card.getAttribute('data-candidate-id'));
    if (!card || !item) return;
    var source = button.closest('.chat-memory-edit-field').querySelector('[data-orphan-scene-body]');
    var target = button.closest('.chat-memory-edit-field').querySelector('[data-orphan-scene-target]');
    var sourceId = source && source.getAttribute('data-orphan-scene-body');
    var sourceBody = source && source.value;
    var targetId = target && target.value;
    captureMemoryDraft(card);
    var draft = ensureL1Draft(card, item);
    if (!draft || draft.invalidJson) { setDailyChatMemoryMessage('先修复完整L1 JSON，再映射场景正文。', 'error'); return; }
    if (!sourceId || !targetId) { setDailyChatMemoryMessage('请选择要保留正文的当前场景。', 'error'); return; }
    draft.sceneBodies[targetId] = sourceBody;
    delete draft.sceneBodies[sourceId];
    refreshDiarySceneBodies(card, draft.document, draft.sceneBodies);
    captureMemoryDraft(card);
    setDailyChatMemoryMessage('旧场景正文已复制到当前场景草稿；保存后仍待审核。', 'ok');
  }

  function discardDailyChatMemorySceneBody(button) {
    var card = button.closest('.chat-memory-card');
    var item = card && memoryItems.get(card.getAttribute('data-candidate-id'));
    if (!card || !item || !confirm('确认从这份待审草稿中移除这段旧场景正文？')) return;
    var source = button.closest('.chat-memory-edit-field').querySelector('[data-orphan-scene-body]');
    var sourceId = source && source.getAttribute('data-orphan-scene-body');
    captureMemoryDraft(card);
    var draft = ensureL1Draft(card, item);
    if (!draft) return;
    if (sourceId) delete draft.sceneBodies[sourceId];
    refreshDiarySceneBodies(card, draft.document, draft.sceneBodies);
    captureMemoryDraft(card);
    setDailyChatMemoryMessage('旧场景正文已从草稿移除；保存仍待审核。', 'ok');
  }

  function toggleDailyChatMemoryL1Mode(button, nextMode) {
    var card = button && button.closest ? button.closest('.chat-memory-card') : null;
    var editor = card && card.querySelector('.chat-memory-l1-editor');
    var item = card && memoryItems.get(card.getAttribute('data-candidate-id'));
    if (!editor || !item || !['body', 'json'].includes(nextMode)) return;
    captureMemoryDraft(card);
    var id = card.getAttribute('data-candidate-id');
    var draft = memoryDrafts.get(id);
    if (draft && draft.type === 'l1' && draft.invalidJson && nextMode !== 'json') {
      setDailyChatMemoryMessage('JSON 尚未解析成功；修复后再切换视图，草稿仍保留。', 'error');
      return;
    }
    var document = draft && draft.type === 'l1' ? draft.document : itemL1Document(item);
    var sceneBodies = draft && draft.type === 'l1' ? draft.sceneBodies : itemSceneBodies(item);
    if (nextMode === 'body') syncL1Form(card, document, sceneBodies);
    var json = card.querySelector('[data-l1-json]');
    if (json && nextMode === 'json' && !(draft && draft.type === 'l1' && draft.invalidJson)) {
      json.value = JSON.stringify(document, null, 2);
    }
    var bodyView = card.querySelector('[data-l1-body-view]');
    var jsonView = card.querySelector('[data-l1-json-view]');
    if (bodyView) bodyView.hidden = nextMode !== 'body';
    if (jsonView) jsonView.hidden = nextMode !== 'json';
    editor.setAttribute('data-l1-editor-mode', nextMode);
    card.setAttribute('data-l1-editor-mode', nextMode);
    captureMemoryDraft(card);
  }

  function loadDailyChatMemoryPending() {
    var target = document.getElementById('daily-chat-memory-pending');
    if (!target || memoryBusy) return;
    loadMemorySourceTasks(target);
    captureMemoryDrafts();
    var epoch = ++memoryLoadEpoch;
    return authFetch(dailyChatMemoryApiBase() + '/api/daily-chat-memory/pending?limit=100')
      .then(function (res) {
        if (!res) throw new Error('未连接，保留当前内容和草稿');
        if (!res.ok) throw new Error('读取失败 (HTTP ' + res.status + ')');
        return res.json();
      })
      .then(function (data) {
        if (data && data.error) throw new Error(data.error || '读取失败');
        if (epoch !== memoryLoadEpoch || memoryBusy) return;
        restoreMemoryState();
        captureMemoryDrafts();
        var items = (data && data.items) || [];
        var seen = new Set(items.map(function (item) { return item.id; }));
        memoryDrafts.forEach(function (draft, id) {
          if (!seen.has(id) && memoryItems.has(id)) { items.push(Object.assign({}, memoryItems.get(id), { draft_orphan: true })); seen.add(id); }
        });
        memoryRequests.forEach(function (request) {
          (request.body.candidate_ids || []).forEach(function (id) {
            if (!seen.has(id) && memoryItems.has(id)) { items.push(Object.assign({}, memoryItems.get(id), { draft_orphan: true })); seen.add(id); }
          });
        });
        items.forEach(function (item) { memoryItems.set(item.id, item); });
        target.innerHTML = renderDailyChatMemoryPending(items);
        syncUncertainLocks();
      })
      .catch(function (e) {
      if (epoch === memoryLoadEpoch) setDailyChatMemoryMessage('读取失败，草稿保留: ' + e.message, 'error');
      });
  }

  function loadMemorySourceTasks(target) {
    var panel = document.getElementById('memory-source-tasks');
    if (!panel) {
      panel = document.createElement('details'); panel.id = 'memory-source-tasks';
      var summary = document.createElement('summary'); summary.textContent = '行动结果的记忆准备状态'; panel.appendChild(summary);
      var refresh = document.createElement('button'); refresh.textContent = '刷新准备状态';
      refresh.addEventListener('click', function() { loadMemorySourceTasks(target); }); panel.appendChild(refresh);
      var notice = document.createElement('p'); notice.setAttribute('data-source-task-notice', ''); notice.setAttribute('role', 'status'); panel.appendChild(notice);
      var content = document.createElement('div'); content.setAttribute('data-source-task-list', ''); panel.appendChild(content);
      target.parentNode.insertBefore(panel, target);
    }
    var list = panel.querySelector('[data-source-task-list]');
    var notice = panel.querySelector('[data-source-task-notice]');
    var epoch = ++memorySourceLoadEpoch; notice.textContent = '正在刷新准备状态…';
    authFetch(dailyChatMemoryApiBase() + '/api/memory-authority/source-tasks?limit=100').then(function(res) {
      if (!res || !res.ok) throw new Error('暂不可读'); return res.json();
    }).then(function(data) {
      if (epoch !== memorySourceLoadEpoch) return;
      if (!data || !Array.isArray(data.tasks)) throw new Error('回执格式不符');
      notice.textContent = '准备状态已刷新。';
      list.replaceChildren();
      var help = document.createElement('p'); help.textContent = '来源提交与记忆采用分开。后台按预算整理，也可用原补扫入口处理；这里只显示前100条任务。'; list.appendChild(help);
      data.tasks.forEach(function(task) {
        var row = document.createElement('div'); row.className = 'chat-memory-card';
        var status = document.createElement('p');
        var labels = {queued:'等待整理', generating:'处理中', prepared:'候选已生成', failed:'整理失败'};
        status.textContent = (labels[task.phase] || task.phase) + ' · ' + task.action_id; row.appendChild(status);
        if (task.issue) { var issue = document.createElement('code'); issue.textContent = task.issue.code || '未完成'; row.appendChild(issue); }
        if (task.candidate_ids && task.candidate_ids.length) {
          var refs = document.createElement('p'); refs.textContent = '对应候选：' + task.candidate_ids.join('、') + '；采用状态以候选和正式记忆回执为准。'; row.appendChild(refs);
        }
        if (task.phase === 'failed' || task.phase === 'generating') {
          var retry = document.createElement('button'); retry.textContent = task.phase === 'failed' ? '重试整理' : '中断后恢复整理';
          retry.addEventListener('click', async function() {
            if (!window.confirm('重新排队整理此来源；可能再次调用已配置模型，不会重新执行原行动。')) return;
            retry.disabled = true;
            try {
              var res = await authFetch(dailyChatMemoryApiBase() + '/api/memory-authority/source-tasks/' + encodeURIComponent(task.task_id) + '/retry',
                {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({expected_phase:task.phase})});
              if (!res || !res.ok) throw new Error('未重新排队，请刷新核对任务状态');
              loadMemorySourceTasks(target);
            } catch (error) { status.textContent = error.message; retry.disabled = false; }
          }); row.appendChild(retry);
        }
        list.appendChild(row);
      });
    }).catch(function() {
      if (epoch !== memorySourceLoadEpoch) return;
      notice.textContent = '准备状态刷新失败；以下保留上次回执，可能已过期。已有候选和草稿保持。';
    });
  }

  function renderDailyChatMemoryPending(items) {
    if (!items.length) return '<div class="loading">暂无待确认候选。</div>';
    var toolbar = renderDailyChatMemoryToolbar(items);
    var cards = items.map(renderDailyChatMemoryCard).join('');
    return toolbar + '<div class="chat-memory-list" id="daily-chat-memory-cards">' + cards + '</div>';
  }

  function renderDailyChatMemoryToolbar(items) {
    var pendable = items.filter(function (item) { return item.status === 'pending'; });
    if (!pendable.length) return '';
    return '' +
      '<div class="chat-memory-toolbar">' +
        '<label class="chat-memory-batch-select">' +
          '<input type="checkbox" id="daily-chat-memory-select-all" onchange="toggleDailyChatMemorySelectAll(this)" /> 全选' +
        '</label>' +
        '<span class="chat-memory-batch-count" id="daily-chat-memory-batch-count">已选 0 条</span>' +
        '<label class="chat-memory-reject-reason">拒绝原因' +
          '<select id="daily-chat-memory-batch-reason">' +
            '<option value="too_generic">太泛泛</option>' +
            '<option value="not_important">不重要</option>' +
            '<option value="wrong">内容错误</option>' +
            '<option value="duplicate">重复</option>' +
            '<option value="other">其他</option>' +
          '</select>' +
        '</label>' +
        '<label class="chat-memory-reject-note">备注<input type="text" id="daily-chat-memory-batch-note" maxlength="120" placeholder="可选，简短原因" /></label>' +
        '<button type="button" onclick="batchDailyChatMemoryConfirm(this, \'confirm\')">批量写入选中</button>' +
        '<button type="button" class="danger" onclick="batchDailyChatMemoryConfirm(this, \'reject\')">批量拒绝选中</button>' +
        '<button type="button" onclick="batchDailyChatMemoryConfirm(this, \'defer\')">批量暂缓选中</button>' +
      '</div>';
  }

  function renderDailyChatMemoryCard(item) {
    var candidate = Object.assign({}, item.candidate || {});
    var id = item.id || '';
    var draft = memoryDrafts.get(id);
    var revision = draft ? draft.revision : Number(item.authority_revision || candidate.authority_revision || 0);
    var l1Document = itemL1Document(item);
    var l1Draft = draft && draft.type === 'l1' ? draft : null;
    var fields = draft && draft.type !== 'l1' ? draft.fields : itemFields(item);
    if (!l1Document) Object.assign(candidate, fields, { proposed_memory: fields.content });
    var legacy = legacyNoOriginal(item);
    var blocked = confirmBlocked(item);
    var editRequired = needsOwnerEdit(item);
    var hasSource = hasSourcePreview(item);
    var excerpt = String(candidate.original_excerpt || '').trim() || (legacy ? '历史候选无原文摘录' : '（无来源片段）');
    var proposed = String((l1Draft ? l1Draft.document : l1Document || {}).narrative
      ? (l1Draft ? l1Draft.document : l1Document).narrative.body
      : (candidate.proposed_memory || candidate.content || '')).trim() || '（缺少建议记忆）';
    var cardTitle = String((l1Draft ? l1Draft.document : l1Document || {}).narrative
      ? (l1Draft ? l1Draft.document : l1Document).narrative.title
      : (candidate.title || id));
    var flags = (item.display && item.display.soft_flags) || candidate.soft_flags || [];
    var sourceParts = [];
    if (candidate.source_event_ids && candidate.source_event_ids.length) {
      sourceParts.push('事件 ' + candidate.source_event_ids.slice(0, 12).join(', '));
    }
    if (candidate.source_turn_ids && candidate.source_turn_ids.length) {
      sourceParts.push('轮次 ' + candidate.source_turn_ids.slice(0, 12).join(', '));
    }
    var sourceText = sourceParts.length ? sourceParts.join(' · ') : '（无来源）';
    var sourceHash = String(candidate.source_hash || '').slice(0, 8);
    var staleNote = item.stale
      ? '<div class="chat-memory-stale">来源无法核对（' + esc(String(item.stale_reason || 'candidate_source_invalid')) + '），不可写入，请拒绝。</div>'
      : '';
    var blockedNote = blocked && !item.stale
      ? '<div class="chat-memory-stale">' + esc(String(item.display && item.display.confirm_blocked_reason || '候选无法核对，不可写入，请拒绝。')) + '</div>'
      : '';
    var editNote = editRequired && !blocked
      ? '<div class="chat-memory-edit-needed">建议记忆与原文高度重叠：请先编辑成可读记忆后再写入。</div>'
      : '';
    var confirmDisabled = blocked ? ' disabled title="候选无法核对，不可写入"' : '';
    var sourcePreviewButton = hasSource
      ? '<button type="button" class="chat-memory-source-btn" onclick="toggleDailyChatMemorySource(this, \'' + jsString(id) + '\')">查看完整原文</button>'
      : '';
    return '' +
      '<div class="chat-memory-card' + (editRequired ? ' chat-memory-card-edit-needed' : '') + '" data-candidate-id="' + escAttr(id) + '" data-revision="' + escAttr(revision) + '" data-editing="' + (draft && draft.expanded ? 'true' : 'false') + '"' + (l1Document ? ' data-l1-editor-mode="' + escAttr(l1Draft && l1Draft.mode || 'body') + '"' : '') + (editRequired ? ' data-needs-edit="true"' : '') + '>' +
        '<div class="chat-memory-card-head">' +
          '<label class="chat-memory-select">' +
            '<input type="checkbox" data-select="' + escAttr(id) + '" onchange="refreshDailyChatMemorySelection()" />' +
          '</label>' +
          '<strong>' + esc(cardTitle || id) + '</strong>' +
          renderSoftFlagChips(flags) +
        '</div>' +
        '<div class="chat-memory-card-grid">' +
          '<div class="chat-memory-column chat-memory-column-primary">' +
            '<div class="chat-memory-column-title">建议记忆</div>' +
            '<div class="chat-memory-card-body">' + esc(proposed) + '</div>' +
          '</div>' +
          '<div class="chat-memory-column">' +
            '<div class="chat-memory-column-title">来源摘录</div>' +
            '<details class="chat-memory-excerpt-details"' + (legacy ? ' open' : '') + '>' +
              '<summary>' + (legacy ? '历史候选无原文摘录' : '展开来源摘录') + '</summary>' +
              '<div class="chat-memory-excerpt' + (legacy ? ' chat-memory-legacy' : '') + '">' + esc(excerpt) + '</div>' +
            '</details>' +
            '<div class="chat-memory-card-source">' + esc(sourceText) + '</div>' +
            '<div class="chat-memory-card-meta">' +
              esc((candidate.kind || candidate.candidate_type || 'memory') + ' · ' + (item.date || '') + ' · confidence ' + (candidate.confidence == null ? '' : candidate.confidence)) +
              (sourceHash ? ' · src ' + esc(sourceHash) : '') +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div class="chat-memory-source-panel" hidden>' +
          '<div class="chat-memory-source-loading">读取完整原文...</div>' +
        '</div>' +
        '<div class="chat-memory-semantics">' + (l1Document ? renderL1ReviewDetails(item, l1Document) : renderMemorySemantics(item, draft)) + '</div>' + staleNote + blockedNote + editNote +
        (draft ? '<p class="chat-memory-edit-needed">有未保存修改' + (item.draft_orphan || revision !== item.authority_revision ? '；服务器版本或状态已变化，请核对后再操作' : '') + '。</p>' : '') +
        '<div class="chat-memory-edit-panel"' + (draft && draft.expanded ? '' : ' hidden') + '>' +
          (l1Document ? renderL1EditPanel(item, l1Document, l1Draft) :
            '<label class="chat-memory-edit-field">标题' +
              '<input type="text" data-field="title" maxlength="100" value="' + escAttr(candidate.title || id) + '" />' +
            '</label>' +
            '<label class="chat-memory-edit-field">正文（建议记忆，写入记忆桶的内容）' +
              '<textarea data-field="content" maxlength="12000" rows="6">' + esc(proposed) + '</textarea><small>最多 12,000 字符；保存修改不会采用入库。离开页面前请保存。</small>' +
            '</label>' +
            '<div class="chat-memory-edit-grid">' +
              '<label class="chat-memory-edit-field">类型' +
                '<input type="text" data-field="kind" value="' + escAttr(candidate.kind || 'memory') + '" />' +
              '</label>' +
              '<label class="chat-memory-edit-field">域' +
                '<input type="text" data-field="domain" value="' + escAttr(listText(candidate.domain)) + '" />' +
              '</label>' +
              '<label class="chat-memory-edit-field">标签' +
                '<input type="text" data-field="tags" value="' + escAttr(listText(candidate.tags)) + '" />' +
              '</label>' +
              '<label class="chat-memory-edit-field">重要度' +
                '<input type="number" min="1" max="10" data-field="importance" value="' + escAttr(candidate.importance || '') + '" />' +
              '</label>' +
              '<label class="chat-memory-edit-field">置信度' +
                '<input type="number" min="0" max="1" step="0.01" data-field="confidence" value="' + escAttr(candidate.confidence) + '" />' +
              '</label>' +
              '<label class="chat-memory-edit-field">事件日期（可选；不是入库日期）' +
                '<input type="date" data-field="event_date" value="' + escAttr(fields.event_date) + '" /><small>未改此字段会保留原时间依据；改正文后旧时间需复核。</small>' +
              '</label>' +
            '</div>') +
        '</div>' +
        '<div class="chat-memory-card-actions">' +
          '<button type="button" onclick="toggleDailyChatMemoryEdit(this)">编辑</button>' +
          '<button type="button" onclick="saveDailyChatMemoryEdit(this)">保存修改（仍待审核）</button>' +
          '<button type="button" onclick="discardDailyChatMemoryDraft(this)">放弃修改／载入新版</button>' +
          sourcePreviewButton +
          '<button type="button"' + confirmDisabled + ' onclick="confirmDailyChatMemory(this, \'' + jsString(id) + '\', \'confirm\')">采用并入库</button>' +
          '<button type="button" onclick="confirmDailyChatMemory(this, \'' + jsString(id) + '\', \'defer\')">暂缓</button>' +
          '<label class="chat-memory-reject-reason">拒绝原因' +
            '<select id="reject-reason-' + escAttr(id) + '">' +
              '<option value="too_generic">太泛泛</option>' +
              '<option value="not_important">不重要</option>' +
              '<option value="wrong">内容错误</option>' +
              '<option value="duplicate">重复</option>' +
              '<option value="other" selected>其他</option>' +
            '</select>' +
          '</label>' +
          '<input type="text" class="chat-memory-reject-note-inline" id="reject-note-' + escAttr(id) + '" maxlength="120" placeholder="备注(可选)" />' +
          '<button type="button" class="danger" onclick="confirmDailyChatMemory(this, \'' + jsString(id) + '\', \'reject\')">拒绝</button>' +
        '</div>' +
      '</div>';
  }

  function renderMemorySemantics(item, draft) {
    var stored = (item.candidate || {}).semantic_annotations;
    if (!stored || stored.version !== 'memory-semantics-v1') return '';
    var summary = (item.display || {}).semantics || {};
    if (summary.state !== 'current' || (draft && draft.edits && Object.prototype.hasOwnProperty.call(draft.edits, 'content'))) {
      return '<p class="chat-memory-edit-needed">正文已改变：旧语义标注待复核，不继续作为当前标注使用。</p>';
    }
    var kinds = { preference: '偏好', boundary: '边界', commitment: '承诺', shared_experience: '共同经历', key_event: '重要事件', reflection: '理解与感想', project_state: '项目状态', identity: '身份提议' };
    var bases = { owner_statement: '主人原话', assistant_commitment: '顾衍的承诺原话', shared_experience: '经历陈述', assistant_interpretation: '顾衍当时的理解' };
    var states = { stated: '当时陈述', changed: '当时表达了变化', cancelled: '当时表达了取消', fulfilled: '当时表达了完成', uncertain: '不确定' };
    var qualifiers = { topic: '主题', value: '取值／对象', conditions: '适用条件', exceptions: '例外', valid_time: '适用时间原话' };
    var rows = (Array.isArray(stored.items) ? stored.items : []).filter(function (a) { return a && typeof a === 'object'; }).slice(0, 8).map(function (annotation) {
      var role = (annotation.subject_ref || {}).role;
      var subject = role === 'user' ? '主人' : (role === 'assistant' ? '顾衍' : '主体待核');
      var parts = [subject + '的' + (kinds[annotation.semantic_kind] || '语义标注'), bases[annotation.assertion_basis] || '证据待核', states[annotation.assertion_state] || '当时陈述'];
      Object.keys(qualifiers).forEach(function (key) {
        var value = (annotation.qualifiers || {})[key];
        if (value) parts.push(qualifiers[key] + '：' + value);
      });
      if (annotation.conditions_status === 'not_stated') parts.push('条件未说明，不代表永久或无条件');
      var evidence = (annotation.evidence_refs || [])[0] || {};
      if (evidence.event_id) parts.push((evidence.namespace === 'canonical_event' ? 'canonical 消息 ' : '原始来源 ') + evidence.event_id + (evidence.version_id ? '／' + evidence.version_id : ''));
      return '<li>' + esc(parts.join(' · ')) + '</li>';
    }).join('');
    var links = (Array.isArray(stored.links) ? stored.links : []).filter(function (l) { return l && typeof l === 'object'; }).slice(0, 12).map(function (link) {
      return '<li>' + esc('与 ' + link.target_candidate_id + ' 有共同来源；关联时状态 ' + link.target_status_at_link + '。不代表同一事实，未覆盖正文。') + '</li>';
    }).join('');
    var unresolved = Number(summary.unresolved_count || 0);
    var coverage = Array.isArray(stored.source_coverage) ? stored.source_coverage : [];
    var bridged = coverage.filter(function (s) { return s && s.bridge_status === 'exact_runtime_bridge'; }).length;
    return '<details class="chat-memory-excerpt-details"><summary>语义与来源：' + Number(summary.annotation_count || 0) + ' 项标注 · ' + Number(summary.source_link_count || 0) + ' 个同源关联</summary>' +
      '<p>引文位置已核对不等于事实已确认；这些标注不会创建任务，也不能证明任务当前已完成。</p>' +
      (rows ? '<ul>' + rows + '</ul>' : '<p>未附语义标注，仍可按原规则审核记忆。</p>') +
      (links ? '<ul>' + links + '</ul>' : '<p>尚无可证明的同源关联；不按相似文字合并。</p>') +
      (bridged ? '<p>' + bridged + ' 条来源已按消息身份、版本和完整内容校验值跨入口对应。</p>' : '') +
      (unresolved ? '<p>' + unresolved + ' 项标注未通过原句／说话人核对，未作为有效标注保存。</p>' : '') + '</details>';
  }

  function renderDailyChatMemorySourcePanel(data) {
    var blocks = [];
    (data.turns || []).forEach(function (turn) {
      if (turn.text) {
        blocks.push(renderSourceBlock('轮次 ' + turn.id + ' · 我', turn.text, turn.truncated, turn.continue_after, 'turn', turn.id));
      } else {
        if (turn.user_text) blocks.push(renderSourceBlock('轮次 ' + turn.id + ' · 我', turn.user_text, false, -1, null, null));
        if (turn.assistant_text) blocks.push(renderSourceBlock('轮次 ' + turn.id + ' · 润润', turn.assistant_text, false, -1, null, null));
      }
    });
    (data.events || []).forEach(function (event) {
      if (event.text) {
        var roleLabel = event.role === 'user' ? '我' : '润润';
        blocks.push(renderSourceBlock('事件 ' + event.id + ' · ' + roleLabel, event.text, event.truncated, event.continue_after, 'event', event.id));
      }
    });
    if (!blocks.length) {
      return '<div class="chat-memory-source-empty">未找到可展开的完整原文（来源事件可能已被清理或撤回）。</div>';
    }
    return blocks.join('');
  }

  function renderSourceBlock(label, text, truncated, continueAfter, sourceKind, sourceId) {
    var more = truncated && continueAfter >= 0
      ? '<button type="button" class="chat-memory-source-more" data-kind="' + escAttr(sourceKind) + '" data-sid="' + escAttr(sourceId) + '" data-offset="' + escAttr(continueAfter) + '" onclick="loadDailyChatMemorySourceMore(this)">继续加载完整原文</button>'
      : '';
    return '<div class="chat-memory-source-block"><span class="chat-memory-source-role">' + esc(label) + '</span>' + esc(text) + more + '</div>';
  }

  async function loadDailyChatMemorySourceMore(button) {
    var kind = button.getAttribute('data-kind');
    var sourceId = button.getAttribute('data-sid');
    var offset = button.getAttribute('data-offset');
    var candidateId = (button.closest('.chat-memory-card') || {}).getAttribute ? button.closest('.chat-memory-card').getAttribute('data-candidate-id') : null;
    if (!candidateId) return;
    button.disabled = true;
    try {
      var url = dailyChatMemoryApiBase() + '/api/daily-chat-memory/source-preview?candidate_id=' + encodeURIComponent(candidateId) +
        '&source_kind=' + encodeURIComponent(kind) + '&source_id=' + encodeURIComponent(sourceId) + '&offset=' + encodeURIComponent(offset);
      var res = await authFetch(url);
      if (!res) return;
      var data = await res.json();
      if (!res.ok) throw new Error(data.error || '读取失败');
      var chunk = null;
      if (kind === 'event') chunk = (data.events || [])[0];
      else chunk = (data.turns || [])[0];
      if (!chunk || !chunk.text) { button.parentNode.remove(); return; }
      var moreHtml = chunk.truncated && chunk.continue_after >= 0
        ? '<button type="button" class="chat-memory-source-more" data-kind="' + escAttr(kind) + '" data-sid="' + escAttr(sourceId) + '" data-offset="' + escAttr(chunk.continue_after) + '" onclick="loadDailyChatMemorySourceMore(this)">继续加载完整原文</button>'
        : '';
      var wrap = document.createElement('div');
      wrap.className = 'chat-memory-source-chunk';
      wrap.textContent = chunk.text;
      button.parentNode.appendChild(wrap);
      button.parentNode.appendChild(moreHtml ? createElFromHtml(moreHtml) : document.createTextNode(''));
      button.remove();
    } catch (e) {
      button.disabled = false;
    }
  }

  function createElFromHtml(html) {
    var template = document.createElement('template');
    template.innerHTML = html.trim();
    return template.content.firstChild;
  }

  async function toggleDailyChatMemorySource(button, id) {
    var card = button && button.closest ? button.closest('.chat-memory-card') : null;
    var panel = card && card.querySelector ? card.querySelector('.chat-memory-source-panel') : null;
    if (!panel) return;
    if (!panel.hidden) {
      panel.hidden = true;
      button.textContent = '查看完整原文';
      return;
    }
    panel.hidden = false;
    panel.innerHTML = '<div class="chat-memory-source-loading">读取完整原文...</div>';
    button.textContent = '收起原文';
    try {
      var res = await authFetch(dailyChatMemoryApiBase() + '/api/daily-chat-memory/source-preview?candidate_id=' + encodeURIComponent(id));
      if (!res) return;
      var data = await res.json();
      if (!res.ok) throw new Error(data.error || '读取失败');
      panel.innerHTML = renderDailyChatMemorySourcePanel(data);
    } catch (e) {
      panel.innerHTML = '<div class="chat-memory-source-empty">读取失败: ' + esc(e.message) + '</div>';
    }
  }

  function listText(value) {
    return Array.isArray(value) ? value.join(', ') : String(value || '');
  }

  function selectedDailyChatMemoryIds() {
    var checkboxes = document.querySelectorAll('#daily-chat-memory-cards input[data-select]:checked');
    var ids = [];
    checkboxes.forEach(function (checkbox) { ids.push(checkbox.getAttribute('data-select')); });
    return ids;
  }

  function refreshDailyChatMemorySelection() {
    var countEl = document.getElementById('daily-chat-memory-batch-count');
    if (countEl) countEl.textContent = '已选 ' + selectedDailyChatMemoryIds().length + ' 条';
  }

  function toggleDailyChatMemorySelectAll(checkbox) {
    document.querySelectorAll('#daily-chat-memory-cards input[data-select]').forEach(function (el) {
      el.checked = checkbox.checked;
    });
    refreshDailyChatMemorySelection();
  }

  function dailyChatMemoryField(card, name) {
    var el = card && card.querySelector ? card.querySelector('[data-field="' + name + '"]') : null;
    return el ? String(el.value || '').trim() : '';
  }

  function readDailyChatMemoryEdits(card) {
    if (!card) return null;
    var edits = {
      title: dailyChatMemoryField(card, 'title'),
      content: dailyChatMemoryField(card, 'content'),
      kind: dailyChatMemoryField(card, 'kind'),
      domain: splitTerms(dailyChatMemoryField(card, 'domain')),
      tags: splitTerms(dailyChatMemoryField(card, 'tags')),
      event_date: dailyChatMemoryField(card, 'event_date'),
    };
    var importance = dailyChatMemoryField(card, 'importance');
    var confidence = dailyChatMemoryField(card, 'confidence');
    if (importance) edits.importance = Number(importance);
    if (confidence) edits.confidence = Number(confidence);
    return edits;
  }

  function toggleDailyChatMemoryEdit(button) {
    if (memoryBusy) return;
    var card = button && button.closest ? button.closest('.chat-memory-card') : null;
    var panel = card && card.querySelector ? card.querySelector('.chat-memory-edit-panel') : null;
    if (!panel) return;
    panel.hidden = !panel.hidden;
    if (card) card.setAttribute('data-editing', panel.hidden ? 'false' : 'true');
    button.textContent = panel.hidden ? '编辑' : '收起编辑';
    captureMemoryDraft(card);
  }

  function openDailyChatMemoryEdit(card) {
    if (!card) return;
    var button = card.querySelector && card.querySelector('.chat-memory-card-actions button');
    var panel = card.querySelector ? card.querySelector('.chat-memory-edit-panel') : null;
    if (panel && panel.hidden) {
      panel.hidden = false;
      card.setAttribute('data-editing', 'true');
      if (button) button.textContent = '收起编辑';
    }
  }

  async function postDailyChatMemoryConfirm(body) {
    var res = await authFetch(dailyChatMemoryApiBase() + '/api/daily-chat-memory/confirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res) return null;
    var data = await res.json();
    if (!res.ok && !data) throw new Error('操作失败 (HTTP ' + res.status + ')');
    if (!res.ok) {
      data = data || {};
      var error = new Error(data.error || data.reason || '操作失败 (HTTP ' + res.status + ')');
      error.definiteRejection = res.status >= 400 && res.status < 500;
      throw error;
    }
    return data;
  }

  async function confirmDailyChatMemory(buttonOrId, idOrAction, maybeAction) {
    var button = typeof buttonOrId === 'object' ? buttonOrId : null;
    var id = button ? idOrAction : buttonOrId;
    return performMemoryAction([id], button ? maybeAction : idOrAction, false);
  }

  async function batchDailyChatMemoryConfirm(button, action) {
    var ids = selectedDailyChatMemoryIds();
    if (!ids.length) {
      setDailyChatMemoryMessage('请先勾选要操作的候选。', 'error');
      return;
    }
    return performMemoryAction(ids, action, true);
  }

  async function performMemoryAction(ids, action, batch) {
    if (memoryBusy) return;
    captureMemoryDrafts();
    ids = ids.slice().sort();
    var isReject = action === 'reject', isDefer = action === 'defer';
    var verb = isReject ? '拒绝' : (isDefer ? '暂缓' : '采用并入库');
    if (!confirm(verb + ' ' + ids.length + ' 条候选？' + (isReject ? '这些条目的未保存修改将放弃。' : '先保存所选条目的修改；保存本身不代表入库。'))) return;
    memoryBusy = true;
    ++memoryLoadEpoch;
    setMemoryControlsBusy(true);
    var refresh = false;
    var key = action + ':' + ids.join(',');
    try {
      ids.forEach(function (id) {
        var pending = pendingMemoryAction(id);
        if (pending && pending[0] !== key) throw new Error('该候选上次审核结果尚未确认，请先重试原操作。');
        if (isReject && memoryRequests.has('edit:' + id)) throw new Error('保存结果尚未确认，请先重试保存。');
      });
      var retry = memoryRequests.get(key);
      if (!retry && !isReject) for (var id of ids) {
        await saveMemoryDraft(id);
        if (memoryDrafts.has(id)) throw new Error('保存期间又有新修改，已保留草稿；请保存这些修改后再采用。');
      }
      var revisions = {};
      if (!retry) ids.forEach(function (id) {
        var item = memoryItems.get(id);
        if (!item || !Number.isInteger(item.authority_revision) || item.draft_orphan) throw new Error('候选版本不可用，请重新读取并核对。');
        revisions[id] = item.authority_revision;
      });
      var input = { candidate_ids: ids, action: action, expected_revisions: revisions,
        confirm: isReject ? 'REJECT' : (isDefer ? 'DEFER' : 'WRITE') };
      if (isReject) {
        var reason = document.getElementById(batch ? 'daily-chat-memory-batch-reason' : 'reject-reason-' + ids[0]);
        var note = document.getElementById(batch ? 'daily-chat-memory-batch-note' : 'reject-note-' + ids[0]);
        if (reason) input.reason = reason.value;
        if (note && note.value.trim()) input.reason_note = note.value.trim();
      }
      var body = retry ? retry.body : stableMemoryRequest(key, input);
      var data = await postDailyChatMemoryConfirm(body);
      if (!data) throw new Error('结果未确认，请重试原操作，不要重新创建候选。');
      if (data.status === 'rate_limited') { memoryRequests.delete(key); persistMemoryState(); throw new Error('操作太频繁，请稍后重试。'); }
      if (data.status === 'conflict') { memoryRequests.delete(key); persistMemoryState(); throw new Error('版本或请求冲突，请核对候选。'); }
      var expected = isReject ? ['rejected'] : (isDefer ? ['deferred'] : ['created', 'exists']);
      var results = data.results || [];
      var applied = results.filter(function (r) { return expected.includes(r.status); });
      var complete = results.length === ids.length && new Set(results.map(function (r) { return r.id; })).size === ids.length && results.every(function (r) {
        return ids.includes(r.id) && (expected.includes(r.status) || ['revision_conflict', 'needs_owner_edit', 'invalid_source', 'missing', 'rejected', 'deferred', 'commit_failed'].includes(r.status));
      });
      if (complete) {
        memoryRequests.delete(key);
        applied.forEach(function (r) { memoryDrafts.delete(r.id); memoryItems.delete(r.id); });
        persistMemoryState();
      }
      var failed = results.filter(function (r) { return !expected.includes(r.status); });
      var detail = failed.some(function (r) { return r.status === 'revision_conflict'; }) ? '版本已变化，请载入新版核对。'
        : failed.some(function (r) { return r.status === 'needs_owner_edit'; }) ? '部分候选需要实际修改正文后才能采用。'
        : '其余未成功或结果未确认，请核对；已保存的修改仍然保留。';
      setDailyChatMemoryMessage(verb + '成功 ' + applied.length + '／' + ids.length + ' 条。' + (applied.length === ids.length ? '' : detail), applied.length === ids.length ? 'ok' : 'error');
      refresh = applied.length === ids.length || !memoryRequests.has(key);
    } catch (e) {
      if (e.definiteRejection) { memoryRequests.delete(key); persistMemoryState(); }
      setDailyChatMemoryMessage('未完成：' + e.message, 'error');
    } finally {
      memoryBusy = false;
      setMemoryControlsBusy(false);
      if (refresh) loadDailyChatMemoryPending();
    }
  }

  function renderDailyChatMemoryRuns(data) {
    var cursor = data.cursor || {};
    var runs = data.runs || [];
    var cursorLine = 'watermark：last_raw_event_id=' + (cursor.last_raw_event_id || 0) + (cursor.updated_at ? '（' + esc(cursor.updated_at) + '）' : '');
    var runLines = '';
    if (!runs.length) {
      runLines = '<div class="chat-memory-run-empty">尚无运行记录。</div>';
    } else {
      runLines = runs.map(function (run) {
        var hard = run.hard_rejects || {};
        var hardText = Object.keys(hard).length ? Object.keys(hard).map(function (k) { return k + ':' + hard[k]; }).join('，') : '0';
        var statusText = run.status === 'zero_candidates' ? '0 条候选（窗口已全部检查）' : (run.status === 'degraded_empty_outputs' ? '空转（模型答复为空，未推水位）' : run.status);
        return '<div class="chat-memory-run-item">' +
          '<div class="chat-memory-run-head">' + esc(run.date || '') + ' · ' + esc(statusText) + ' · ' + esc(String(run.completed_at || run.created_at || '').slice(0, 19)) + '</div>' +
          '<div class="chat-memory-run-meta">seq ' + esc(run.source_start_seq || 0) + '→' + esc(run.source_end_seq || 0) +
            ' · 轮次 ' + esc(run.eligible_turn_count || 0) +
            ' · 窗口 ' + esc(run.window_count || 0) + '（跳过噪音 ' + esc(run.skipped_noise_window_count || 0) + '）' +
            ' · 模型调用 ' + esc(run.model_call_count || 0) +
            ' · 模型候选 ' + esc(run.model_candidate_count || 0) +
            ' · 空输出 ' + esc(run.empty_output_count || 0) +
            ' · 解析失败 ' + esc(run.parse_failure_count || 0) +
            ' · 硬拒绝 ' + esc(hardText) +
            ' · 进入Review ' + esc(run.pending_count || 0) +
            ' · 合并去重 ' + esc(run.merged_duplicates || 0) +
            (run.error_category ? ' · 错误 ' + esc(run.error_category) : '') +
          '</div>' +
        '</div>';
      }).join('');
    }
    return '<div class="chat-memory-runs">' +
      '<div class="chat-memory-runs-title">最近运行（' + runs.length + ' 条，纯运行信息）</div>' +
      '<div class="chat-memory-runs-cursor">' + cursorLine + '</div>' +
      runLines +
      '<div class="chat-memory-runs-actions">' +
        '<button type="button" class="chat-memory-rerun-btn" onclick="rerunDailyChatMemory(this)">补扫新增区间</button>' +
        '<span class="chat-memory-runs-hint">仅扫描尚未处理区间；会发起一次真实模型调用并产生费用。</span>' +
      '</div>' +
    '</div>';
  }

  async function loadDailyChatMemoryRuns() {
    var target = document.getElementById('daily-chat-memory-runs');
    if (!target) return;
    try {
      var res = await authFetch(dailyChatMemoryApiBase() + '/api/daily-chat-memory/runs?limit=10');
      if (!res) return;
      var data = await res.json();
      if (!res.ok) throw new Error(data.error || '读取失败');
      target.innerHTML = renderDailyChatMemoryRuns(data);
    } catch (e) {
      target.innerHTML = '<div class="chat-memory-runs"><div class="chat-memory-run-empty">运行信息读取失败: ' + esc(e.message) + '</div></div>';
    }
  }

  async function rerunDailyChatMemory(button) {
    if (!confirm('确认补扫新增区间？这会发起一次真实模型调用并产生费用。')) return;
    button.disabled = true;
    try {
      var body = { request_id: newDailyChatMemoryRequestId() };
      var res = await authFetch(dailyChatMemoryApiBase() + '/api/daily-chat-memory/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res) return;
      var data = await res.json();
      if (!res.ok) throw new Error(data.error || '补扫失败');
      if (data.status === 'locked') {
        setDailyChatMemoryMessage('已有运行在进行中，请稍后再试。', 'error');
      } else {
        setDailyChatMemoryMessage('补扫完成：' + (data.status === 'zero_candidates' ? '0 条候选（区间已检查）' : data.status + '，候选 ' + (data.candidates || []).length + ' 条'), 'ok');
      }
      loadDailyChatMemoryRuns();
      loadDailyChatMemoryPending();
    } catch (e) {
      setDailyChatMemoryMessage('补扫失败: ' + e.message, 'error');
    } finally {
      button.disabled = false;
    }
  }

  function initDailyChatMemoryTab() {
    loadDailyChatMemoryPending();
    loadDailyChatMemoryRuns();
  }

  window.setDailyChatMemoryMessage = setDailyChatMemoryMessage;
  window.loadDailyChatMemoryPending = loadDailyChatMemoryPending;
  window.renderDailyChatMemoryPending = renderDailyChatMemoryPending;
  window.confirmDailyChatMemory = confirmDailyChatMemory;
  window.batchDailyChatMemoryConfirm = batchDailyChatMemoryConfirm;
  window.toggleDailyChatMemoryEdit = toggleDailyChatMemoryEdit;
  window.toggleDailyChatMemoryL1Mode = toggleDailyChatMemoryL1Mode;
  window.mapDailyChatMemorySceneBody = mapDailyChatMemorySceneBody;
  window.discardDailyChatMemorySceneBody = discardDailyChatMemorySceneBody;
  window.saveDailyChatMemoryEdit = saveDailyChatMemoryEdit;
  window.discardDailyChatMemoryDraft = discardDailyChatMemoryDraft;
  window.toggleDailyChatMemorySelectAll = toggleDailyChatMemorySelectAll;
  window.refreshDailyChatMemorySelection = refreshDailyChatMemorySelection;
  window.toggleDailyChatMemorySource = toggleDailyChatMemorySource;
  window.loadDailyChatMemorySourceMore = loadDailyChatMemorySourceMore;
  window.loadDailyChatMemoryRuns = loadDailyChatMemoryRuns;
  window.rerunDailyChatMemory = rerunDailyChatMemory;
  window.initDailyChatMemoryTab = initDailyChatMemoryTab;

  if (typeof getActiveTab === 'function' && getActiveTab() === 'chat-memory') {
    initDailyChatMemoryTab();
  }
})();
