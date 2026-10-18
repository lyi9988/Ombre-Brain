(function () {
  var memoryItems = new Map();
  var memoryDrafts = new Map();
  var memoryRequests = new Map();
  var memoryLoadEpoch = 0;
  var memorySourceLoadEpoch = 0;
  var memoryVisibleItems = [];
  var memoryPage = {};
  var sourceTaskPage = {};
  var memoryBusy = false;
  var memoryStateRestored = false;
  var sourceTaskDetails = new Map();
  var sourceTaskDrafts = new Map();
  var sourceTaskOpenIds = new Set();
  var sourceTaskDetailEpochs = new Map();
  var sourceTaskDetailErrors = new Map();
  var sourceTaskSaving = new Set();
  var sourceTaskTasks = [];
  var modelIoPanels = new Map();
  var MEMORY_STATE_KEY = 'guyan.ombre.daily-chat-memory-review.v1';

  var MODEL_IO_STAGES = ['chat_l1', 'presentation_l2'];
  var MODEL_IO_LIMIT = 20;

  function modelIoMemoryId(value) {
    return typeof value === 'string' ? value.trim() : '';
  }

  function modelIoViewKey(kind, memoryId) {
    return String(kind || 'review') + ':' + String(memoryId || '');
  }

  function modelIoStageLabel(stage) {
    var labels = { chat_l1: '完整第一遍记忆 L1', presentation_l2: '第二遍可读注入记忆 L2' };
    return labels[stage] || String(stage || '未提供阶段');
  }

  function modelIoText(value) {
    if (value == null) return '';
    if (typeof value === 'string') return value;
    try { return JSON.stringify(value, null, 2); } catch (_) { return String(value); }
  }

  function modelIoValueLabel(value) {
    if (value == null || value === '') return '未留存';
    return modelIoText(value);
  }

  function modelIoTokenSummary(usage) {
    if (!usage || typeof usage !== 'object' || Array.isArray(usage)) return 'Token 用量未留存';
    var fields = [
      ['prompt_tokens', 'prompt'], ['input_tokens', 'input'],
      ['completion_tokens', 'completion'], ['output_tokens', 'output'],
      ['reasoning_tokens', 'reasoning'], ['total_tokens', 'total'],
    ];
    var parts = fields.filter(function (field) {
      return Object.prototype.hasOwnProperty.call(usage, field[0]) && usage[field[0]] != null;
    }).map(function (field) { return field[1] + ' ' + modelIoText(usage[field[0]]); });
    return parts.length ? 'Token ' + parts.join(' · ') : 'Token 用量已留存';
  }

  function modelIoValidationSummary(validation) {
    if (validation == null) return '未留存';
    if (typeof validation !== 'object' || Array.isArray(validation)) return modelIoText(validation);
    var parts = [];
    ['status', 'state', 'ok', 'valid', 'error', 'code'].forEach(function (key) {
      if (Object.prototype.hasOwnProperty.call(validation, key) && validation[key] != null) {
        parts.push(key + ': ' + modelIoText(validation[key]));
      }
    });
    return parts.length ? parts.join(' · ') : '结果已留存，展开查看完整结构。';
  }

  function modelIoAppendText(parent, tagName, className, text) {
    var element = document.createElement(tagName || 'div');
    if (className) element.className = className;
    element.textContent = text == null ? '' : String(text);
    parent.appendChild(element);
    return element;
  }

  function modelIoAppendJsonDetails(parent, label, value) {
    var details = document.createElement('details');
    details.className = 'chat-memory-model-io-json';
    var summary = document.createElement('summary');
    summary.textContent = label;
    details.appendChild(summary);
    var pre = document.createElement('pre');
    pre.textContent = value == null ? '未留存' : modelIoText(value);
    details.appendChild(pre);
    parent.appendChild(details);
    return details;
  }

  function modelIoCopy(button, text, status) {
    var value = String(text == null ? '' : text);
    var copied = false;
    var clipboard = window.navigator && window.navigator.clipboard;
    var write;
    try {
      write = clipboard && typeof clipboard.writeText === 'function'
        ? Promise.resolve(clipboard.writeText(value)).then(function () { copied = true; })
        : Promise.reject(new Error('clipboard_unavailable'));
    } catch (error) { write = Promise.reject(error); }
    return write.catch(function () {
      var input = document.createElement('textarea');
      input.value = value;
      input.readOnly = true;
      input.setAttribute('aria-hidden', 'true');
      input.style.position = 'fixed'; input.style.left = '-10000px'; input.style.top = '0';
      document.body.appendChild(input);
      input.select();
      try { copied = Boolean(document.execCommand && document.execCommand('copy')); }
      catch (_) { copied = false; }
      input.remove();
    }).then(function () {
      if (status) status.textContent = copied ? '已复制到剪贴板。' : '浏览器未开放复制权限；可直接选中正文复制。';
      if (button) button.disabled = false;
    });
  }

  function modelIoCopyButton(parent, label, value, status) {
    var button = document.createElement('button');
    button.type = 'button'; button.className = 'chat-memory-model-io-copy'; button.textContent = label;
    button.addEventListener('click', function () {
      button.disabled = true;
      modelIoCopy(button, value, status);
    });
    parent.appendChild(button);
    return button;
  }

  function modelIoAppendMessage(parent, message, index, status) {
    var row = document.createElement('div'); row.className = 'chat-memory-model-io-message';
    var role = message && message.role != null ? String(message.role) : '未提供角色';
    var content = message && Object.prototype.hasOwnProperty.call(message, 'content')
      ? modelIoText(message.content) : '消息正文未留存';
    var head = document.createElement('div'); head.className = 'chat-memory-model-io-message-head';
    modelIoAppendText(head, 'strong', '', '消息 ' + (index + 1) + ' · ' + role);
    modelIoCopyButton(head, '复制此条输入', content, status);
    row.appendChild(head);
    modelIoAppendText(row, 'pre', 'chat-memory-model-io-content', content);
    parent.appendChild(row);
  }

  function modelIoAppendRecord(parent, record, status) {
    record = record && typeof record === 'object' ? record : {};
    var stage = String(record.stage || 'unknown');
    var details = document.createElement('details');
    details.className = 'chat-memory-model-io-record';
    details.setAttribute('data-model-io-stage', stage);
    var summary = document.createElement('summary');
    var attempt = record.attempt;
    var numericAttempt = stage === 'chat_l1' && attempt != null && /^\d+$/.test(String(attempt));
    var attemptText = attempt == null ? '尝试标识未留存'
      : (numericAttempt ? '第 ' + modelIoText(attempt) + ' 次尝试' : '本轮调用');
    var stateText = record.status == null ? '状态未留存' : modelIoText(record.status);
    var finishText = record.finish_reason == null || record.finish_reason === ''
      ? 'finish_reason 未留存' : 'finish_reason ' + modelIoText(record.finish_reason);
    var started = record.started_at || record.ended_at || '';
    summary.textContent = modelIoStageLabel(stage) + ' · ' + attemptText + ' · ' + stateText +
      ' · ' + finishText + (started ? ' · ' + modelIoText(started) : '') +
      ' · ' + modelIoTokenSummary(record.usage);
    details.appendChild(summary);

    var body = document.createElement('div'); body.className = 'chat-memory-model-io-record-body';
    var metadata = document.createElement('dl'); metadata.className = 'chat-memory-model-io-meta';
    [
      ['阶段', record.stage], ['任务 ID', record.task_id], ['调用记录 ID', record.io_id],
      ['尝试序号或标识', record.attempt], ['任务调用标识', record.attempt_id], ['状态', record.status],
      ['开始时间', record.started_at], ['结束时间', record.ended_at], ['结束原因', record.finish_reason],
      ['草稿版本', record.draft_revision],
    ].forEach(function (field) {
      modelIoAppendText(metadata, 'dt', '', field[0]);
      modelIoAppendText(metadata, 'dd', '', modelIoValueLabel(field[1]));
    });
    body.appendChild(metadata);
    modelIoAppendJsonDetails(body, '完整 usage 与 token 计数', record.usage);
    modelIoAppendJsonDetails(body, '来源版本', record.source_revisions);
    modelIoAppendJsonDetails(body, '模型与生成信息', record.generation);
    modelIoAppendJsonDetails(body, 'completion_parameters', record.completion_parameters);
    var validation = record.validation;
    var validationSummary = '结构验证结果：' + modelIoValidationSummary(validation);
    modelIoAppendText(body, 'p', 'chat-memory-model-io-validation', validationSummary);
    if (validation != null) modelIoAppendJsonDetails(body, '结构验证完整结果', validation);

    var messages = Array.isArray(record.messages) ? record.messages
      : (Array.isArray(record.input_messages) ? record.input_messages : []);
    modelIoAppendText(body, 'h5', '', '模型实际输入');
    if (messages.length) messages.forEach(function (message, index) {
      modelIoAppendMessage(body, message, index, status);
    });
    else modelIoAppendText(body, 'p', 'chat-memory-model-io-empty', '输入消息未留存。');

    var hasRaw = Object.prototype.hasOwnProperty.call(record, 'raw_output') && record.raw_output != null;
    var hasText = Object.prototype.hasOwnProperty.call(record, 'output_text') && record.output_text != null;
    var output = hasRaw ? modelIoText(record.raw_output) : (hasText ? modelIoText(record.output_text) : null);
    modelIoAppendText(body, 'h5', '', '模型可见输出');
    if (output == null) {
      modelIoAppendText(body, 'p', 'chat-memory-model-io-empty', '模型可见输出未留存。');
    } else {
      var outputHead = document.createElement('div'); outputHead.className = 'chat-memory-model-io-message-head';
      modelIoCopyButton(outputHead, '复制完整输出', output, status);
      body.appendChild(outputHead);
      modelIoAppendText(body, 'pre', 'chat-memory-model-io-content chat-memory-model-io-output', output);
    }
    details.appendChild(body);
    parent.appendChild(details);
    return details;
  }

  function modelIoEnsureStage(records, stage) {
    var section = Array.from(records.querySelectorAll('[data-model-io-stage-section]')).find(function (node) {
      return node.getAttribute('data-model-io-stage-section') === String(stage || 'unknown');
    });
    if (section) return section;
    section = document.createElement('section'); section.className = 'chat-memory-model-io-stage';
    section.setAttribute('data-model-io-stage-section', stage || 'unknown');
    modelIoAppendText(section, 'h4', '', modelIoStageLabel(stage));
    var items = document.createElement('div'); items.className = 'chat-memory-model-io-stage-items';
    items.setAttribute('data-model-io-stage-items', stage || 'unknown');
    section.appendChild(items);
    records.appendChild(section);
    return section;
  }

  function modelIoRenderPage(panel, response, items, append, requestedOffset) {
    var state = panel._modelIoState;
    var records = panel.querySelector('[data-model-io-records]');
    var status = panel.querySelector('[data-model-io-status]');
    var stages = Array.isArray(response.supported_stages) && response.supported_stages.length
      ? response.supported_stages : MODEL_IO_STAGES;
    if (!append) {
      records.replaceChildren();
      state.supportedStages = stages.slice();
      stages.forEach(function (stage) {
        var section = modelIoEnsureStage(records, stage);
        var itemsEl = section.querySelector('[data-model-io-stage-items]');
        modelIoAppendText(itemsEl, 'p', 'chat-memory-model-io-empty',
          '此阶段尚无已留存的调用记录；旧输出无法补回。');
      });
    }
    items.forEach(function (record) {
      record = record && typeof record === 'object' ? record : {};
      var stage = String(record.stage || 'unknown');
      var section = modelIoEnsureStage(records, stage);
      var stageItems = section.querySelector('[data-model-io-stage-items]');
      var empty = stageItems.querySelector('.chat-memory-model-io-empty');
      if (empty) empty.remove();
      modelIoAppendRecord(stageItems, record, status);
    });
    if (!items.length && !append && response.capture_state === 'missing' && !stages.length) {
      modelIoAppendText(records, 'p', 'chat-memory-model-io-empty',
        '此阶段尚无已留存的调用记录；旧输出无法补回。');
    }
    state.loaded = true;
    state.captureState = String(response.capture_state || (items.length ? 'recorded' : 'missing'));
    state.nextOffset = Number.isInteger(Number(response.next_offset))
      ? Number(response.next_offset) : requestedOffset + items.length;
    state.hasMore = Boolean(response.has_more);
    var totalLoaded = records.querySelectorAll('.chat-memory-model-io-record').length;
    status.textContent = items.length
      ? '已读取 ' + totalLoaded + ' 条模型调用记录；按阶段展开后可查看完整输入、输出和校验。'
      : (state.captureState === 'missing'
        ? '没有已留存的模型调用记录；旧输出无法补回。'
        : (totalLoaded ? '本页没有更多记录；已读取 ' + totalLoaded + ' 条。' : '当前没有已留存的模型调用记录。'));
    var more = panel.querySelector('[data-model-io-more]');
    if (more) more.hidden = !state.hasMore;
  }

  async function fetchDailyChatMemoryModelIo(memoryId, panel, offset, refresh) {
    var state = panel && panel._modelIoState;
    if (!panel || !state || state.loading) return;
    state.loading = true;
    var status = panel.querySelector('[data-model-io-status]');
    var refreshButton = panel.querySelector('[data-model-io-refresh]');
    var moreButton = panel.querySelector('[data-model-io-more]');
    if (refreshButton) refreshButton.disabled = true;
    if (moreButton) moreButton.disabled = true;
    if (!state.loaded) status.textContent = '正在读取已留存的模型调用记录…';
    try {
      var requestedOffset = refresh ? 0 : Number(offset || 0);
      var url = dailyChatMemoryApiBase() + '/api/memory-authority/memories/' +
        encodeURIComponent(memoryId) + '/model-io?limit=' + MODEL_IO_LIMIT + '&offset=' + requestedOffset;
      var response = await authFetch(url);
      if (!response) throw new Error('未连接，模型调用记录保留在服务端。');
      var data = await response.json();
      if (!response.ok || !data || (data.status && data.status !== 'ok')) {
        throw new Error(data && data.error ? data.error : ('读取失败 (HTTP ' + response.status + ')'));
      }
      var items = Array.isArray(data.items) ? data.items
        : (Array.isArray(data.records) ? data.records : []);
      modelIoRenderPage(panel, data, items, !refresh && requestedOffset > 0, requestedOffset);
    } catch (error) {
      status.textContent = '模型输入与输出读取失败；已有记录仍保留：' + String(error && error.message || error);
    } finally {
      state.loading = false;
      if (refreshButton) refreshButton.disabled = false;
      if (moreButton) moreButton.disabled = false;
    }
  }

  function createDailyChatMemoryModelIoPanel(memoryId, viewKind) {
    var panel = document.createElement('section');
    panel.className = 'chat-memory-model-io-panel'; panel.hidden = true;
    panel.setAttribute('data-memory-id', memoryId);
    panel.setAttribute('aria-label', '模型输入与输出');
    panel._modelIoState = { loaded: false, loading: false, hasMore: false, nextOffset: 0,
      captureState: '', supportedStages: MODEL_IO_STAGES.slice() };
    panel._modelIoKind = viewKind;
    panel._modelIoKey = modelIoViewKey(viewKind, memoryId);
    var head = document.createElement('div'); head.className = 'chat-memory-model-io-head';
    modelIoAppendText(head, 'strong', '', '模型输入与输出');
    var actions = document.createElement('div'); actions.className = 'chat-memory-model-io-actions';
    var refresh = document.createElement('button'); refresh.type = 'button'; refresh.textContent = '刷新记录';
    refresh.setAttribute('data-model-io-refresh', '');
    refresh.addEventListener('click', function () { fetchDailyChatMemoryModelIo(memoryId, panel, 0, true); });
    actions.appendChild(refresh);
    var close = document.createElement('button'); close.type = 'button'; close.textContent = '收起';
    close.addEventListener('click', function () {
      panel.hidden = true;
      syncModelIoToggle(panel);
    });
    actions.appendChild(close); head.appendChild(actions); panel.appendChild(head);
    modelIoAppendText(panel, 'p', 'chat-memory-model-io-status',
      '展开后按需读取；每条调用记录可独立展开查看完整消息。');
    panel.querySelector('.chat-memory-model-io-status').setAttribute('data-model-io-status', '');
    var records = document.createElement('div'); records.className = 'chat-memory-model-io-records';
    records.setAttribute('data-model-io-records', ''); panel.appendChild(records);
    var more = document.createElement('button'); more.type = 'button'; more.textContent = '继续查看旧记录';
    more.className = 'chat-memory-model-io-more'; more.hidden = true; more.setAttribute('data-model-io-more', '');
    more.addEventListener('click', function () {
      fetchDailyChatMemoryModelIo(memoryId, panel, panel._modelIoState.nextOffset, false);
    });
    panel.appendChild(more);
    return panel;
  }

  function syncModelIoToggle(panel) {
    var card = panel && panel.closest ? panel.closest('.chat-memory-card') : null;
    var viewKey = panel && panel._modelIoKey;
    var button = card && Array.from(card.querySelectorAll('[data-model-io-toggle]')).find(function (candidate) {
      return candidate.getAttribute('data-model-io-view-key') === viewKey;
    });
    if (!button) return;
    button.textContent = panel.hidden ? '模型输入与输出' : '收起模型输入与输出';
    button.setAttribute('aria-expanded', panel.hidden ? 'false' : 'true');
  }

  function toggleDailyChatMemoryModelIo(button, memoryId, viewKind) {
    var id = modelIoMemoryId(memoryId);
    if (!id || !button) return;
    var card = button.closest ? button.closest('.chat-memory-card') : null;
    if (!card) return;
    var key = modelIoViewKey(viewKind, id);
    var panel = modelIoPanels.get(key);
    if (!panel) {
      panel = createDailyChatMemoryModelIoPanel(id, viewKind);
      modelIoPanels.set(key, panel);
    }
    if (panel.parentNode !== card) card.appendChild(panel);
    panel.hidden = !panel.hidden;
    syncModelIoToggle(panel);
    if (!panel.hidden && !panel._modelIoState.loaded) fetchDailyChatMemoryModelIo(id, panel, 0, false);
  }

  function createDailyChatMemoryModelIoToggle(memoryId, viewKind, label) {
    var id = modelIoMemoryId(memoryId);
    if (!id) return null;
    var button = document.createElement('button');
    button.type = 'button'; button.textContent = label || '模型输入与输出';
    button.setAttribute('data-model-io-toggle', '');
    button.setAttribute('data-model-io-id', id);
    button.setAttribute('data-model-io-kind', viewKind);
    button.setAttribute('data-model-io-view-key', modelIoViewKey(viewKind, id));
    button.setAttribute('aria-expanded', 'false');
    button.addEventListener('click', function () {
      toggleDailyChatMemoryModelIo(button, id, viewKind);
    });
    return button;
  }

  function restoreDailyChatMemoryModelIoPanels(target) {
    if (!target) return;
    target.querySelectorAll('.chat-memory-card[data-candidate-id]').forEach(function (card) {
      var id = modelIoMemoryId(card.getAttribute('data-candidate-id'));
      if (!id) return;
      var panel = modelIoPanels.get(modelIoViewKey('review', id));
      var slot = card.querySelector('[data-model-io-slot="review"]');
      if (panel && slot) {
        slot.replaceWith(panel);
        syncModelIoToggle(panel);
      }
    });
    bindDailyChatMemoryModelIoButtons(target);
  }

  function bindDailyChatMemoryModelIoButtons(target) {
    if (!target) return;
    target.querySelectorAll('[data-model-io-toggle][data-model-io-id]').forEach(function (button) {
      if (button._modelIoBound) return;
      button._modelIoBound = true;
      button.addEventListener('click', function () {
        toggleDailyChatMemoryModelIo(button, button.getAttribute('data-model-io-id'),
          button.getAttribute('data-model-io-kind') || 'review');
      });
    });
  }

  function cloneValue(value) {
    return value == null ? value : JSON.parse(JSON.stringify(value));
  }

  function persistMemoryState() {
    try {
      if (!window.sessionStorage) return;
      var drafts = Array.from(memoryDrafts.entries());
      var requests = Array.from(memoryRequests.entries());
      var sourceTaskDraftsState = Array.from(sourceTaskDrafts.entries()).filter(function (entry) {
        return entry[1] && (entry[1].dirty || entry[1].conflict || entry[1].unconfirmed);
      });
      if (!drafts.length && !requests.length && !sourceTaskDraftsState.length) {
        window.sessionStorage.removeItem(MEMORY_STATE_KEY);
        return;
      }
      window.sessionStorage.setItem(MEMORY_STATE_KEY, JSON.stringify({ version: 2, drafts: drafts, requests: requests,
        sourceTaskDrafts: sourceTaskDraftsState }));
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
      if (!saved || (saved.version !== 1 && saved.version !== 2)) return;
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
      (Array.isArray(saved.sourceTaskDrafts) ? saved.sourceTaskDrafts : []).forEach(function (pair) {
        if (Array.isArray(pair) && typeof pair[0] === 'string' && pair[1] && typeof pair[1] === 'object') {
          sourceTaskDrafts.set(pair[0], pair[1]);
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

  function cognitionInputText(value) {
    if (value == null) return '';
    return typeof value === 'string' ? value : displayValue(value);
  }

  function cognitionValueLists(document) {
    var defaults = {
      cognition_roles: ['observation', 'question', 'hypothesis', 'insight', 'view', 'practice'],
      cognition_statuses: ['tentative', 'supported', 'contested', 'revised', 'resolved'],
    };
    var snapshot = document && document.generation && document.generation.catalog_snapshot || {};
    function values(axis) {
      var source = Array.isArray(snapshot[axis]) ? snapshot[axis] : defaults[axis];
      var definitions = Array.isArray(snapshot.definitions) ? snapshot.definitions : [];
      var definitionAxis = axis === 'cognition_roles' ? 'cognition_role' : 'cognition_status';
      var fromDefinitions = definitions.filter(function (item) { return item && (item.axis === axis || item.axis === definitionAxis) && typeof item.id === 'string'; })
        .map(function (item) { return item.id; });
      return Array.from(new Set(source.concat(fromDefinitions).filter(function (item) { return typeof item === 'string' && item; })));
    }
    return { cognition_roles: values('cognition_roles'), cognition_statuses: values('cognition_statuses') };
  }

  function captureL1CognitionForm(card, l1Document, statementIndex) {
    var statements = Array.isArray(l1Document && l1Document.statements) ? l1Document.statements : [];
    var index = Number(statementIndex);
    var statement = Number.isInteger(index) ? statements[index] : null;
    if (!statement || typeof statement !== 'object' || Array.isArray(statement)) return;
    var current = statement.cognition;
    if (current != null && (typeof current !== 'object' || Array.isArray(current))) return;
    var cognition = current == null ? {} : current;
    var changed = false;
    function updateComponent(componentKey) {
      var stateInput = card.querySelector('[data-cognition-field="' + componentKey + '.state"]');
      var valueInput = card.querySelector('[data-cognition-field="' + componentKey + '.value"]');
      if (!stateInput && !valueInput) return;
      var original = cognition[componentKey];
      if (original != null && (typeof original !== 'object' || Array.isArray(original))) return;
      var component = original || {};
      var stateText = stateInput ? String(stateInput.value || '') : cognitionInputText(component.state);
      var valueText = valueInput ? String(valueInput.value || '') : cognitionInputText(component.value);
      var previousState = cognitionInputText(component.state);
      var previousValue = cognitionInputText(component.value);
      if (stateText === previousState && valueText === previousValue) return;
      changed = true;
      component = Object.assign({}, component);
      if (stateText === '' && valueText !== '') stateText = 'known';
      if (stateText === '') stateText = 'not_stated';
      component.state = stateText;
      component.value = valueText === '' ? null : valueText;
      if (!Object.prototype.hasOwnProperty.call(component, 'evidence_refs')) component.evidence_refs = [];
      cognition[componentKey] = component;
    }
    updateComponent('role'); updateComponent('epistemic_status');
    ['about_refs', 'open_questions'].forEach(function (key) {
      var input = card.querySelector('[data-cognition-field="' + key + '"]');
      if (!input) return;
      var previous = cognition[key];
      var previousText = Array.isArray(previous) ? previous.join('\n') : (previous == null ? '' : JSON.stringify(previous));
      if (input.value === previousText) return;
      changed = true;
      cognition[key] = String(input.value || '').split(/\r?\n/).map(function (value) { return value.trim(); }).filter(Boolean);
    });
    if (changed) {
      ['role', 'epistemic_status'].forEach(function (key) {
        var component = cognition[key];
        if (component == null) cognition[key] = { state: 'not_stated', value: null, evidence_refs: [] };
        else if (typeof component !== 'object' || Array.isArray(component)) return;
        else {
          if (!Object.prototype.hasOwnProperty.call(component, 'state')) component.state = 'not_stated';
          if (!Object.prototype.hasOwnProperty.call(component, 'value')) component.value = null;
          if (!Object.prototype.hasOwnProperty.call(component, 'evidence_refs')) component.evidence_refs = [];
        }
      });
      if (!Object.prototype.hasOwnProperty.call(cognition, 'about_refs')) cognition.about_refs = [];
      if (!Object.prototype.hasOwnProperty.call(cognition, 'open_questions')) cognition.open_questions = [];
      statement.cognition = cognition;
    }
  }

  function syncL1CognitionForm(card, l1Document, statementIndex) {
    var statements = Array.isArray(l1Document && l1Document.statements) ? l1Document.statements : [];
    var select = card && card.querySelector('[data-cognition-statement]');
    if (!select) return;
    var domDocument = card.ownerDocument || (typeof window !== 'undefined' ? window.document : null);
    if (!domDocument || typeof domDocument.createElement !== 'function') return;
    var index = Number(statementIndex);
    if (!Number.isInteger(index) || index < 0) index = 0;
    if (statements.length && index >= statements.length) index = statements.length - 1;
    if (!statements.length) index = 0;
    select.replaceChildren();
    statements.forEach(function (statement, itemIndex) {
      var option = domDocument.createElement('option');
      var identifier = statement && (statement.statement_id || statement.id) || ('陈述 ' + (itemIndex + 1));
      var content = String(statement && statement.content || '').replace(/\s+/g, ' ').slice(0, 54);
      option.value = String(itemIndex);
      option.textContent = (itemIndex + 1) + '. ' + identifier + (content ? ' · ' + content : '');
      select.appendChild(option);
    });
    if (!statements.length) {
      var empty = domDocument.createElement('option'); empty.value = ''; empty.textContent = '当前没有可编辑的现有陈述'; empty.disabled = true; select.appendChild(empty);
      select.value = '';
      select.setAttribute('data-cognition-active-index', '0');
      card.querySelectorAll('[data-cognition-field]').forEach(function (input) { input.value = ''; input.disabled = true; });
      select.disabled = true;
      return;
    }
    select.value = String(index);
    select.setAttribute('data-cognition-active-index', String(index));
    select.disabled = false;
    var statement = statements[index] || {};
    var cognition = statement.cognition && typeof statement.cognition === 'object' && !Array.isArray(statement.cognition) ? statement.cognition : {};
    var role = cognition.role && typeof cognition.role === 'object' && !Array.isArray(cognition.role) ? cognition.role : {};
    var status = cognition.epistemic_status && typeof cognition.epistemic_status === 'object' && !Array.isArray(cognition.epistemic_status) ? cognition.epistemic_status : {};
    var values = {
      'role.state': cognitionInputText(role.state), 'role.value': cognitionInputText(role.value),
      'epistemic_status.state': cognitionInputText(status.state), 'epistemic_status.value': cognitionInputText(status.value),
      about_refs: Array.isArray(cognition.about_refs) ? cognition.about_refs.join('\n') : cognitionInputText(cognition.about_refs),
      open_questions: Array.isArray(cognition.open_questions) ? cognition.open_questions.join('\n') : cognitionInputText(cognition.open_questions),
    };
    var activeRoot = card && typeof card.getRootNode === 'function' ? card.getRootNode() : domDocument;
    if (!activeRoot || !activeRoot.activeElement) activeRoot = domDocument;
    var activeElement = activeRoot && activeRoot.activeElement;
    var activeElements = new Set();
    while (activeElement && !activeElements.has(activeElement)) {
      if (activeElement === domDocument.activeElement && activeElement === select) break;
      activeElements.add(activeElement);
      var nestedShadow = activeElement.shadowRoot;
      if (!nestedShadow || !nestedShadow.activeElement) break;
      activeElement = nestedShadow.activeElement;
    }
    card.querySelectorAll('[data-cognition-field]').forEach(function (input) {
      if (activeElement !== input) input.value = values[input.getAttribute('data-cognition-field')] || '';
      input.disabled = false;
    });
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
      var cognitionSelect = card.querySelector('[data-cognition-statement]');
      var cognitionStatementIndex = prior && prior.type === 'l1' && Number.isInteger(prior.cognitionStatementIndex)
        ? prior.cognitionStatementIndex : Number(cognitionSelect && cognitionSelect.getAttribute('data-cognition-active-index')) || 0;
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
        captureL1CognitionForm(card, document, cognitionStatementIndex);
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
          invalidJson: invalidJson, mode: mode, cognitionStatementIndex: cognitionStatementIndex,
          revision: prior ? prior.revision : Number(card.getAttribute('data-revision')),
          expanded: card.getAttribute('data-editing') === 'true' });
      } else memoryDrafts.delete(id);
      var error = card.querySelector('[data-l1-json-error]');
      if (error) error.textContent = invalidJson ? 'JSON 尚未解析成功；草稿已保留，修复 JSON 后才能保存或采用。' : '';
      var semantics = card.querySelector('.chat-memory-semantics');
      if (semantics && !invalidJson) {
        var wasOpen = Boolean(semantics.querySelector('details[open]'));
        semantics.innerHTML = renderL1ReviewDetails(item, document);
        var details = semantics.querySelector('details');
        if (details && wasOpen) details.open = true;
      }
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
      cognitionStatementIndex: currentDraft.cognitionStatementIndex || 0,
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
          syncL1Form(card, draft.document, draft.sceneBodies, draft.cognitionStatementIndex);
          var json = card.querySelector('[data-l1-json]');
          if (json) json.value = draft.rawJson || JSON.stringify(draft.document, null, 2);
          var error = card.querySelector('[data-l1-json-error]');
          if (error) error.textContent = draft.invalidJson ? 'JSON 尚未解析成功；草稿已保留，修复 JSON 后才能保存或采用。' : '';
        } else {
          syncL1Form(card, l1Document, itemSceneBodies(item), 0);
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
    var sourceEditor = event.target.closest && event.target.closest('[data-chat-submission-editor]');
    if (sourceEditor) captureSourceTaskDraft(sourceEditor);
  });
  if (document.addEventListener) document.addEventListener('change', function (event) {
    var select = event.target.closest && event.target.closest('[data-cognition-statement]');
    if (!select) return;
    var card = select.closest('.chat-memory-card');
    if (!card) return;
    captureMemoryDraft(card);
    var id = card.getAttribute('data-candidate-id');
    var draft = memoryDrafts.get(id);
    var item = memoryItems.get(id);
    var index = Number(select.value);
    if (!Number.isInteger(index) || index < 0) return;
    select.setAttribute('data-cognition-active-index', String(index));
    if (draft && draft.type === 'l1') {
      draft.cognitionStatementIndex = index;
      if (!draft.invalidJson) syncL1CognitionForm(card, draft.document, index);
      persistMemoryState();
    } else if (item) syncL1CognitionForm(card, itemL1Document(item), index);
  });
  if (window.addEventListener) window.addEventListener('beforeunload', function (event) {
    captureMemoryDrafts();
    captureSourceTaskEditors(document);
    var hasSourceTaskWork = Array.from(sourceTaskDrafts.values()).some(function(state) {
      return state && (state.dirty || state.conflict || state.unconfirmed);
    });
    if (memoryDrafts.size || memoryBusy || memoryRequests.size || hasSourceTaskWork || sourceTaskSaving.size) {
      event.preventDefault(); event.returnValue = '';
    }
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

  function semanticValue(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return displayValue(value);
    var labels = { not_stated: '未说明', unresolved: '待确认', not_applicable: '不适用', unbounded: '明确不限定结束' };
    return value.state === 'known' ? displayValue(value.value) : (labels[value.state] || displayValue(value));
  }

  function cognitionFieldSummary(value) {
    if (value == null) return '未提供';
    if (!value || typeof value !== 'object' || Array.isArray(value)) return displayValue(value);
    var state = value.state == null ? 'state 未提供' : 'state ' + displayValue(value.state);
    var fieldValue = value.value == null ? 'value 未提供' : 'value ' + displayValue(value.value);
    return state + ' · ' + fieldValue;
  }

  function cognitionEntityRef(reference, entities) {
    var id = typeof reference === 'string' ? reference : displayValue(reference);
    var entity = (Array.isArray(entities) ? entities : []).find(function (item) { return item && item.entity_id === id; });
    return entity ? displayValue(entity.display_name || entity.name || id) + ' · ' + id : id;
  }

  function statementSourceRefs(statement) {
    var refs = new Set();
    function collect(node) {
      if (!node || typeof node !== 'object') return;
      if (Array.isArray(node)) { node.forEach(collect); return; }
      Object.keys(node).forEach(function (key) {
        if (key === 'evidence_refs' && Array.isArray(node[key])) node[key].forEach(function (ref) { refs.add(ref); });
        else collect(node[key]);
      });
    }
    collect(statement);
    return refs;
  }

  function bindingRole(binding) {
    try { return JSON.parse(binding.source_id).role || ''; } catch (_) { return ''; }
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
    var statements = Array.isArray(document.statements) ? document.statements : [];
    var referenced = statements.map(statementSourceRefs);
    var usedBindings = bindings.filter(function (binding) { return referenced.some(function (refs) { return refs.has(binding.binding_id); }); });
    var bindingRows = bindings.map(function (binding, index) {
      if (!binding || typeof binding !== 'object') return '<li>' + esc(displayValue(binding)) + '</li>';
      var uses = referenced.filter(function (refs) { return refs.has(binding.binding_id); }).length;
      var role = bindingRole(binding);
      var speaker = role === 'user' ? ' · 主人' : (role === 'assistant' ? ' · 顾衍' : '');
      return '<li><details data-l1-source-binding><summary>材料 ' + (index + 1) + ' · ' +
        (uses ? uses + ' 条陈述引用' : '上下文材料，未被陈述引用') + speaker + '</summary>' +
        '<pre>' + esc(JSON.stringify(binding, null, 2)) + '</pre></details></li>';
    }).join('');
    var refRows = sourceRefs.map(function (value) { return '<li>' + esc(displayValue(value)) + '</li>'; }).join('');
    var generation = document.generation || candidate.generation || {};
    var generationRows = Object.keys(generation).sort().map(function (key) {
      return '<li><details><summary>' + esc(key) + '</summary><pre>' +
        esc(typeof generation[key] === 'object' ? JSON.stringify(generation[key], null, 2) : displayValue(generation[key])) + '</pre></details></li>';
    }).join('');
    var kindLabels = { experience: '经历', state: '状态', fact: '事实', preference: '偏好', boundary: '边界',
      intention: '意向', interpretation: '理解', procedure: '做法' };
    var definitions = (generation.catalog_snapshot || {}).definitions || [];
    function classificationLabel(axis, value) {
      var resolved = semanticValue(value);
      var definition = definitions.find(function (item) { return item.axis === axis && item.id === resolved; });
      return definition && definition.label || (axis === 'kinds' ? kindLabels[resolved] : '') || resolved;
    }
    var scopeLabels = { episode: '当次经历', time_range: '明确时间范围', recurring: '按条件反复适用', unknown: '适用时间未明确' };
    var statementRows = statements.map(function (statement, index) {
      var classification = statement.classification || {};
      var kind = semanticValue(classification.kind);
      var time = statement.time || {}, event = time.event || {}, applicability = time.applicability || {};
      var conditions = statement.conditions || {};
      var refs = referenced[index];
      var cognition = statement.cognition && typeof statement.cognition === 'object' && !Array.isArray(statement.cognition) ? statement.cognition : {};
      var entity = (document.entities || []).find(function (item) { return item.entity_id === (statement.subject_ref || {}).value; });
      var subjectRole = entity && (entity.entity_type || {}).value;
      var subject = subjectRole === 'user' ? '主人' : (subjectRole === 'assistant' ? '顾衍' : (entity && entity.name || '主体未明确'));
      var aboutRefs = Array.isArray(cognition.about_refs) ? cognition.about_refs : null;
      var aboutText = aboutRefs ? (aboutRefs.length ? aboutRefs.map(function (reference) { return cognitionEntityRef(reference, document.entities); }).join('、') : '无') : '未提供';
      var questions = Array.isArray(cognition.open_questions) ? cognition.open_questions : null;
      var questionsText = questions ? (questions.length ? questions.map(displayValue).join('；') : '无') : '未提供';
      var basisRefs = (classification.basis || {}).evidence_refs || [];
      var directRefs = new Set(basisRefs.length ? basisRefs : (statement.evidence_refs || []));
      var evidence = bindings.filter(function (binding) { return directRefs.has(binding.binding_id); });
      var needsEvidenceReview = subjectRole === 'user' && (classification.basis || {}).value === 'direct_expression' &&
        evidence.length && evidence.every(function (binding) { return bindingRole(binding) === 'assistant'; });
      return '<li class="chat-memory-l1-statement"><strong>' + esc(statement.content || '（缺少陈述正文）') + '</strong>' +
        '<p>认知主体 subject_ref：' + esc(subject) + ' · ' + esc(cognitionFieldSummary(statement.subject_ref)) + ' · ' + esc(kindLabels[kind] || kind) + ' · ' + esc(scopeLabels[applicability.temporal_scope] || '适用时间未明确') +
        ' · ' + refs.size + ' 项来源引用</p>' +
        '<div class="chat-memory-cognition" data-l1-cognition><p>认知角色：' + esc(cognitionFieldSummary(cognition.role)) +
        ' · 认知状态：' + esc(cognitionFieldSummary(cognition.epistemic_status)) + '</p><p>讨论对象 about_refs：' + esc(aboutText) +
        '</p><p>未解问题 open_questions：' + esc(questionsText) + '</p></div>' +
        (needsEvidenceReview ? '<p class="chat-memory-edit-needed" data-l1-evidence-warning>当前标为主人直接表达，但引用材料均为顾衍发言；请核对主体与证据性质，不能据此确认主人长期偏好。</p>' : '') +
        '<details><summary>分类、时间与条件</summary>' +
        '<p>类型：' + esc(classificationLabel('kinds', classification.kind)) + '；表达：' + esc(classificationLabel('modalities', classification.modality)) +
        '；依据：' + esc(classificationLabel('bases', classification.basis)) + '；稳定性：' + esc(classificationLabel('stabilities', classification.stability)) + '</p>' +
        '<p>事件开始：' + esc(semanticValue(event.start)) + '；事件结束：' + esc(semanticValue(event.end)) + '</p>' +
        '<p>适用开始：' + esc(semanticValue(applicability.start)) + '；适用结束：' + esc(semanticValue(applicability.end)) + '</p>' +
        '<p>条件：' + esc(conditions.text || semanticValue(conditions)) + '</p>' +
        '<pre>' + esc(JSON.stringify(statement, null, 2)) + '</pre></details></li>';
    }).join('');
    var coverage = bundle && bundle.coverage || candidate.coverage || item && item.coverage;
    return '<details class="chat-memory-excerpt-details chat-memory-l1-details" data-l1-statements>' +
      '<summary>结构化记忆 · ' + statements.length + ' 条陈述 · ' + (document.scenes || []).length + ' 个场景</summary>' +
      '<p>这里记录各条陈述的性质；来源核对通过不等于这些结论已被主人确认。</p>' +
      (statementRows ? '<ol>' + statementRows + '</ol>' : '<p>没有提取到独立陈述。</p>') + '</details>' +
      '<details class="chat-memory-excerpt-details chat-memory-l1-details" data-l1-sources>' +
      '<summary>来源定位 · ' + bindings.length + ' 项材料 · ' + usedBindings.length + ' 项被陈述引用</summary>' +
      '<p>绑定材料包含当时提供的上下文；陈述引用才是该陈述标注的依据，引用本身不证明结论正确。</p>' +
      '<p><strong>写入时来源核对状态：</strong>' + esc(sourceState === 'verified' ? '已核对' : displayValue(sourceState)) + '；当前资格在采用时重新检查。</p>' +
      (coverage ? '<p><strong>覆盖：</strong>' + esc(displayValue(coverage)) + '</p>' : '') +
      (bindingRows ? '<ul>' + bindingRows + '</ul>' : '<p>当前文档没有来源绑定；不代表来源已验证。</p>') +
      (refRows ? '<p><strong>候选来源引用：</strong></p><ul>' + refRows + '</ul>' : '') +
      '</details>' +
      '<details class="chat-memory-excerpt-details chat-memory-l1-details" data-l1-technical>' +
      '<summary>技术详情与生成版本</summary><p>版本、校验值与目录快照用于溯源，不代表它们全部参与召回评分。</p>' +
      (generationRows ? '<ul>' + generationRows + '</ul>' : '<p>未提供生成版本信息。</p>') +
      '</details>';
  }

  function renderL1CognitionEditor(item, document, draft) {
    var statements = Array.isArray(document && document.statements) ? document.statements : [];
    var selectedIndex = draft && Number.isInteger(draft.cognitionStatementIndex) ? draft.cognitionStatementIndex : 0;
    if (selectedIndex < 0 || selectedIndex >= statements.length) selectedIndex = 0;
    var statementOptions = statements.map(function (statement, index) {
      var identifier = statement && (statement.statement_id || statement.id) || ('陈述 ' + (index + 1));
      var content = String(statement && statement.content || '').replace(/\s+/g, ' ').slice(0, 54);
      return '<option value="' + index + '"' + (index === selectedIndex ? ' selected' : '') + '>' + esc((index + 1) + '. ' + identifier + (content ? ' · ' + content : '')) + '</option>';
    }).join('');
    var selected = statements[selectedIndex] || {};
    var cognition = selected.cognition && typeof selected.cognition === 'object' && !Array.isArray(selected.cognition) ? selected.cognition : {};
    var role = cognition.role && typeof cognition.role === 'object' && !Array.isArray(cognition.role) ? cognition.role : {};
    var epistemicStatus = cognition.epistemic_status && typeof cognition.epistemic_status === 'object' && !Array.isArray(cognition.epistemic_status) ? cognition.epistemic_status : {};
    var lists = cognitionValueLists(document);
    var idSuffix = String(item && item.id || 'candidate').replace(/[^a-zA-Z0-9_-]/g, '_');
    var roleListId = 'chat-memory-cognition-roles-' + idSuffix;
    var statusListId = 'chat-memory-cognition-statuses-' + idSuffix;
    function datalist(id, values, current) {
      var choices = Array.from(new Set(values.concat(current ? [current] : [])));
      return '<datalist id="' + escAttr(id) + '">' + choices.map(function (value) { return '<option value="' + escAttr(value) + '"></option>'; }).join('') + '</datalist>';
    }
    var disabled = statements.length ? '' : ' disabled';
    return '<section class="chat-memory-cognition-editor" data-l1-cognition-editor>' +
      '<strong>轻量认知编辑 · 选择已有陈述</strong>' +
      '<p class="chat-memory-edit-needed">认知主体 subject_ref 与讨论对象 about_refs 分开保存；不会自动把主人改成顾衍。空白继续保持未知，证据引用不会由此编辑器改写。</p>' +
      '<label class="chat-memory-edit-field">现有陈述<select data-cognition-statement data-cognition-active-index="' + selectedIndex + '"' + disabled + '>' +
        (statementOptions || '<option value="" disabled selected>当前没有可编辑的现有陈述</option>') + '</select></label>' +
      '<div class="chat-memory-cognition-grid">' +
        '<label class="chat-memory-edit-field">认知角色 state<input type="text" data-cognition-field="role.state" value="' + escAttr(cognitionInputText(role.state)) + '"' + disabled + ' /></label>' +
        '<label class="chat-memory-edit-field">认知角色 value<input type="text" list="' + escAttr(roleListId) + '" data-cognition-field="role.value" value="' + escAttr(cognitionInputText(role.value)) + '"' + disabled + ' /></label>' +
        '<label class="chat-memory-edit-field">认知状态 state<input type="text" data-cognition-field="epistemic_status.state" value="' + escAttr(cognitionInputText(epistemicStatus.state)) + '"' + disabled + ' /></label>' +
        '<label class="chat-memory-edit-field">认知状态 value<input type="text" list="' + escAttr(statusListId) + '" data-cognition-field="epistemic_status.value" value="' + escAttr(cognitionInputText(epistemicStatus.value)) + '"' + disabled + ' /></label>' +
        '<label class="chat-memory-edit-field">讨论对象 about_refs · 每行一个 entity_id<textarea rows="3" data-cognition-field="about_refs"' + disabled + '>' + esc(Array.isArray(cognition.about_refs) ? cognition.about_refs.join('\n') : cognitionInputText(cognition.about_refs)) + '</textarea></label>' +
        '<label class="chat-memory-edit-field">未解问题 open_questions · 每行一项<textarea rows="3" data-cognition-field="open_questions"' + disabled + '>' + esc(Array.isArray(cognition.open_questions) ? cognition.open_questions.join('\n') : cognitionInputText(cognition.open_questions)) + '</textarea></label>' +
      '</div>' + datalist(roleListId, lists.cognition_roles, role.value) + datalist(statusListId, lists.cognition_statuses, epistemicStatus.value) +
      (statements.length ? '' : '<p class="chat-memory-edit-needed">当前没有现有陈述；此入口只编辑已有项，可先用完整L1 JSON增加陈述。</p>') +
      '</section>';
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
    bodyFields += renderL1CognitionEditor(item, current, draft);
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

  function syncL1Form(card, document, sceneBodies, cognitionStatementIndex) {
    var title = card.querySelector('[data-l1-title]');
    var body = card.querySelector('[data-l1-body]');
    if (title) title.value = String((document.narrative || {}).title || '');
    if (body) body.value = String((document.narrative || {}).body || '');
    refreshDiarySceneBodies(card, document, sceneBodies);
    syncL1CognitionForm(card, document, cognitionStatementIndex == null ? 0 : cognitionStatementIndex);
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
        cognitionStatementIndex: Number(card.querySelector('[data-cognition-statement]')?.getAttribute('data-cognition-active-index')) || 0,
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
    if (nextMode === 'body') syncL1Form(card, document, sceneBodies, draft && draft.type === 'l1' ? draft.cognitionStatementIndex : 0);
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

  function loadDailyChatMemoryPending(append) {
    var target = document.getElementById('daily-chat-memory-pending');
    if (!target || memoryBusy) return;
    if (!append) loadMemorySourceTasks(target);
    captureMemoryDrafts();
    var epoch = ++memoryLoadEpoch;
    var offset = append ? Number(memoryPage.next_offset || 0) : 0;
    return authFetch(dailyChatMemoryApiBase() + '/api/daily-chat-memory/pending?status=review_visible&limit=100&offset=' + offset)
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
        if (append) {
          var merged = new Map(memoryVisibleItems.map(function(item) { return [item.id, item]; }));
          items.forEach(function(item) { merged.set(item.id, item); }); items = Array.from(merged.values());
        }
        memoryVisibleItems = items.slice();
        memoryPage = data || {};
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
        var paging = document.createElement('div'); paging.setAttribute('data-memory-pagination', '');
        var count = document.createElement('p');
        count.textContent = '已保存候选共 ' + Number(data.saved_total == null ? data.total || 0 : data.saved_total) +
          ' 条；当前列表共 ' + Number(data.total || 0) + ' 条，已加载 ' + memoryVisibleItems.length + ' 条。';
        paging.appendChild(count);
        if (data.has_more) {
          var more = document.createElement('button'); more.type = 'button'; more.textContent = '加载更多候选';
          more.setAttribute('data-memory-load-more', '');
          more.addEventListener('click', function() { more.disabled = true; loadDailyChatMemoryPending(true); });
          paging.appendChild(more);
        }
        target.appendChild(paging);
        restoreDailyChatMemoryModelIoPanels(target);
        syncUncertainLocks();
      })
      .catch(function (e) {
      if (epoch === memoryLoadEpoch) setDailyChatMemoryMessage('读取失败，草稿保留: ' + e.message, 'error');
      var more = target.querySelector('[data-memory-load-more]'); if (more) more.disabled = false;
      });
  }

  function loadMemorySourceTasks(target, append) {
    var panel = document.getElementById('memory-source-tasks');
    if (!panel) {
      panel = document.createElement('details'); panel.id = 'memory-source-tasks';
      var summary = document.createElement('summary'); summary.textContent = '来源提交的记忆准备状态'; panel.appendChild(summary);
      var refresh = document.createElement('button'); refresh.type = 'button'; refresh.textContent = '刷新准备状态';
      refresh.addEventListener('click', function() { loadMemorySourceTasks(target); }); panel.appendChild(refresh);
      var notice = document.createElement('p'); notice.setAttribute('data-source-task-notice', ''); notice.setAttribute('role', 'status'); panel.appendChild(notice);
      var content = document.createElement('div'); content.setAttribute('data-source-task-list', ''); panel.appendChild(content);
      target.parentNode.insertBefore(panel, target);
    }
    restoreMemoryState();
    var list = panel.querySelector('[data-source-task-list]');
    var notice = panel.querySelector('[data-source-task-notice]');
    var epoch = ++memorySourceLoadEpoch; notice.textContent = '正在刷新准备状态…';
    var offset = append ? Number(sourceTaskPage.next_offset || 0) : 0;
    authFetch(dailyChatMemoryApiBase() + '/api/memory-authority/source-tasks?limit=100&offset=' + offset).then(function(res) {
      if (!res || !res.ok) throw new Error('暂不可读'); return res.json();
    }).then(function(data) {
      if (epoch !== memorySourceLoadEpoch) return;
      if (!data || !Array.isArray(data.tasks)) throw new Error('回执格式不符');
      captureSourceTaskEditors(list);
      if (append) {
        var merged = new Map(sourceTaskTasks.map(function(task) { return [task.task_id, task]; }));
        data.tasks.forEach(function(task) { merged.set(task.task_id, task); }); sourceTaskTasks = Array.from(merged.values());
      } else sourceTaskTasks = data.tasks.slice();
      sourceTaskPage = data;
      notice.textContent = '已保存准备任务 ' + Number(data.total || 0) + ' 条，已加载 ' + sourceTaskTasks.length + ' 条。';
      renderSourceTaskList(list, target);
      var more = panel.querySelector('[data-source-task-load-more]');
      if (more) more.remove();
      if (data.has_more) {
        more = document.createElement('button'); more.type = 'button'; more.textContent = '加载更多准备任务';
        more.setAttribute('data-source-task-load-more', '');
        more.addEventListener('click', function() { more.disabled = true; loadMemorySourceTasks(target, true); });
        panel.appendChild(more);
      }
      sourceTaskTasks.forEach(function(task) {
        if (task && task.source_kind === 'chat_submission' && sourceTaskOpenIds.has(String(task.task_id || ''))) {
          var cached = sourceTaskDetails.get(String(task.task_id || ''));
          var state = sourceTaskDrafts.get(String(task.task_id || ''));
          if (!cached || (state && state.unconfirmed) || (cached && sourceTaskRevisionChanged({
              baseRevision: cached.task && cached.task.revision,
              baseDraftRevision: cached.task && cached.task.draft_revision,
            }, task)) || (state && state.dirty && sourceTaskRevisionChanged(state, task))) {
            loadSourceTaskDetail(String(task.task_id || ''), target, true);
          } else if (state && (state.dirty || state.unconfirmed) &&
              sourceTaskRevisionChanged(state, task)) {
            state.conflict = true;
            state.latestServerRevision = task.revision;
            state.lastMessage = '服务器草稿版本已变化；当前未保存内容已保留。请核对后再保存。';
            persistMemoryState();
            renderSourceTaskList(list, target);
          }
        }
      });
    }).catch(function() {
      if (epoch !== memorySourceLoadEpoch) return;
      notice.textContent = '准备状态刷新失败；以下保留上次回执，可能已过期。已有候选和草稿保持。';
      var more = panel.querySelector('[data-source-task-load-more]'); if (more) more.disabled = false;
    });
  }

  function sourceTaskPhaseLabel(task) {
    var phase = String(task && task.phase || 'unknown');
    if (phase === 'queued') return '等待准备';
    if (phase === 'generating') return '准备中';
    if (phase === 'failed') return '准备失败';
    if (phase === 'prepared') return '候选已准备 · 待 Review；尚未正式入库';
    return phase;
  }

  function sourceTaskApiUrl(taskId) {
    return dailyChatMemoryApiBase() + '/api/memory-authority/source-tasks/' + encodeURIComponent(taskId);
  }

  function sourceTaskDraftValues(draft) {
    draft = draft && typeof draft === 'object' ? draft : {};
    var narrative = draft.narrative && typeof draft.narrative === 'object' ? draft.narrative : {};
    var hints = Array.isArray(draft.semantic_hints) ? draft.semantic_hints : [];
    return { title: typeof narrative.title === 'string' ? narrative.title : '',
      body: typeof narrative.body === 'string' ? narrative.body : '',
      hintsRaw: JSON.stringify(hints, null, 2) };
  }

  function sourceTaskDraftPayload(values) {
    values = values || {};
    var body = String(values.body == null ? '' : values.body);
    if (!body.trim()) throw new Error('提交正文不能为空。');
    var draft = { narrative: { body: body } };
    var title = String(values.title == null ? '' : values.title);
    if (title.trim()) draft.narrative.title = title;
    var rawHints = String(values.hintsRaw == null ? '' : values.hintsRaw).trim();
    if (rawHints) {
      var hints;
      try { hints = JSON.parse(rawHints); } catch (_) { throw new Error('语义提示必须是有效 JSON 数组。'); }
      if (!Array.isArray(hints)) throw new Error('语义提示必须是 JSON 数组，例如 [{"text":"..."}]。');
      hints.forEach(function(hint, index) {
        if (!hint || typeof hint !== 'object' || Array.isArray(hint) || typeof hint.text !== 'string' || !hint.text.trim()) {
          throw new Error('第 ' + (index + 1) + ' 条语义提示需要填写非空 text。');
        }
      });
      if (hints.length) draft.semantic_hints = hints;
    }
    return draft;
  }

  function sourceTaskDraftsEqual(left, right) {
    try { return sameValue(sourceTaskDraftPayload(left), sourceTaskDraftPayload(right)); }
    catch (_) { return false; }
  }

  function sourceTaskRawDraftsEqual(left, right) {
    left = left || {}; right = right || {};
    return sameValue({ title: String(left.title == null ? '' : left.title),
      body: String(left.body == null ? '' : left.body),
      hintsRaw: String(left.hintsRaw == null ? '' : left.hintsRaw) },
    { title: String(right.title == null ? '' : right.title),
      body: String(right.body == null ? '' : right.body),
      hintsRaw: String(right.hintsRaw == null ? '' : right.hintsRaw) });
  }

  function sourceTaskRevisionChanged(state, task) {
    if (!state || !task) return false;
    return String(state.baseRevision == null ? '' : state.baseRevision) !== String(task.revision == null ? '' : task.revision) ||
      String(state.baseDraftRevision == null ? '' : state.baseDraftRevision) !== String(task.draft_revision == null ? '' : task.draft_revision);
  }

  function sourceTaskCreateDraftState(detail) {
    var values = sourceTaskDraftValues(detail && detail.draft);
    var task = detail && detail.task || {};
    return { title: values.title, body: values.body, hintsRaw: values.hintsRaw,
      base: cloneValue(values), baseRevision: task.revision, baseDraftRevision: task.draft_revision,
      latestLoadedRevision: task.revision,
      dirty: false, conflict: false, unconfirmed: false, lastMessage: '', lastError: '' };
  }

  function captureSourceTaskDraft(editor) {
    if (!editor) return;
    var taskId = editor.getAttribute('data-chat-submission-editor');
    if (!taskId) return;
    var state = sourceTaskDrafts.get(taskId);
    if (!state) return;
    var title = editor.querySelector('[data-chat-submission-title]');
    var body = editor.querySelector('[data-chat-submission-body]');
    var hints = editor.querySelector('[data-chat-submission-hints]');
    state.title = title ? String(title.value || '') : state.title;
    state.body = body ? String(body.value || '') : state.body;
    state.hintsRaw = hints ? String(hints.value || '') : state.hintsRaw;
    var validation = '';
    try { sourceTaskDraftPayload(state); } catch (error) { validation = error.message; }
    state.dirty = validation ? !sourceTaskRawDraftsEqual(state, state.base || {}) : !sourceTaskDraftsEqual(state, state.base || {});
    state.lastError = validation;
    var dirty = editor.querySelector('[data-chat-submission-dirty]');
    if (dirty) dirty.textContent = state.dirty ? '有未保存修改；刷新状态、读取失败或版本冲突不会替换这些输入。' : '当前内容与服务器已保存草稿一致。';
    var error = editor.querySelector('[data-chat-submission-error]');
    if (error) error.textContent = validation || '';
    var save = editor.querySelector('[data-chat-submission-save]');
    if (save) save.disabled = Boolean(validation || state.saving || state.conflict || state.unconfirmed || !state.dirty);
    persistMemoryState();
  }

  function captureSourceTaskEditors(container) {
    if (!container || !container.querySelectorAll) return;
    container.querySelectorAll('[data-chat-submission-editor]').forEach(captureSourceTaskDraft);
  }

  function sourceTaskReceiptLabel(value, kind) {
    if (value == null || value === '') return kind === 'memory' ? '尚无正式入库回执' : '暂无准备回执';
    var labels = kind === 'memory'
      ? { absent: '尚未入库', pending: 'Review 待审（尚未正式入库）', pending_review: 'Review 待审（尚未正式入库）', review_pending: 'Review 待审（尚未正式入库）', confirmed: '已正式入库', committed: '已正式入库', rejected: '未采用', revoked: '已撤回' }
      : { queued: '排队中', generating: '准备中', prepared: '候选已生成', failed: '准备失败', complete: '已完成' };
    var key = String(value);
    return labels[key] || key;
  }

  function renderSourceTaskList(list, target) {
    captureSourceTaskEditors(list);
    list.replaceChildren();
    var help = document.createElement('p');
    help.textContent = '聊天提交稿与正式记忆采用分开。提交稿可在这里查看和编辑；候选生成后，请到下方现有 Review 卡片完成审核与完整 L1 编辑。行动结果继续显示在同一列表；使用“加载更多准备任务”查看后续记录。';
    list.appendChild(help);
    if (!sourceTaskTasks.length) {
      var empty = document.createElement('p'); empty.textContent = '暂无来源提交任务。'; list.appendChild(empty); return;
    }
    sourceTaskTasks.forEach(function(task) {
      if (task && task.source_kind === 'chat_submission') list.appendChild(renderChatSubmissionTask(task, target));
      else list.appendChild(renderActionSourceTask(task, target));
    });
  }

  function renderActionSourceTask(task, target) {
    var row = document.createElement('div'); row.className = 'chat-memory-card';
    var status = document.createElement('p');
    var labels = { queued:'等待整理', generating:'处理中', prepared:'候选已生成', failed:'整理失败' };
    status.textContent = (labels[task.phase] || task.phase) + ' · ' + task.action_id; row.appendChild(status);
    if (task.issue) { var issue = document.createElement('code'); issue.textContent = task.issue.code || '未完成'; row.appendChild(issue); }
    if (task.candidate_ids && task.candidate_ids.length) {
      var refs = document.createElement('p'); refs.textContent = '对应候选：' + task.candidate_ids.join('、') + '；采用状态以候选和正式记忆回执为准。'; row.appendChild(refs);
    }
    var modelIoIds = Array.isArray(task.candidate_ids)
      ? Array.from(new Set(task.candidate_ids.map(modelIoMemoryId).filter(Boolean))) : [];
    var actions = document.createElement('div'); actions.className = 'chat-memory-card-actions';
    if (task.phase === 'failed' || task.phase === 'generating') {
      var retry = document.createElement('button'); retry.type = 'button'; retry.textContent = task.phase === 'failed' ? '重试整理' : '中断后恢复整理';
      retry.addEventListener('click', async function() {
        if (!window.confirm('重新排队整理此来源；可能再次调用已配置模型，不会重新执行原行动。')) return;
        retry.disabled = true;
        try {
          var res = await authFetch(sourceTaskApiUrl(task.task_id) + '/retry',
            {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({expected_phase:task.phase})});
          if (!res || !res.ok) throw new Error('未重新排队，请刷新核对任务状态');
          loadMemorySourceTasks(target);
        } catch (error) { status.textContent = error.message; retry.disabled = false; }
      }); actions.appendChild(retry);
    }
    modelIoIds.forEach(function (memoryId) {
      var label = modelIoIds.length > 1 ? '模型输入与输出 · ' + memoryId : '模型输入与输出';
      var toggle = createDailyChatMemoryModelIoToggle(memoryId, 'action-candidate', label);
      if (toggle) actions.appendChild(toggle);
    });
    if (actions.childNodes.length) row.appendChild(actions);
    modelIoIds.forEach(function (memoryId) {
      var panel = modelIoPanels.get(modelIoViewKey('action-candidate', memoryId));
      if (panel) { row.appendChild(panel); syncModelIoToggle(panel); }
    });
    return row;
  }

  function renderChatSubmissionTask(task, target) {
    var taskId = String(task.task_id || '');
    var modelIoId = modelIoMemoryId(taskId);
    var row = document.createElement('div'); row.className = 'chat-memory-card';
    row.style.overflowWrap = 'anywhere';
    row.setAttribute('data-chat-submission-task-id', taskId);
    var status = document.createElement('p');
    var shortTaskId = taskId.length > 26 ? taskId.slice(0, 14) + '…' + taskId.slice(-6) : taskId;
    status.textContent = '聊天提交 · ' + sourceTaskPhaseLabel(task) + (shortTaskId ? ' · ' + shortTaskId : '');
    row.appendChild(status);
    if (task.issue) {
      var issue = document.createElement('p'); issue.className = 'chat-memory-stale';
      issue.textContent = '失败原因：' + String(task.issue.code || '未完成') + (task.issue.field_path ? ' · 字段 ' + String(task.issue.field_path) : '');
      row.appendChild(issue);
    }
    if (task.candidate_ids && task.candidate_ids.length) {
      var refs = document.createElement('p');
      refs.textContent = '对应 Review 候选：' + task.candidate_ids.join('、') + '；生成候选不代表正式入库。'; row.appendChild(refs);
    }
    var detail = sourceTaskDetails.get(taskId);
    if (detail && detail.receipt) {
      var receipt = document.createElement('p'); receipt.className = 'chat-memory-card-meta';
      receipt.textContent = '正式入库：' + sourceTaskReceiptLabel(detail.receipt.memory_status, 'memory') +
        ' · 准备：' + sourceTaskReceiptLabel(detail.receipt.preparation_status, 'preparation'); row.appendChild(receipt);
    } else {
      var receiptHint = document.createElement('p'); receiptHint.className = 'chat-memory-card-meta';
      receiptHint.textContent = task.phase === 'prepared'
        ? '正式入库：待 Review 确认后才会写入。'
        : '正式入库：尚未显示回执；打开详情读取权威状态。';
      row.appendChild(receiptHint);
    }
    var actions = document.createElement('div'); actions.className = 'chat-memory-card-actions';
    var toggle = document.createElement('button'); toggle.type = 'button';
    toggle.textContent = sourceTaskOpenIds.has(taskId) ? '收起提交稿' : (task.phase === 'prepared' ? '查看提交回执' : '查看/编辑提交稿');
    toggle.addEventListener('click', function() {
      captureSourceTaskEditors(row);
      if (sourceTaskOpenIds.has(taskId)) sourceTaskOpenIds.delete(taskId);
      else sourceTaskOpenIds.add(taskId);
      renderSourceTaskList(row.parentNode, target);
      if (sourceTaskOpenIds.has(taskId) && !sourceTaskDetails.has(taskId)) loadSourceTaskDetail(taskId, target, true);
    });
    actions.appendChild(toggle);
    if (task.phase === 'failed') {
      var retry = document.createElement('button'); retry.type = 'button'; retry.textContent = '重试准备';
      retry.addEventListener('click', function() { retryChatSubmissionTask(task, retry, target); });
      actions.appendChild(retry);
    }
    if (modelIoId) {
      var modelIoToggle = createDailyChatMemoryModelIoToggle(modelIoId, 'task');
      if (modelIoToggle) actions.appendChild(modelIoToggle);
    }
    row.appendChild(actions);
    if (modelIoId) {
      var existingModelIoPanel = modelIoPanels.get(modelIoViewKey('task', modelIoId));
      if (existingModelIoPanel) {
        row.appendChild(existingModelIoPanel);
        syncModelIoToggle(existingModelIoPanel);
      }
    }
    if (sourceTaskOpenIds.has(taskId)) {
      var detailPanel = document.createElement('div'); detailPanel.className = 'chat-memory-edit-panel';
      detailPanel.setAttribute('data-chat-submission-detail', taskId);
      if (detail) renderChatSubmissionDetail(detailPanel, task, detail, target);
      else {
        var loading = document.createElement('p');
        var detailError = sourceTaskDetailErrors.get(taskId);
        loading.className = detailError ? 'chat-memory-stale' : '';
        loading.textContent = detailError || '正在读取持久提交稿…'; detailPanel.appendChild(loading);
      }
      row.appendChild(detailPanel);
    }
    return row;
  }

  function renderChatSubmissionDetail(panel, task, detail, target) {
    var taskId = String(task.task_id || '');
    var state = sourceTaskDrafts.get(taskId);
    if (!state) {
      state = sourceTaskCreateDraftState(detail);
      sourceTaskDrafts.set(taskId, state);
    }
    var serverTask = detail.task || task;
    if ((state.dirty || state.unconfirmed) && sourceTaskRevisionChanged(state, serverTask)) {
      state.conflict = true;
      state.latestServerRevision = serverTask.revision;
      if (!state.lastMessage) state.lastMessage = '服务器草稿版本已变化；当前未保存内容已保留。请核对后再保存。';
    }
    var receipt = detail.receipt || {};
    var receiptLine = document.createElement('p'); receiptLine.className = 'chat-memory-card-meta';
    receiptLine.textContent = '准备状态：' + sourceTaskReceiptLabel(receipt.preparation_status || serverTask.phase, 'preparation') +
      ' · 正式入库回执：' + sourceTaskReceiptLabel(receipt.memory_status, 'memory') +
      ' · revision ' + String(serverTask.revision == null ? '未知' : serverTask.revision) +
      ' · draft revision ' + String(serverTask.draft_revision == null ? '未知' : serverTask.draft_revision);
    panel.appendChild(receiptLine);
    appendSourceTaskHistory(panel, detail.history);

    if (serverTask.phase === 'prepared' || task.phase === 'prepared') {
      var reviewGuide = document.createElement('p'); reviewGuide.className = 'chat-memory-edit-needed';
      reviewGuide.textContent = '候选已生成并进入待审核阶段，这里只保留提交稿回执。请在下方现有 Review 候选卡片中打开“编辑”，完成完整 L1 编辑并审核；正式记忆只有在 Review 确认后才入库。';
      panel.appendChild(reviewGuide);
      if (state.dirty || state.unconfirmed) {
        var preserved = document.createElement('p'); preserved.className = 'chat-memory-stale';
        preserved.textContent = '你有尚未保存的本地修改，已保留在当前浏览器会话中；准备任务进入待审核后不能再改提交稿。可复制以下修改内容，再到 Review 候选卡片编辑。';
        panel.appendChild(preserved);
        appendSourceTaskDraftPreview(panel, state);
      }
      appendSourceTaskReadOnlyDraft(panel, detail.draft);
      return;
    }

    if (task.phase === 'failed') {
      var failure = document.createElement('p'); failure.className = 'chat-memory-stale';
      failure.textContent = '这次准备失败；提交稿仍保存在服务器，可先修订后保存，再重试准备。重试不会重新执行原聊天或行动。';
      panel.appendChild(failure);
    }
    var editor = document.createElement('div'); editor.setAttribute('data-chat-submission-editor', taskId);
    var notice = document.createElement('p'); notice.setAttribute('data-chat-submission-notice', '');
    notice.className = state.conflict ? 'chat-memory-stale' : 'chat-memory-card-meta';
    notice.textContent = state.lastMessage || (state.dirty ? '有未保存修改；读取或刷新不会覆盖当前输入。' : '编辑的是服务器持久提交稿；保存后会创建新的草稿版本。');
    editor.appendChild(notice);
    var titleLabel = document.createElement('label'); titleLabel.className = 'chat-memory-edit-field'; titleLabel.textContent = '标题（可选）';
    var titleInput = document.createElement('input'); titleInput.type = 'text'; titleInput.setAttribute('data-chat-submission-title', ''); titleInput.value = state.title; titleLabel.appendChild(titleInput); editor.appendChild(titleLabel);
    var bodyLabel = document.createElement('label'); bodyLabel.className = 'chat-memory-edit-field'; bodyLabel.textContent = '完整提交正文';
    var bodyInput = document.createElement('textarea'); bodyInput.setAttribute('data-chat-submission-body', ''); bodyInput.value = state.body; bodyInput.rows = 8; bodyLabel.appendChild(bodyInput); editor.appendChild(bodyLabel);
    var hintsLabel = document.createElement('label'); hintsLabel.className = 'chat-memory-edit-field'; hintsLabel.textContent = '语义提示 JSON（可选，数组元素需包含 text）';
    var hintsInput = document.createElement('textarea'); hintsInput.setAttribute('data-chat-submission-hints', ''); hintsInput.value = state.hintsRaw; hintsInput.rows = 4; hintsLabel.appendChild(hintsInput); editor.appendChild(hintsLabel);
    var dirty = document.createElement('p'); dirty.className = 'chat-memory-edit-needed'; dirty.setAttribute('data-chat-submission-dirty', '');
    dirty.textContent = state.dirty ? '有未保存修改；刷新状态、读取失败或版本冲突不会替换这些输入。' : '当前内容与服务器已保存草稿一致。'; editor.appendChild(dirty);
    var error = document.createElement('p'); error.className = 'chat-memory-stale'; error.setAttribute('data-chat-submission-error', ''); error.textContent = state.lastError || ''; editor.appendChild(error);
    var actions = document.createElement('div'); actions.className = 'chat-memory-card-actions';
    var save = document.createElement('button'); save.type = 'button'; save.textContent = state.saving ? '正在保存…' : '保存提交稿';
    save.setAttribute('data-chat-submission-save', ''); save.disabled = Boolean(state.saving || state.conflict || state.unconfirmed || !state.dirty || state.lastError);
    save.addEventListener('click', function() { saveChatSubmissionDraft(taskId, target, save); }); actions.appendChild(save);
    var refresh = document.createElement('button'); refresh.type = 'button'; refresh.textContent = '读取服务器最新草稿';
    refresh.addEventListener('click', function() { loadSourceTaskDetail(taskId, target, true); }); actions.appendChild(refresh);
    editor.appendChild(actions); panel.appendChild(editor);
    if (state.saving) editor.querySelectorAll('input, textarea, button').forEach(function(input) { input.disabled = true; });
    if (state.conflict && state.latestServerRevision != null) {
      var conflict = document.createElement('p'); conflict.className = 'chat-memory-stale';
      conflict.textContent = '服务器 revision 已到 ' + String(state.latestServerRevision) + '。保留本地编辑；请先读取并核对服务器版本后，再继续保存。';
      panel.appendChild(conflict);
      appendSourceTaskReadOnlyDraft(panel, detail.draft);
      if (String(state.latestLoadedRevision == null ? '' : state.latestLoadedRevision) === String(detail.task.revision == null ? '' : detail.task.revision)) {
        var rebase = document.createElement('button'); rebase.type = 'button';
        rebase.textContent = '保留当前编辑并以服务器版本为基线';
        rebase.addEventListener('click', function() {
          captureSourceTaskDraft(editor);
          state.base = sourceTaskDraftValues(detail.draft);
          state.baseRevision = detail.task.revision;
          state.baseDraftRevision = detail.task.draft_revision;
          state.latestServerRevision = detail.task.revision;
          state.conflict = false; state.unconfirmed = false;
          state.dirty = !sourceTaskDraftsEqual(state, state.base);
          state.lastMessage = '已保留当前输入，并改用服务器最新版本作为 CAS 基线。请核对后再保存。';
          state.lastError = '';
          persistMemoryState();
          renderSourceTaskList(panel.closest('[data-source-task-list]'), target);
        });
        panel.appendChild(rebase);
      } else {
        var awaiting = document.createElement('p'); awaiting.className = 'chat-memory-stale';
        awaiting.textContent = '等待读取服务器最新草稿后，才能选择继续保留当前编辑。'; panel.appendChild(awaiting);
      }
    }
    editor.querySelectorAll('input, textarea').forEach(function(input) {
      input.addEventListener('input', function() { captureSourceTaskDraft(editor); });
    });
  }

  function appendSourceTaskReadOnlyDraft(panel, draft) {
    var values = sourceTaskDraftValues(draft);
    var title = document.createElement('p'); title.textContent = '提交标题：' + (values.title || '（未填写）'); panel.appendChild(title);
    var body = document.createElement('div'); body.className = 'chat-memory-card-body';
    var heading = document.createElement('strong'); heading.textContent = '已保存的完整提交正文'; body.appendChild(heading);
    var text = document.createElement('p'); text.style.whiteSpace = 'pre-wrap'; text.textContent = values.body || '（正文为空）'; body.appendChild(text); panel.appendChild(body);
    if (values.hintsRaw !== '[]') {
      var hints = document.createElement('pre'); hints.style.whiteSpace = 'pre-wrap'; hints.style.overflowWrap = 'anywhere';
      hints.textContent = values.hintsRaw; panel.appendChild(hints);
    }
  }

  function appendSourceTaskDraftPreview(panel, state) {
    var title = document.createElement('p'); title.textContent = '保留的本地标题：' + (state.title || '（未填写）'); panel.appendChild(title);
    var body = document.createElement('div'); body.className = 'chat-memory-card-body';
    var heading = document.createElement('strong'); heading.textContent = '保留的未保存完整正文'; body.appendChild(heading);
    var text = document.createElement('p'); text.style.whiteSpace = 'pre-wrap'; text.textContent = state.body || '（正文为空）'; body.appendChild(text); panel.appendChild(body);
    if (String(state.hintsRaw || '').trim() && String(state.hintsRaw).trim() !== '[]') {
      var hints = document.createElement('pre'); hints.style.whiteSpace = 'pre-wrap'; hints.style.overflowWrap = 'anywhere';
      hints.textContent = '保留的语义提示 JSON：\n' + state.hintsRaw; panel.appendChild(hints);
    }
  }

  function appendSourceTaskHistory(panel, history) {
    if (!Array.isArray(history) || !history.length) return;
    var details = document.createElement('details');
    var summary = document.createElement('summary'); summary.textContent = '提交历史（' + history.length + ' 条）'; details.appendChild(summary);
    history.slice(-10).reverse().forEach(function(entry) {
      var line = document.createElement('p');
      var parts = [];
      if (entry.phase || entry.status) parts.push(String(entry.phase || entry.status));
      if (entry.revision != null) parts.push('revision ' + String(entry.revision));
      if (entry.draft_revision != null) parts.push('draft revision ' + String(entry.draft_revision));
      if (entry.issue && entry.issue.code) parts.push('原因 ' + String(entry.issue.code));
      if (entry.created_at || entry.updated_at) parts.push(String(entry.created_at || entry.updated_at));
      line.textContent = parts.join(' · ') || '提交状态已记录'; details.appendChild(line);
    });
    panel.appendChild(details);
  }

  async function readSourceTaskResponse(res) {
    if (!res) throw new Error('未连接，当前编辑已保留。');
    var data = {};
    try { data = await res.json(); } catch (_) { data = {}; }
    return { ok: Boolean(res.ok), status: Number(res.status || 0), data: data };
  }

  async function fetchSourceTaskDetail(taskId) {
    var result = await readSourceTaskResponse(await authFetch(sourceTaskApiUrl(taskId)));
    if (!result.ok) {
      var code = result.data && (result.data.error || result.data.code);
      throw Object.assign(new Error(String(code || '提交稿读取失败 (HTTP ' + result.status + ')')), { status: result.status, code: code });
    }
    if (!result.data || !result.data.task || !result.data.draft) throw new Error('提交稿详情格式不符；当前编辑已保留。');
    return result.data;
  }

  async function loadSourceTaskDetail(taskId, target, force) {
    if (!taskId) return;
    restoreMemoryState();
    var epoch = (sourceTaskDetailEpochs.get(taskId) || 0) + 1;
    sourceTaskDetailEpochs.set(taskId, epoch);
    var list = document.querySelector('#memory-source-tasks [data-source-task-list]');
    captureSourceTaskEditors(list);
    var prior = sourceTaskDrafts.get(taskId);
    if (prior && prior.dirty) prior.lastMessage = '正在读取服务器状态；当前未保存输入保持不变。';
    try {
      var detail = await fetchSourceTaskDetail(taskId);
      if (sourceTaskDetailEpochs.get(taskId) !== epoch) return detail;
      captureSourceTaskEditors(list);
      sourceTaskDetailErrors.delete(taskId);
      sourceTaskDetails.set(taskId, detail);
      var current = sourceTaskDrafts.get(taskId);
      if (!current || (!current.dirty && !current.unconfirmed)) {
        var next = sourceTaskCreateDraftState(detail);
        if (current && current.saving) next.saving = current.saving;
        sourceTaskDrafts.set(taskId, next);
      } else if (current.unconfirmed &&
          String(current.baseDraftRevision == null ? '' : current.baseDraftRevision) !== String(detail.task.draft_revision == null ? '' : detail.task.draft_revision) &&
          sourceTaskDraftsEqual(current, sourceTaskDraftValues(detail.draft))) {
        var confirmed = sourceTaskCreateDraftState(detail);
        confirmed.lastMessage = '保存响应未确认，但服务器读回已证明草稿内容和新 draft revision 均已持久保存。';
        sourceTaskDrafts.set(taskId, confirmed);
        persistMemoryState();
      } else {
        current.latestServerRevision = detail.task.revision;
        current.latestLoadedRevision = detail.task.revision;
        if (sourceTaskRevisionChanged(current, detail.task)) {
          current.conflict = true;
          current.lastMessage = '服务器草稿版本已变化；当前未保存内容已保留。请核对后再保存。';
        } else if (current.unconfirmed) {
          current.conflict = true;
          current.lastMessage = '保存响应未确认；服务器读回尚未包含当前输入。输入已保留，请核对服务器版本后再继续。';
        }
      }
      if (list) renderSourceTaskList(list, target);
      return detail;
    } catch (error) {
      if (sourceTaskDetailEpochs.get(taskId) !== epoch) return null;
      var state = sourceTaskDrafts.get(taskId);
      if (state) { state.lastMessage = '服务器状态读取失败；当前输入已保留。' + error.message; state.lastError = state.dirty ? state.lastError : error.message; }
      sourceTaskDetailErrors.set(taskId, '提交稿详情读取失败；请重试。' + error.message + (state && state.dirty ? ' 当前未保存输入已保留。' : ''));
      if (list) renderSourceTaskList(list, target);
      else setDailyChatMemoryMessage('提交稿读取失败，编辑已保留: ' + error.message, 'error');
      return null;
    }
  }

  async function saveChatSubmissionDraft(taskId, target, button) {
    var state = sourceTaskDrafts.get(taskId);
    var detail = sourceTaskDetails.get(taskId);
    var list = document.querySelector('#memory-source-tasks [data-source-task-list]');
    var editor = list && list.querySelector('[data-chat-submission-editor="' + taskId.replace(/"/g, '\\"') + '"]');
    captureSourceTaskDraft(editor);
    if (!state || !detail || !state.dirty || state.lastError || state.saving) return;
    if (detail.task && detail.task.phase === 'prepared') {
      state.lastMessage = '候选已准备；请在下方 Review 候选卡片中编辑完整 L1。';
      renderSourceTaskList(list, target); return;
    }
    var draft;
    try { draft = sourceTaskDraftPayload(state); }
    catch (error) { state.lastError = error.message; renderSourceTaskList(list, target); return; }
    var expectedRevision = state.baseRevision;
    var expectedDraftRevision = state.baseDraftRevision;
    var requestId = newDailyChatMemoryRequestId();
    state.saving = true; state.lastMessage = '正在按当前 revision 保存…'; state.lastError = '';
    sourceTaskSaving.add(taskId); persistMemoryState();
    if (button) button.disabled = true;
    try {
      var result = await readSourceTaskResponse(await authFetch(sourceTaskApiUrl(taskId), {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expected_revision: expectedRevision, request_id: requestId, draft: draft })
      }));
      if (!result.ok) {
        var code = result.data && (result.data.error || result.data.code);
        var message = code === 'edit_prepared_candidate_in_review'
          ? '候选已准备，请到下方 Review 候选卡片完成完整 L1 编辑。'
          : code === 'revision_conflict' || result.status === 409
            ? '服务器版本冲突；当前输入已保留。正在读取服务器版本供核对。'
            : String(code || '保存失败 (HTTP ' + result.status + ')') + '；当前输入已保留。';
        state.lastMessage = message; state.lastError = message;
        if (result.status === 409 && code !== 'edit_prepared_candidate_in_review') {
          state.conflict = true;
          state.latestLoadedRevision = null;
          state.latestServerRevision = result.data && result.data.task && result.data.task.revision != null
            ? result.data.task.revision : expectedRevision;
        } else if (result.status >= 500) {
          state.conflict = true; state.unconfirmed = true; state.latestLoadedRevision = null;
          state.latestServerRevision = expectedRevision;
        }
        state.saving = false; sourceTaskSaving.delete(taskId); persistMemoryState();
        if (result.status === 409 && code !== 'edit_prepared_candidate_in_review') await loadSourceTaskDetail(taskId, target, true);
        else if (result.status >= 500) await loadSourceTaskDetail(taskId, target, true);
        else if (result.status === 409) loadDailyChatMemoryPending();
        else if (list) renderSourceTaskList(list, target);
        return;
      }

      var readback = await fetchSourceTaskDetail(taskId);
      if (!sourceTaskDraftsEqual(sourceTaskDraftValues(readback.draft), state)) {
        throw new Error('服务器读回与提交内容不一致；本地编辑仍保留，请读取最新草稿核对。');
      }
      if (sameValue(expectedDraftRevision, readback.task.draft_revision)) {
        throw new Error('服务器读回内容一致，但 draft revision 未变化；本地编辑仍保留，请核对提交回执。');
      }
      sourceTaskDetails.set(taskId, readback);
      var next = sourceTaskCreateDraftState(readback);
      next.lastMessage = '已保存并从服务器读回；draft revision ' + String(readback.task.draft_revision) + '。';
      sourceTaskDrafts.set(taskId, next);
      sourceTaskSaving.delete(taskId); persistMemoryState();
      var taskIndex = sourceTaskTasks.findIndex(function(task) { return String(task.task_id || '') === taskId; });
      if (taskIndex >= 0) sourceTaskTasks[taskIndex] = readback.task;
      if (list) renderSourceTaskList(list, target);
      setDailyChatMemoryMessage(next.lastMessage, 'ok');
      loadMemorySourceTasks(target);
    } catch (error) {
      state.saving = false; state.unconfirmed = true; state.lastMessage = error.message + ' 当前输入已保留。'; state.lastError = error.message;
      state.conflict = true; state.latestLoadedRevision = null;
      state.latestServerRevision = state.latestServerRevision == null ? expectedRevision : state.latestServerRevision;
      sourceTaskSaving.delete(taskId); persistMemoryState();
      if (list) renderSourceTaskList(list, target);
      setDailyChatMemoryMessage('提交稿保存结果尚未核实，编辑已保留: ' + error.message, 'error');
      await loadSourceTaskDetail(taskId, target, true);
    }
  }

  async function retryChatSubmissionTask(task, button, target) {
    if (!window.confirm('重试准备此聊天提交；可能调用已配置模型，但不会重新执行聊天或写入正式记忆。')) return;
    button.disabled = true;
    try {
      var result = await readSourceTaskResponse(await authFetch(sourceTaskApiUrl(task.task_id) + '/retry', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ expected_phase: 'failed' })
      }));
      if (!result.ok) throw new Error(String(result.data && (result.data.error || result.data.code) || '未重新排队，请刷新核对任务状态'));
      loadMemorySourceTasks(target);
    } catch (error) {
      button.disabled = false;
      setDailyChatMemoryMessage('未能重试准备；失败回执和提交稿保持: ' + error.message, 'error');
    }
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
    var modelIoId = modelIoMemoryId(id);
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
    var bindings = l1Document && Array.isArray(l1Document.evidence_bindings) ? l1Document.evidence_bindings : [];
    var hasExcerpt = Boolean(String(candidate.original_excerpt || '').trim());
    var excerpt = String(candidate.original_excerpt || '').trim() || (bindings.length
      ? '已绑定 ' + bindings.length + ' 项来源材料，未提供单独摘录。点击“查看完整原文”读取来源。'
      : (legacy ? '历史候选无原文摘录' : '（无来源片段）'));
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
    var sourceText = bindings.length ? '来源材料 ' + bindings.length + ' 项；具体引用见下方来源定位。'
      : (sourceParts.length ? sourceParts.join(' · ') : '未提供来源定位');
    var sourceHash = String(candidate.source_hash || '').slice(0, 8);
    var preparation = item.preparation_status || candidate.preparation_status;
    var preparationCounts = item.preparation_counts || candidate.preparation_counts || {};
    var preparationLabels = { waiting_budget: '已保留，等待后续处理', stale_source: '来源已变化，保留原稿待核对', processed: '已处理',
      requires_manual_retry: '原响应部分校验失败；完整合法稿已保存，补扫可重试错误部分' };
    var preparationNote = preparation ? '<p class="chat-memory-edit-needed">' + esc(preparationLabels[preparation] || preparation) +
      ' · 已生成 ' + Number(preparationCounts.generated || 0) + ' / 已排队 ' + Number(preparationCounts.queued || 0) +
      ' / 已处理 ' + Number(preparationCounts.processed || 0) + ' / 已采用 ' + Number(preparationCounts.adopted || 0) + '</p>' : '';
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
    var modelIoButton = modelIoId
      ? '<button type="button" data-model-io-toggle data-model-io-id="' + escAttr(modelIoId) + '" data-model-io-kind="review" data-model-io-view-key="' + escAttr(modelIoViewKey('review', modelIoId)) + '" aria-expanded="false">模型输入与输出</button>'
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
            '<div class="chat-memory-column-title">来源依据</div>' +
            '<details class="chat-memory-excerpt-details"' + (legacy && !bindings.length ? ' open' : '') + '>' +
              '<summary>' + (hasExcerpt ? '展开来源摘录' : (bindings.length ? '已绑定来源，未提供单独摘录' : '未提供原文摘录')) + '</summary>' +
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
        preparationNote + '<div class="chat-memory-semantics">' + (l1Document ? renderL1ReviewDetails(item, l1Draft ? l1Draft.document : l1Document) : renderMemorySemantics(item, draft)) + '</div>' + staleNote + blockedNote + editNote +
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
          modelIoButton +
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
        (modelIoId ? '<div data-model-io-slot="review"></div>' : '') +
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
    var note = data.source_authority === 'aizizhu_canonical_snapshot'
      ? '<p class="chat-memory-source-empty">以下为提交时保留的对话来源快照；记录时间不代表事件发生时间，也不代表内容目前仍成立。</p>' : '';
    (data.turns || []).forEach(function (turn) {
      if (turn.text) {
        blocks.push(renderSourceBlock('对话轮次 ' + turn.id + ' · 主人与顾衍', turn.text, turn.truncated, turn.continue_after, 'turn', turn.id));
      } else {
        if (turn.user_text) blocks.push(renderSourceBlock('轮次 ' + turn.id + ' · 主人', turn.user_text, false, -1, null, null));
        if (turn.assistant_text) blocks.push(renderSourceBlock('轮次 ' + turn.id + ' · 顾衍', turn.assistant_text, false, -1, null, null));
      }
    });
    (data.events || []).forEach(function (event) {
      if (event.text) {
        var roleLabel = event.role === 'user' ? '主人' : (event.role === 'assistant' ? '顾衍' : '其他来源');
        var recorded = event.created_at ? ' · 记录时间 ' + event.created_at : '';
        blocks.push(renderSourceBlock('来源 ' + (blocks.length + 1) + ' · ' + roleLabel + recorded, event.text, event.truncated, event.continue_after, 'event', event.id));
      }
    });
    if (!blocks.length) {
      return '<div class="chat-memory-source-empty">未找到可展开的完整原文（来源事件可能已被清理或撤回）。</div>';
    }
    return note + blocks.join('');
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
    var latestRunByDate = {};
    runs.forEach(function(run) { latestRunByDate[run.date] = run; });
    var cursorLine = 'watermark：last_raw_event_id=' + (cursor.last_raw_event_id || 0) + (cursor.updated_at ? '（' + esc(cursor.updated_at) + '）' : '');
    var runLines = '';
    if (!runs.length) {
      runLines = '<div class="chat-memory-run-empty">尚无运行记录。</div>';
    } else {
      runLines = runs.map(function (run) {
        var hard = run.hard_rejects || {};
        var hardText = Object.keys(hard).length ? Object.keys(hard).map(function (k) { return k + ':' + hard[k]; }).join('，') : '0';
        var statusText = run.status === 'zero_candidates' ? '0 条候选（窗口已全部检查）' : (run.status === 'degraded_empty_outputs' ? '空转（模型答复为空，未推水位）' : run.status);
        var progress = run.progress;
        var retryDate = /^\d{4}-\d{2}-\d{2}$/.test(String(run.date || '')) && latestRunByDate[run.date] === run &&
          (run.status === 'needs_review' || run.status === 'failed' || run.status === 'degraded_empty_outputs' ||
            (run.status === 'partial' && (!progress || progress.state === 'stopped')));
        var progressText = progress ? ' · 当前批窗口 ' + Number(progress.current_batch_completed_windows || 0) + '/' + Number(progress.current_batch_total_windows || 0) +
          ' · 已保存候选 ' + Number(progress.generated_candidates_saved || 0) + ' · 剩余已生成待处理 ' + Number(progress.remaining_prepared_candidates || 0) +
          ' · ' + (progress.state === 'complete' ? '该日期来源已整理完成' : progress.state === 'stopped' ? '整理已停止，待核对原因' : '全天仍在整理中') : '';
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
            esc(progressText) + (run.error_category ? ' · 原因 ' + esc(run.error_category) : '') +
          '</div>' + (retryDate ? '<button type="button" class="chat-memory-rerun-btn" data-memory-retry-date="' + escAttr(run.date) +
            '" onclick="rerunDailyChatMemory(this)">重试该日期未完成部分</button>' : '') +
        '</div>';
      }).join('');
    }
    return '<div class="chat-memory-runs">' +
      '<div class="chat-memory-runs-title">最近运行（' + runs.length + ' 条，纯运行信息）</div>' +
      '<div class="chat-memory-runs-cursor">' + cursorLine + '</div>' +
      runLines +
      '<div class="chat-memory-runs-actions">' +
        '<button type="button" class="chat-memory-rerun-btn" onclick="rerunDailyChatMemory(this)">补扫新增区间</button>' +
        '<span class="chat-memory-runs-hint">按窗口连续整理尚未处理区间；复用已生成稿，可能多次调用配置模型并产生费用。</span>' +
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
    var retryDate = button.getAttribute('data-memory-retry-date');
    if (!confirm(retryDate ? '确认重试 ' + retryDate + ' 未完成的整理？会完整读取该日期、复用已保存稿，重新处理失败或缺少就绪结果的窗口，可能产生模型费用。'
      : '确认连续整理新增区间？会复用已生成稿，按窗口调用配置模型并产生费用，直到该日期完成或遇到真实失败。')) return;
    button.disabled = true;
    try {
      var body = { request_id: newDailyChatMemoryRequestId() };
      if (retryDate) { body.date = retryDate; body.retry_failed_windows = true; }
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
      } else if (data.status === 'partial' || data.status === 'needs_review' || data.status === 'failed') {
        setDailyChatMemoryMessage('整理尚未完成：' + String(data.reason || data.error_category || data.status) + '；已生成的候选保持可查看。', 'error');
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
  window.toggleDailyChatMemoryModelIo = toggleDailyChatMemoryModelIo;
  window.loadDailyChatMemorySourceMore = loadDailyChatMemorySourceMore;
  window.loadDailyChatMemoryRuns = loadDailyChatMemoryRuns;
  window.rerunDailyChatMemory = rerunDailyChatMemory;
  window.initDailyChatMemoryTab = initDailyChatMemoryTab;

  if (typeof getActiveTab === 'function' && getActiveTab() === 'chat-memory') {
    initDailyChatMemoryTab();
  }
})();
