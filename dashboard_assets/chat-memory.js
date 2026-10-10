(function () {
  var memoryItems = new Map();
  var memoryDrafts = new Map();
  var memoryRequests = new Map();
  var memoryLoadEpoch = 0;
  var memoryBusy = false;

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

  function captureMemoryDraft(card) {
    if (!card) return;
    var id = card.getAttribute('data-candidate-id');
    var item = memoryItems.get(id);
    if (!item) return;
    var prior = memoryDrafts.get(id);
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
      card.querySelectorAll('input[data-field], textarea[data-field]').forEach(function (el) {
        if (locked) { el.disabled = true; el.setAttribute('data-uncertain-lock', 'true'); }
        else if (el.hasAttribute('data-uncertain-lock')) { el.disabled = false; el.removeAttribute('data-uncertain-lock'); }
      });
    });
  }

  async function saveMemoryDraft(id) {
    if (pendingMemoryAction(id)) throw new Error('采用或审核结果尚未确认，请先重试原操作；暂不修改候选。');
    var draft = memoryDrafts.get(id);
    if (!draft) return;
    var key = 'edit:' + id;
    var body = stableMemoryRequest(key, { candidate_id: id, edit: draft.edits, expected_revision: draft.revision });
    var res = await authFetch(dailyChatMemoryApiBase() + '/api/daily-chat-memory/edit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!res) throw new Error('未确认保存结果，请恢复连接后重试；草稿已保留。');
    var data = await res.json();
    if (!res.ok) {
      if (res.status >= 400 && res.status < 500) memoryRequests.delete(key);
      throw new Error(data.error === 'revision_conflict' ? '候选已在其他页面修改，请核对新版本；草稿已保留。' : (data.error || '保存失败'));
    }
    if (data.status !== 'saved' || data.adopted !== false || !data.item) throw new Error('保存回执不完整，请重试核对。');
    memoryItems.set(id, data.item);
    memoryDrafts.delete(id);
    memoryRequests.delete(key);
    var card = Array.from(document.querySelectorAll('.chat-memory-card')).find(function (c) { return c.getAttribute('data-candidate-id') === id; });
    if (card) {
      card.setAttribute('data-revision', String(data.item.authority_revision));
      var savedFields = itemFields(data.item);
      Object.keys(savedFields).forEach(function (field) {
        var input = card.querySelector('[data-field="' + field + '"]');
        if (input) input.value = Array.isArray(savedFields[field]) ? savedFields[field].join(', ') : String(savedFields[field]);
      });
      var bodyEl = card.querySelector('.chat-memory-card-body');
      if (bodyEl) bodyEl.textContent = data.item.candidate.proposed_memory || data.item.candidate.content;
      var titleEl = card.querySelector('.chat-memory-card-head strong');
      if (titleEl) titleEl.textContent = savedFields.title;
    }
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
      setDailyChatMemoryMessage('修改已保存，仍待审核；没有采用入库。', 'ok');
    } catch (e) { setDailyChatMemoryMessage(e.message, 'error'); }
    finally { memoryBusy = false; setMemoryControlsBusy(false); }
  }

  function setMemoryControlsBusy(value) {
    document.querySelectorAll('#daily-chat-memory-pending input, #daily-chat-memory-pending textarea, #daily-chat-memory-pending button, #daily-chat-memory-pending select').forEach(function (el) {
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

  function loadDailyChatMemoryPending() {
    var target = document.getElementById('daily-chat-memory-pending');
    if (!target || memoryBusy) return;
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
    var fields = draft ? draft.fields : itemFields(item);
    Object.assign(candidate, fields, { proposed_memory: fields.content });
    var legacy = legacyNoOriginal(item);
    var blocked = confirmBlocked(item);
    var editRequired = needsOwnerEdit(item);
    var hasSource = hasSourcePreview(item);
    var excerpt = String(candidate.original_excerpt || '').trim() || (legacy ? '历史候选无原文摘录' : '（无来源片段）');
    var proposed = String(candidate.proposed_memory || candidate.content || '').trim() || '（缺少建议记忆）';
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
      '<div class="chat-memory-card' + (editRequired ? ' chat-memory-card-edit-needed' : '') + '" data-candidate-id="' + escAttr(id) + '" data-revision="' + escAttr(revision) + '" data-editing="' + (draft && draft.expanded ? 'true' : 'false') + '"' + (editRequired ? ' data-needs-edit="true"' : '') + '>' +
        '<div class="chat-memory-card-head">' +
          '<label class="chat-memory-select">' +
            '<input type="checkbox" data-select="' + escAttr(id) + '" onchange="refreshDailyChatMemorySelection()" />' +
          '</label>' +
          '<strong>' + esc(candidate.title || id) + '</strong>' +
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
        '<div class="chat-memory-semantics">' + renderMemorySemantics(item, draft) + '</div>' + staleNote + blockedNote + editNote +
        (draft ? '<p class="chat-memory-edit-needed">有未保存修改' + (item.draft_orphan || revision !== item.authority_revision ? '；服务器版本或状态已变化，请核对后再操作' : '') + '。</p>' : '') +
        '<div class="chat-memory-edit-panel"' + (draft && draft.expanded ? '' : ' hidden') + '>' +
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
          '</div>' +
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
      if (!retry && !isReject) for (var id of ids) await saveMemoryDraft(id);
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
      if (data.status === 'rate_limited') { memoryRequests.delete(key); throw new Error('操作太频繁，请稍后重试。'); }
      if (data.status === 'conflict') { memoryRequests.delete(key); throw new Error('版本或请求冲突，请核对候选。'); }
      var expected = isReject ? ['rejected'] : (isDefer ? ['deferred'] : ['created', 'exists']);
      var results = data.results || [];
      var applied = results.filter(function (r) { return expected.includes(r.status); });
      var complete = results.length === ids.length && new Set(results.map(function (r) { return r.id; })).size === ids.length && results.every(function (r) {
        return ids.includes(r.id) && (expected.includes(r.status) || ['revision_conflict', 'needs_owner_edit', 'invalid_source', 'missing', 'rejected', 'deferred', 'commit_failed'].includes(r.status));
      });
      if (complete) {
        memoryRequests.delete(key);
        applied.forEach(function (r) { memoryDrafts.delete(r.id); memoryItems.delete(r.id); });
      }
      var failed = results.filter(function (r) { return !expected.includes(r.status); });
      var detail = failed.some(function (r) { return r.status === 'revision_conflict'; }) ? '版本已变化，请载入新版核对。'
        : failed.some(function (r) { return r.status === 'needs_owner_edit'; }) ? '部分候选需要实际修改正文后才能采用。'
        : '其余未成功或结果未确认，请核对；已保存的修改仍然保留。';
      setDailyChatMemoryMessage(verb + '成功 ' + applied.length + '／' + ids.length + ' 条。' + (applied.length === ids.length ? '' : detail), applied.length === ids.length ? 'ok' : 'error');
      refresh = applied.length === ids.length || !memoryRequests.has(key);
    } catch (e) {
      if (e.definiteRejection) memoryRequests.delete(key);
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
