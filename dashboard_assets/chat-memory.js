(function () {
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

  function legacyNoOriginal(item) {
    var display = item && item.display ? item.display : {};
    return Boolean(display.legacy_no_original);
  }

  function confirmBlocked(item) {
    var display = item && item.display ? item.display : {};
    return Boolean(display.confirm_blocked);
  }

  async function loadDailyChatMemoryPending() {
    var target = document.getElementById('daily-chat-memory-pending');
    if (!target) return;
    target.innerHTML = '<div class="loading">读取候选...</div>';
    try {
      var res = await authFetch(dailyChatMemoryApiBase() + '/api/daily-chat-memory/pending?limit=100');
      if (!res) return;
      var data = await res.json();
      if (!res.ok) throw new Error(data.error || '读取失败');
      target.innerHTML = renderDailyChatMemoryPending(data.items || []);
    } catch (e) {
      target.innerHTML = '<div class="loading">读取失败: ' + esc(e.message) + '</div>';
    }
  }

  function renderDailyChatMemoryPending(items) {
    var toolbar = renderDailyChatMemoryToolbar(items);
    if (!items.length) {
      return '<div class="loading">暂无待确认候选。</div>';
    }
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
      '</div>';
  }

  function renderDailyChatMemoryCard(item) {
    var candidate = item.candidate || {};
    var id = item.id || '';
    var legacy = legacyNoOriginal(item);
    var blocked = confirmBlocked(item);
    var excerpt = String(candidate.original_excerpt || '').trim() || (legacy ? '历史候选无原文摘录' : '（无摘录）');
    var proposed = String(candidate.proposed_memory || candidate.content || '').trim();
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
      ? '<div class="chat-memory-stale">' + esc(String(item.display && item.display.confirm_blocked_reason || '来源无法核对，不可写入，请拒绝。')) + '</div>'
      : '';
    var confirmDisabled = blocked ? ' disabled title="来源无法核对，不可写入"' : '';
    return '' +
      '<div class="chat-memory-card" data-candidate-id="' + escAttr(id) + '">' +
        '<div class="chat-memory-card-head">' +
          '<label class="chat-memory-select">' +
            '<input type="checkbox" data-select="' + escAttr(id) + '" onchange="refreshDailyChatMemorySelection()" />' +
          '</label>' +
          '<strong>' + esc(candidate.title || id) + '</strong>' +
        '</div>' +
        '<div class="chat-memory-card-grid">' +
          '<div class="chat-memory-column">' +
            '<div class="chat-memory-column-title">原文摘录</div>' +
            '<div class="chat-memory-excerpt' + (legacy ? ' chat-memory-legacy' : '') + '">' + esc(excerpt) + '</div>' +
          '</div>' +
          '<div class="chat-memory-column">' +
            '<div class="chat-memory-column-title">建议记忆</div>' +
            '<div class="chat-memory-card-body">' + esc(proposed) + '</div>' +
          '</div>' +
          '<div class="chat-memory-column">' +
            '<div class="chat-memory-column-title">来源</div>' +
            '<div class="chat-memory-card-source">' + esc(sourceText) + '</div>' +
            '<div class="chat-memory-card-meta">' +
              esc((candidate.kind || candidate.candidate_type || 'memory') + ' · ' + (item.date || '') + ' · confidence ' + (candidate.confidence || '')) +
              (sourceHash ? ' · src ' + esc(sourceHash) : '') +
            '</div>' +
          '</div>' +
        '</div>' +
        staleNote + blockedNote +
        '<div class="chat-memory-edit-panel" hidden>' +
          '<label class="chat-memory-edit-field">标题' +
            '<input type="text" data-field="title" value="' + escAttr(candidate.title || id) + '" />' +
          '</label>' +
          '<label class="chat-memory-edit-field">正文（写入记忆桶的内容）' +
            '<textarea data-field="content" rows="4">' + esc(proposed) + '</textarea>' +
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
              '<input type="number" min="0" max="1" step="0.01" data-field="confidence" value="' + escAttr(candidate.confidence || '') + '" />' +
            '</label>' +
          '</div>' +
        '</div>' +
        '<div class="chat-memory-card-actions">' +
          '<button type="button" onclick="toggleDailyChatMemoryEdit(this)">编辑</button>' +
          '<button type="button"' + confirmDisabled + ' onclick="confirmDailyChatMemory(this, \'' + jsString(id) + '\', \'confirm\')">写入</button>' +
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
      domain: dailyChatMemoryField(card, 'domain'),
      tags: dailyChatMemoryField(card, 'tags'),
    };
    var importance = dailyChatMemoryField(card, 'importance');
    var confidence = dailyChatMemoryField(card, 'confidence');
    if (importance) edits.importance = Number(importance);
    if (confidence) edits.confidence = Number(confidence);
    return edits;
  }

  function toggleDailyChatMemoryEdit(button) {
    var card = button && button.closest ? button.closest('.chat-memory-card') : null;
    var panel = card && card.querySelector ? card.querySelector('.chat-memory-edit-panel') : null;
    if (!panel) return;
    panel.hidden = !panel.hidden;
    if (card) card.setAttribute('data-editing', panel.hidden ? 'false' : 'true');
    button.textContent = panel.hidden ? '编辑' : '收起编辑';
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
      if (res.status === 409 && data.status === 'ok') return data;
      throw new Error(data.error || data.reason || '操作失败 (HTTP ' + res.status + ')');
    }
    return data;
  }

  async function confirmDailyChatMemory(buttonOrId, idOrAction, maybeAction) {
    var button = typeof buttonOrId === 'object' ? buttonOrId : null;
    var id = button ? idOrAction : buttonOrId;
    var action = button ? maybeAction : idOrAction;
    var isReject = action === 'reject';
    if (!confirm(isReject ? '拒绝这条候选？' : '写入这条长期记忆候选？')) return;
    var card = button && button.closest ? button.closest('.chat-memory-card') : null;
    var body = {
      candidate_ids: [id],
      action: isReject ? 'reject' : 'confirm',
      confirm: isReject ? 'REJECT' : 'WRITE',
      request_id: newDailyChatMemoryRequestId(),
    };
    if (isReject) {
      var reasonSelect = document.getElementById('reject-reason-' + id);
      var noteInput = document.getElementById('reject-note-' + id);
      if (reasonSelect) body.reason = reasonSelect.value;
      if (noteInput && noteInput.value.trim()) body.reason_note = noteInput.value.trim();
    } else {
      var edits = card && card.getAttribute('data-editing') === 'true' ? readDailyChatMemoryEdits(card) : null;
      if (edits) {
        body.edits = {};
        body.edits[id] = edits;
      }
    }
    try {
      var data = await postDailyChatMemoryConfirm(body);
      if (!data) return;
      if (data.status === 'rate_limited') {
        setDailyChatMemoryMessage('操作太频繁，请稍后再试。', 'error');
        return;
      }
      var invalid = (data.results || []).filter(function (r) { return r.status === 'invalid_source'; });
      if (invalid.length) {
        setDailyChatMemoryMessage('部分候选来源无法核对，未写入；请拒绝这些候选。', 'error');
        loadDailyChatMemoryPending();
        return;
      }
      setDailyChatMemoryMessage(isReject ? '已拒绝候选。' : '已写入候选。', 'ok');
      loadDailyChatMemoryPending();
      if (!isReject) loadBuckets();
    } catch (e) {
      setDailyChatMemoryMessage('操作失败: ' + e.message, 'error');
    }
  }

  async function batchDailyChatMemoryConfirm(button, action) {
    var ids = selectedDailyChatMemoryIds();
    if (!ids.length) {
      setDailyChatMemoryMessage('请先勾选要操作的候选。', 'error');
      return;
    }
    var isReject = action === 'reject';
    var verb = isReject ? '拒绝' : '写入';
    if (!confirm('确认批量' + verb + ' ' + ids.length + ' 条候选？此操作不可撤销。')) return;
    var body = {
      candidate_ids: ids,
      action: isReject ? 'reject' : 'confirm',
      confirm: isReject ? 'REJECT' : 'WRITE',
      request_id: newDailyChatMemoryRequestId(),
    };
    if (isReject) {
      var reasonSelect = document.getElementById('daily-chat-memory-batch-reason');
      var noteInput = document.getElementById('daily-chat-memory-batch-note');
      if (reasonSelect) body.reason = reasonSelect.value;
      if (noteInput && noteInput.value.trim()) body.reason_note = noteInput.value.trim();
    }
    button.disabled = true;
    try {
      var data = await postDailyChatMemoryConfirm(body);
      if (!data) return;
      if (data.status === 'rate_limited') {
        setDailyChatMemoryMessage('操作太频繁，请稍后再试。', 'error');
        return;
      }
      var invalid = (data.results || []).filter(function (r) { return r.status === 'invalid_source'; });
      var applied = (data.results || []).filter(function (r) { return r.status === 'created' || r.status === 'exists' || r.status === 'rejected'; }).length;
      if (invalid.length) {
        setDailyChatMemoryMessage('批量完成：成功 ' + applied + ' 条，' + invalid.length + ' 条来源无法核对未写入；请单独拒绝。', 'error');
      } else {
        setDailyChatMemoryMessage('批量' + verb + '完成：' + applied + ' 条。', 'ok');
      }
      loadDailyChatMemoryPending();
      if (!isReject) loadBuckets();
    } catch (e) {
      setDailyChatMemoryMessage('批量操作失败: ' + e.message, 'error');
    } finally {
      button.disabled = false;
    }
  }

  function initDailyChatMemoryTab() {
    loadDailyChatMemoryPending();
  }

  window.setDailyChatMemoryMessage = setDailyChatMemoryMessage;
  window.loadDailyChatMemoryPending = loadDailyChatMemoryPending;
  window.renderDailyChatMemoryPending = renderDailyChatMemoryPending;
  window.confirmDailyChatMemory = confirmDailyChatMemory;
  window.batchDailyChatMemoryConfirm = batchDailyChatMemoryConfirm;
  window.toggleDailyChatMemoryEdit = toggleDailyChatMemoryEdit;
  window.toggleDailyChatMemorySelectAll = toggleDailyChatMemorySelectAll;
  window.refreshDailyChatMemorySelection = refreshDailyChatMemorySelection;
  window.initDailyChatMemoryTab = initDailyChatMemoryTab;

  if (typeof getActiveTab === 'function' && getActiveTab() === 'chat-memory') {
    initDailyChatMemoryTab();
  }
})();
