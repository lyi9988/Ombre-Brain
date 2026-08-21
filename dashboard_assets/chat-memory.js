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
    if (!target) return;
    target.innerHTML = '<div class="loading">读取候选...</div>';
    return authFetch(dailyChatMemoryApiBase() + '/api/daily-chat-memory/pending?limit=100')
      .then(function (res) {
        if (!res) return;
        return res.json();
      })
      .then(function (data) {
        if (data && data.error) throw new Error(data.error || '读取失败');
        target.innerHTML = renderDailyChatMemoryPending((data && data.items) || []);
      })
      .catch(function (e) {
        target.innerHTML = '<div class="loading">读取失败: ' + esc(e.message) + '</div>';
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
    var candidate = item.candidate || {};
    var id = item.id || '';
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
      '<div class="chat-memory-card' + (editRequired ? ' chat-memory-card-edit-needed' : '') + '" data-candidate-id="' + escAttr(id) + '"' + (editRequired ? ' data-needs-edit="true"' : '') + '>' +
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
              esc((candidate.kind || candidate.candidate_type || 'memory') + ' · ' + (item.date || '') + ' · confidence ' + (candidate.confidence || '')) +
              (sourceHash ? ' · src ' + esc(sourceHash) : '') +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div class="chat-memory-source-panel" hidden>' +
          '<div class="chat-memory-source-loading">读取完整原文...</div>' +
        '</div>' +
        staleNote + blockedNote + editNote +
        '<div class="chat-memory-edit-panel" hidden>' +
          '<label class="chat-memory-edit-field">标题' +
            '<input type="text" data-field="title" value="' + escAttr(candidate.title || id) + '" />' +
          '</label>' +
          '<label class="chat-memory-edit-field">正文（建议记忆，写入记忆桶的内容）' +
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
          sourcePreviewButton +
          '<button type="button"' + confirmDisabled + ' onclick="confirmDailyChatMemory(this, \'' + jsString(id) + '\', \'confirm\')">写入</button>' +
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
      throw new Error(data.error || data.reason || '操作失败 (HTTP ' + res.status + ')');
    }
    return data;
  }

  async function confirmDailyChatMemory(buttonOrId, idOrAction, maybeAction) {
    var button = typeof buttonOrId === 'object' ? buttonOrId : null;
    var id = button ? idOrAction : buttonOrId;
    var action = button ? maybeAction : idOrAction;
    var isReject = action === 'reject';
    var isDefer = action === 'defer';
    var verbText = isDefer ? '暂缓' : (isReject ? '拒绝' : '写入');
    if (!confirm(verbText + '这条候选？' + (isDefer ? '（暂缓不等同拒绝，之后可再处理）' : ''))) return;
    var card = button && button.closest ? button.closest('.chat-memory-card') : null;
    var body = {
      candidate_ids: [id],
      action: action,
      confirm: isReject ? 'REJECT' : (isDefer ? 'DEFER' : 'WRITE'),
      request_id: newDailyChatMemoryRequestId(),
    };
    if (isReject) {
      var reasonSelect = document.getElementById('reject-reason-' + id);
      var noteInput = document.getElementById('reject-note-' + id);
      if (reasonSelect) body.reason = reasonSelect.value;
      if (noteInput && noteInput.value.trim()) body.reason_note = noteInput.value.trim();
    } else if (!isDefer) {
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
      var needsEdit = (data.results || []).filter(function (r) { return r.status === 'needs_owner_edit'; });
      var invalid = (data.results || []).filter(function (r) { return r.status === 'invalid_source'; });
      if (needsEdit.length && !isReject && !isDefer) {
        setDailyChatMemoryMessage('这条候选的建议记忆与原文高度重叠：请先编辑成可读记忆后再写入。', 'error');
        openDailyChatMemoryEdit(card);
        return;
      }
      if (invalid.length) {
        setDailyChatMemoryMessage('候选来源无法核对，未写入；请拒绝这些候选。', 'error');
        loadDailyChatMemoryPending();
        return;
      }
      setDailyChatMemoryMessage('已' + verbText + '候选。', 'ok');
      loadDailyChatMemoryPending();
      if (!isReject && !isDefer) loadBuckets();
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
    var isDefer = action === 'defer';
    var verb = isDefer ? '暂缓' : (isReject ? '拒绝' : '写入');
    if (!confirm('确认批量' + verb + ' ' + ids.length + ' 条候选？此操作不可撤销。')) return;
    var body = {
      candidate_ids: ids,
      action: action,
      confirm: isReject ? 'REJECT' : (isDefer ? 'DEFER' : 'WRITE'),
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
      var needsEdit = (data.results || []).filter(function (r) { return r.status === 'needs_owner_edit'; });
      var applied = (data.results || []).filter(function (r) { return r.status === 'created' || r.status === 'exists' || r.status === 'rejected' || r.status === 'deferred'; }).length;
      if (invalid.length || needsEdit.length) {
        setDailyChatMemoryMessage('批量完成：成功 ' + applied + ' 条；' + invalid.length + ' 条来源无法核对、' + needsEdit.length + ' 条需先编辑，请单独处理。', 'error');
      } else {
        setDailyChatMemoryMessage('批量' + verb + '完成：' + applied + ' 条。', 'ok');
      }
      loadDailyChatMemoryPending();
      if (!isReject && !isDefer) loadBuckets();
    } catch (e) {
      setDailyChatMemoryMessage('批量操作失败: ' + e.message, 'error');
    } finally {
      button.disabled = false;
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
