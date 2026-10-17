(function (window) {
  'use strict';

  var STAGES = [
    { id: 'auto_l1', label: '自动整理 L1' },
    { id: 'chat_l1', label: '聊天提交 L1' },
    { id: 'action_l1', label: '行动结果 L1' },
    { id: 'diary_l1', label: '日记 L1' },
    { id: 'presentation_l2', label: '第二遍 L2 注入稿' },
    { id: 'chat_summary', label: '聊天摘要' },
    { id: 'activity_summary', label: '活动摘要' },
  ];
  var TASK_OPTIONS = [
    { key: 'memory_l2_jobs_per_run', label: '每轮 L2 后台任务数', minimum: 0, maximum: 200, defaultValue: 1 },
    { key: 'memory_l2_concurrency', label: 'L2 后台最大并发数', minimum: 1, maximum: 200, defaultValue: 1 },
    { key: 'memory_l2_prepare_interval_seconds', label: 'L2 后台检查间隔（秒）', minimum: 1, maximum: 3600, defaultValue: 5 },
  ];
  var CONNECTIONS = [
    { value: 'legacy', label: '沿用现有连接优先级（legacy）' },
    { value: 'reflection', label: '自动记忆（reflection）' },
    { value: 'dehydration', label: '脱水（dehydration）' },
    { value: 'daily', label: '每日任务（daily）' },
  ];
  var THINKING_MODES = [
    { value: 'inherit', label: '继承连接设置' },
    { value: 'default', label: '使用模型默认（不发送 thinking 参数）' },
    { value: 'enabled', label: '开启' },
    { value: 'disabled', label: '关闭' },
  ];
  var ROUTE_FIELDS = ['connection', 'model', 'thinking_mode', 'reasoning_effort', 'max_tokens'];
  var EFFECTIVE_FIELDS = [
    'stage', 'connection', 'model', 'thinking_mode', 'reasoning_effort', 'max_tokens',
    'configured', 'unavailable_reason', 'model_source', 'thinking_source', 'max_tokens_source',
  ];
  var DRAFT_STORAGE_KEY = 'ombre.memory-task-models.draft.v1';
  var state = {
    host: null,
    status: null,
    draftStatus: null,
    saveButton: null,
    taskOptionInputs: {},
    outputSchemaInput: null,
    outputSchemaRevisionNode: null,
    inputs: {},
    effectiveNodes: {},
    routeNotes: {},
    baseline: null,
    baselineRevision: null,
    baselineTaskOptions: null,
    baselineOutputSchema: null,
    baselineOutputSchemaRevision: null,
    outputSchemaContractValid: false,
    dirty: false,
    saving: false,
    loaded: false,
    routeContractValid: false,
    revisionValid: false,
    localDraft: undefined,
    latestServer: null,
    latestServerTaskOptions: null,
    latestServerOutputSchema: null,
    latestServerOutputSchemaRevision: null,
    readSequence: 0,
    rebaseButton: null,
  };

  function isObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  function ownKeysEqual(value, expected) {
    if (!isObject(value)) return false;
    var actual = Object.keys(value).sort();
    var wanted = expected.slice().sort();
    return actual.length === wanted.length && actual.every(function (key, index) {
      return key === wanted[index];
    });
  }

  function defaultRoute() {
    return {
      connection: 'legacy',
      model: '',
      thinking_mode: 'inherit',
      reasoning_effort: '',
      max_tokens: null,
    };
  }

  function defaultTaskOptions() {
    var values = {};
    TASK_OPTIONS.forEach(function (field) { values[field.key] = field.defaultValue; });
    return values;
  }

  function parseTaskOptions(reflection) {
    if (!isObject(reflection)) return { valid: false, values: null };
    var values = {};
    for (var i = 0; i < TASK_OPTIONS.length; i += 1) {
      var field = TASK_OPTIONS[i];
      var raw = Object.prototype.hasOwnProperty.call(reflection, field.key)
        ? reflection[field.key] : field.defaultValue;
      if (typeof raw !== 'number' || !Number.isInteger(raw)
          || raw < field.minimum || raw > field.maximum) {
        return { valid: false, values: null, field: field.key };
      }
      values[field.key] = raw;
    }
    return { valid: true, values: values };
  }

  function cloneJson(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function canonicalJson(value) {
    if (Array.isArray(value)) {
      return '[' + value.map(canonicalJson).join(',') + ']';
    }
    if (isObject(value)) {
      return '{' + Object.keys(value).sort().map(function (key) {
        return JSON.stringify(key) + ':' + canonicalJson(value[key]);
      }).join(',') + '}';
    }
    return JSON.stringify(value);
  }

  function schemaEqual(left, right) {
    if (!isObject(left) || !isObject(right)) return false;
    return canonicalJson(left) === canonicalJson(right);
  }

  function parseOutputSchema(reflection) {
    if (!isObject(reflection) || !isObject(reflection.memory_l2_output_schema)) {
      return { valid: false, value: null, revision: null };
    }
    var revision = typeof reflection.memory_l2_schema_revision === 'string'
      && reflection.memory_l2_schema_revision.trim()
      ? reflection.memory_l2_schema_revision : null;
    if (!revision) return { valid: false, value: null, revision: null };
    return {
      valid: true,
      value: cloneJson(reflection.memory_l2_output_schema),
      revision: revision,
    };
  }

  function parseOutputSchemaInput() {
    if (!state.outputSchemaInput) return { valid: false, value: null, error: '未读取第二遍输出 Schema。' };
    try {
      var value = JSON.parse(state.outputSchemaInput.value);
      if (!isObject(value)) return { valid: false, value: null, error: '第二遍输出 Schema 必须是 JSON 对象。' };
      return { valid: true, value: value, error: '' };
    } catch (_) {
      return { valid: false, value: null, error: '第二遍输出 Schema JSON 语法无效，请修正后再提交。' };
    }
  }

  function outputSchemaChanged(value) {
    return !state.baselineOutputSchema || !schemaEqual(value, state.baselineOutputSchema);
  }

  function currentSchemaTextChanged() {
    var parsed = parseOutputSchemaInput();
    return !parsed.valid || outputSchemaChanged(parsed.value);
  }

  function updateOutputSchemaInput(value) {
    if (!state.outputSchemaInput || !isObject(value)) return;
    state.outputSchemaInput.value = JSON.stringify(value, null, 2);
  }

  function outputSchemaTextMatches(text, value) {
    if (typeof text !== 'string') return true;
    try {
      var parsed = JSON.parse(text);
      return isObject(parsed) && schemaEqual(parsed, value);
    } catch (_) {
      return false;
    }
  }

  function cloneTaskOptions(options) {
    var copy = {};
    TASK_OPTIONS.forEach(function (field) { copy[field.key] = options[field.key]; });
    return copy;
  }

  function taskOptionsEqual(left, right) {
    return !!left && !!right && TASK_OPTIONS.every(function (field) {
      return left[field.key] === right[field.key];
    });
  }

  function changedTaskOptions(before, current) {
    var changed = {};
    TASK_OPTIONS.forEach(function (field) {
      if (!before || current[field.key] !== before[field.key]) changed[field.key] = current[field.key];
    });
    return changed;
  }

  function taskOptionsMatchSubset(expected, actual) {
    return !!expected && !!actual && Object.keys(expected).every(function (key) {
      return expected[key] === actual[key];
    });
  }

  function emptyRoutes() {
    var routes = {};
    STAGES.forEach(function (stage) { routes[stage.id] = defaultRoute(); });
    return routes;
  }

  function cloneRoutes(routes) {
    var copy = {};
    STAGES.forEach(function (stage) {
      var route = routes[stage.id];
      copy[stage.id] = {
        connection: route.connection,
        model: route.model,
        thinking_mode: route.thinking_mode,
        reasoning_effort: route.reasoning_effort,
        max_tokens: route.max_tokens,
      };
    });
    return copy;
  }

  function parseRoutes(value) {
    if (!ownKeysEqual(value, STAGES.map(function (stage) { return stage.id; }))) {
      return { valid: false, routes: null };
    }
    var parsed = {};
    for (var i = 0; i < STAGES.length; i += 1) {
      var stageId = STAGES[i].id;
      var route = value[stageId];
      if (!ownKeysEqual(route, ROUTE_FIELDS)) return { valid: false, routes: null };
      if (['legacy', 'reflection', 'dehydration', 'daily'].indexOf(route.connection) < 0) {
        return { valid: false, routes: null };
      }
      if (typeof route.model !== 'string') return { valid: false, routes: null };
      if (['inherit', 'default', 'enabled', 'disabled'].indexOf(route.thinking_mode) < 0) {
        return { valid: false, routes: null };
      }
      if (['', 'low', 'medium', 'high'].indexOf(route.reasoning_effort) < 0) {
        return { valid: false, routes: null };
      }
      if (route.max_tokens !== null && (!Number.isInteger(route.max_tokens) || route.max_tokens < 1 || route.max_tokens > 1000000)) {
        return { valid: false, routes: null };
      }
      parsed[stageId] = {
        connection: route.connection,
        model: route.model,
        thinking_mode: route.thinking_mode,
        reasoning_effort: route.reasoning_effort,
        max_tokens: route.max_tokens,
      };
    }
    return { valid: true, routes: parsed };
  }

  function routesEqual(left, right) {
    if (!left || !right) return false;
    for (var i = 0; i < STAGES.length; i += 1) {
      var stageId = STAGES[i].id;
      for (var j = 0; j < ROUTE_FIELDS.length; j += 1) {
        var key = ROUTE_FIELDS[j];
        if (left[stageId][key] !== right[stageId][key]) return false;
      }
    }
    return true;
  }

  function parseEffective(value) {
    if (!ownKeysEqual(value, STAGES.map(function (stage) { return stage.id; }))) return null;
    var parsed = {};
    for (var i = 0; i < STAGES.length; i += 1) {
      var stageId = STAGES[i].id;
      var item = value[stageId];
      if (!isObject(item) || item.stage !== stageId || typeof item.configured !== 'boolean') return null;
      for (var j = 0; j < EFFECTIVE_FIELDS.length; j += 1) {
        if (!Object.prototype.hasOwnProperty.call(item, EFFECTIVE_FIELDS[j])) return null;
      }
      parsed[stageId] = {
        stage: item.stage,
        connection: item.connection,
        model: item.model,
        thinking_mode: item.thinking_mode,
        reasoning_effort: item.reasoning_effort,
        max_tokens: item.max_tokens,
        configured: item.configured,
        unavailable_reason: item.unavailable_reason,
        model_source: item.model_source,
        thinking_source: item.thinking_source,
        max_tokens_source: item.max_tokens_source,
      };
    }
    return parsed;
  }

  function parseDraftRoutes(value) {
    if (!ownKeysEqual(value, STAGES.map(function (stage) { return stage.id; }))) return null;
    var parsed = {};
    for (var i = 0; i < STAGES.length; i += 1) {
      var stageId = STAGES[i].id;
      var route = value[stageId];
      if (!ownKeysEqual(route, ROUTE_FIELDS) || typeof route.max_tokens !== 'string' || route.max_tokens.length > 32) return null;
      if (['legacy', 'reflection', 'dehydration', 'daily'].indexOf(route.connection) < 0) return null;
      if (typeof route.model !== 'string' || ['inherit', 'default', 'enabled', 'disabled'].indexOf(route.thinking_mode) < 0) return null;
      if (['', 'low', 'medium', 'high'].indexOf(route.reasoning_effort) < 0) return null;
      parsed[stageId] = {
        connection: route.connection,
        model: route.model,
        thinking_mode: route.thinking_mode,
        reasoning_effort: route.reasoning_effort,
        max_tokens: route.max_tokens,
      };
    }
    return parsed;
  }

  function readSessionDraft() {
    try {
      var raw = window.sessionStorage.getItem(DRAFT_STORAGE_KEY);
      if (!raw) return null;
      var stored = JSON.parse(raw);
      if (!isObject(stored) || (stored.version !== 1 && stored.version !== 2 && stored.version !== 3)) return null;
      var routes = parseDraftRoutes(stored.routes);
      var baseRoutes = parseRoutes(stored.base_routes);
      if (!routes || !baseRoutes.valid) return null;
      var taskOptions = null;
      var baseTaskOptions = null;
      var outputSchemaText = null;
      var baseOutputSchema = null;
      var baseOutputSchemaRevision = null;
      if (stored.version >= 2) {
        if (!isObject(stored.task_options) || !isObject(stored.base_task_options)) return null;
        taskOptions = {};
        for (var i = 0; i < TASK_OPTIONS.length; i += 1) {
          var field = TASK_OPTIONS[i];
          var value = stored.task_options[field.key];
          if (typeof value !== 'string' || value.length > 16) return null;
          taskOptions[field.key] = value;
        }
        var parsedBaseTaskOptions = parseTaskOptions(stored.base_task_options);
        if (!parsedBaseTaskOptions.valid) return null;
        baseTaskOptions = parsedBaseTaskOptions.values;
      }
      if (stored.version >= 3) {
        if (stored.output_schema_text !== null && typeof stored.output_schema_text !== 'string') return null;
        if (stored.base_output_schema !== null && !isObject(stored.base_output_schema)) return null;
        if (stored.base_output_schema_revision !== null
            && (typeof stored.base_output_schema_revision !== 'string' || !stored.base_output_schema_revision.trim())) return null;
        if (typeof stored.output_schema_text === 'string'
            && (!isObject(stored.base_output_schema) || !stored.base_output_schema_revision)) return null;
        outputSchemaText = stored.output_schema_text;
        baseOutputSchema = stored.base_output_schema ? cloneJson(stored.base_output_schema) : null;
        baseOutputSchemaRevision = stored.base_output_schema_revision || null;
      }
      return {
        routes: routes,
        baseRoutes: baseRoutes.routes,
        baseRevision: typeof stored.base_revision === 'string' && stored.base_revision.length > 0
          ? stored.base_revision : null,
        taskOptions: taskOptions,
        baseTaskOptions: baseTaskOptions,
        outputSchemaText: outputSchemaText,
        baseOutputSchema: baseOutputSchema,
        baseOutputSchemaRevision: baseOutputSchemaRevision,
      };
    } catch (_) {
      return null;
    }
  }

  function draftFromInputs() {
    var routes = {};
    STAGES.forEach(function (stage) {
      routes[stage.id] = {
        connection: getInput(stage.id, 'connection').value,
        model: getInput(stage.id, 'model').value,
        thinking_mode: getInput(stage.id, 'thinking_mode').value,
        reasoning_effort: getInput(stage.id, 'reasoning_effort').value,
        max_tokens: getInput(stage.id, 'max_tokens').value.trim(),
      };
    });
    return routes;
  }

  function taskOptionsDraftFromInputs() {
    var values = {};
    TASK_OPTIONS.forEach(function (field) {
      var input = state.taskOptionInputs[field.key];
      values[field.key] = input ? input.value.trim() : String(field.defaultValue);
    });
    return values;
  }

  function outputSchemaDraftFromInput() {
    return state.outputSchemaInput ? state.outputSchemaInput.value : null;
  }

  function persistSessionDraft() {
    if (!state.dirty || !state.baseline) return;
    var outputSchemaText = state.outputSchemaContractValid && currentSchemaTextChanged()
      ? outputSchemaDraftFromInput() : null;
    var draft = {
      version: 3,
      base_revision: state.baselineRevision,
      base_routes: cloneRoutes(state.baseline),
      routes: draftFromInputs(),
      base_task_options: cloneTaskOptions(state.baselineTaskOptions || defaultTaskOptions()),
      task_options: taskOptionsDraftFromInputs(),
      base_output_schema: state.baselineOutputSchema ? cloneJson(state.baselineOutputSchema) : null,
      base_output_schema_revision: state.baselineOutputSchemaRevision,
      output_schema_text: outputSchemaText,
    };
    state.localDraft = {
      routes: draft.routes,
      baseRoutes: draft.base_routes,
      baseRevision: draft.base_revision,
      taskOptions: draft.task_options,
      baseTaskOptions: draft.base_task_options,
      outputSchemaText: draft.output_schema_text,
      baseOutputSchema: draft.base_output_schema,
      baseOutputSchemaRevision: draft.base_output_schema_revision,
    };
    try {
      window.sessionStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(draft));
    } catch (_) {
      // Keep the live in-memory draft even when browser storage is unavailable.
    }
  }

  function clearSessionDraft() {
    try { window.sessionStorage.removeItem(DRAFT_STORAGE_KEY); } catch (_) {}
    state.localDraft = null;
  }

  function draftMatchesRoutes(draftRoutes, routes) {
    if (!draftRoutes || !routes) return false;
    for (var i = 0; i < STAGES.length; i += 1) {
      var stageId = STAGES[i].id;
      for (var j = 0; j < ROUTE_FIELDS.length; j += 1) {
        var key = ROUTE_FIELDS[j];
        var draftValue = draftRoutes[stageId][key];
        var serverValue = key === 'max_tokens'
          ? (routes[stageId][key] === null ? '' : String(routes[stageId][key]))
          : routes[stageId][key];
        if (draftValue !== serverValue) return false;
      }
    }
    return true;
  }

  function getReflectionConfig(config) {
    return isObject(config) && isObject(config.reflection) ? config.reflection : null;
  }

  function textValue(value, emptyLabel) {
    if (value === null || value === undefined || value === '') return emptyLabel || '未设置';
    if (typeof value === 'boolean') return value ? '是' : '否';
    return String(value);
  }

  function getAuthFetch() {
    if (typeof authFetch === 'function') return authFetch;
    return typeof window.authFetch === 'function' ? window.authFetch : null;
  }

  function getBaseUrl() {
    if (typeof BASE === 'string') return BASE;
    return typeof window.BASE === 'string' ? window.BASE : null;
  }

  function makeElement(tag, className, text) {
    var element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function addOption(select, value, label) {
    var option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  }

  function addStyle() {
    if (document.getElementById('memory-task-models-style')) return;
    var style = document.createElement('style');
    style.id = 'memory-task-models-style';
    style.textContent = [
      '#memory-task-models{width:100%;min-width:0;color:var(--text,#3A3530);}',
      '#memory-task-models *{box-sizing:border-box;}',
      '.mtm-shell{width:100%;max-width:1080px;margin:16px auto 0;min-width:0;}',
      '.mtm-heading{margin:0 0 8px;font-size:18px;color:var(--text,#3A3530);}',
      '.mtm-intro,.mtm-note{margin:0 0 12px;color:var(--text-dim,#665D52);font-size:13px;line-height:1.55;}',
      '.mtm-note{padding:10px 12px;border:1px solid var(--border,#D9CCB8);border-radius:10px;background:var(--surface,var(--card,var(--bg,#FDFCF0)));color:var(--text,#3A3530);}',
      '.mtm-status{min-height:20px;margin:10px 0 6px;font-size:13px;line-height:1.5;color:var(--text,#3A3530);}',
      '.mtm-status.is-error{color:var(--negative,#8B4A4A);}',
      '.mtm-status.is-success{color:var(--positive,#4A7C59);}',
      '.mtm-draft-state{min-height:19px;margin:0 0 10px;color:var(--accent-light,var(--accent,#2F4F4F));font-size:12px;}',
      '.mtm-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;}',
      '.mtm-stage{min-width:0;padding:14px;border:1px solid var(--border,#D9CCB8);border-radius:12px;background:var(--card,var(--surface-solid,var(--bg,#F7F2E8)));color:var(--text,#3A3530);}',
      '.mtm-scheduler{margin-top:14px;padding:14px;border:1px solid var(--border,#D9CCB8);border-radius:12px;background:var(--card,var(--surface-solid,var(--bg,#F7F2E8)));color:var(--text,#3A3530);}',
      '.mtm-scheduler h4{margin:0;font-size:14px;} .mtm-scheduler-note{margin:5px 0 0;color:var(--text-dim,#665D52);font-size:12px;line-height:1.5;}',
      '.mtm-stage h4{margin:0;font-size:15px;line-height:1.4;overflow-wrap:anywhere;}',
      '.mtm-stage code{display:block;margin-top:3px;color:var(--text-dim,#665D52);font-size:11px;}',
      '.mtm-fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:10px;}',
      '.mtm-field{display:flex;flex-direction:column;gap:5px;min-width:0;}',
      '.mtm-field label{font-size:12px;color:var(--text,#3A3530);}',
      '.mtm-field input,.mtm-field select{width:100%;min-width:0;min-height:38px;padding:8px 9px;border:1px solid var(--border,#D9CCB8);border-radius:8px;background:var(--bg,#FDFCF0);color:var(--text,#3A3530);font:inherit;font-size:13px;}',
      '.mtm-schema{margin-top:14px;padding:12px 14px;border:1px solid var(--border,#D9CCB8);border-radius:10px;background:var(--card,var(--surface-solid,var(--bg,#F7F2E8)));}',
      '.mtm-schema summary{cursor:pointer;font-size:13px;font-weight:700;}',
      '.mtm-schema-note,.mtm-schema-revision{margin:7px 0;color:var(--text-dim,#665D52);font-size:12px;line-height:1.5;overflow-wrap:anywhere;}',
      '.mtm-schema-label{display:block;margin:8px 0 5px;font-size:12px;}',
      '.mtm-schema-input{display:block;width:100%;min-height:260px;resize:vertical;padding:10px;border:1px solid var(--border,#D9CCB8);border-radius:8px;background:var(--bg,#FDFCF0);color:var(--text,#3A3530);font:12px/1.5 ui-monospace,Consolas,monospace;tab-size:2;}',
      '.mtm-schema-input:focus{outline:2px solid var(--accent,#2F4F4F);outline-offset:1px;}',
      '.mtm-schema-input:disabled{opacity:.65;cursor:not-allowed;}',
      '.mtm-field input:focus,.mtm-field select:focus{outline:2px solid var(--accent,#2F4F4F);outline-offset:1px;}',
      '.mtm-route-note{margin:10px 0 0;color:var(--text-dim,#665D52);font-size:11px;line-height:1.5;overflow-wrap:anywhere;}',
      '.mtm-effective{margin-top:10px;padding-top:8px;border-top:1px solid var(--border,#D9CCB8);min-width:0;}',
      '.mtm-effective-summary{margin:0;min-width:0;color:var(--text,#3A3530);font-size:12px;line-height:1.5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}',
      '.mtm-effective-details,.mtm-more-settings{margin-top:8px;border-top:1px solid var(--border,#D9CCB8);padding-top:7px;}',
      '.mtm-effective-details summary,.mtm-more-settings summary{cursor:pointer;color:var(--text-dim,#665D52);font-size:12px;}',
      '.mtm-effective-title{margin:8px 0 7px;font-size:12px;font-weight:700;}',
      '.mtm-effective-list{display:grid;grid-template-columns:minmax(105px,34%) minmax(0,1fr);gap:4px 8px;margin:0;font-size:11px;line-height:1.45;}',
      '.mtm-effective-list dt{color:var(--text-dim,#665D52);}',
      '.mtm-effective-list dd{margin:0;min-width:0;overflow-wrap:anywhere;word-break:break-word;}',
      '.mtm-unavailable{grid-column:1/-1;margin:4px 0 0;color:var(--negative,#8B4A4A);overflow-wrap:anywhere;}',
      '.mtm-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:14px;}',
      '.mtm-save{min-height:40px;padding:9px 16px;border:0;border-radius:9px;background:var(--accent,#2F4F4F);color:#fff;font:inherit;font-size:14px;font-weight:700;cursor:pointer;}',
      '.mtm-save:disabled{opacity:.58;cursor:wait;}',
      '.mtm-rebase{min-height:40px;padding:9px 14px;border:1px solid var(--border,#D9CCB8);border-radius:9px;background:var(--surface-solid,var(--bg,#FDFCF0));color:var(--text,#3A3530);font:inherit;font-size:13px;cursor:pointer;}',
      '.mtm-rebase[hidden]{display:none;}',
      '@media(max-width:680px){.mtm-shell{margin-top:12px}.mtm-grid{grid-template-columns:minmax(0,1fr)}.mtm-fields{grid-template-columns:minmax(0,1fr)}.mtm-stage{padding:12px}.mtm-actions{display:grid;grid-template-columns:minmax(0,1fr)}.mtm-save,.mtm-rebase{width:100%}.mtm-effective-list{grid-template-columns:minmax(94px,36%) minmax(0,1fr)}}',
    ].join('\n');
    document.head.appendChild(style);
  }

  function createField(stageId, key, label, kind, options) {
    var wrapper = makeElement('div', 'mtm-field');
    var id = 'mtm-' + stageId + '-' + key;
    var labelNode = makeElement('label', '', label);
    labelNode.htmlFor = id;
    wrapper.appendChild(labelNode);
    var input;
    if (kind === 'select') {
      input = document.createElement('select');
      options.forEach(function (item) { addOption(input, item.value, item.label); });
    } else {
      input = document.createElement('input');
      input.type = kind;
      if (kind === 'number') {
        input.min = '1';
        input.max = '1000000';
        input.step = '1';
        input.placeholder = '留空表示未设上限';
        input.inputMode = 'numeric';
      } else if (key === 'model') {
        input.placeholder = '留空继承所选连接的模型';
        input.autocomplete = 'off';
      }
    }
    input.id = id;
    input.dataset.stage = stageId;
    input.dataset.routeField = key;
    input.addEventListener('input', onInput);
    input.addEventListener('change', onInput);
    wrapper.appendChild(input);
    return { wrapper: wrapper, input: input };
  }

  function createTaskOptionField(field) {
    var wrapper = makeElement('div', 'mtm-field');
    var id = 'mtm-' + field.key;
    var label = makeElement('label', '', field.label + '（' + field.minimum + '–' + field.maximum + '）');
    label.htmlFor = id;
    wrapper.appendChild(label);
    var input = document.createElement('input');
    input.type = 'number'; input.id = id;
    input.min = String(field.minimum); input.max = String(field.maximum); input.step = '1';
    input.inputMode = 'numeric'; input.dataset.taskOption = field.key;
    input.addEventListener('input', onInput); input.addEventListener('change', onInput);
    wrapper.appendChild(input);
    state.taskOptionInputs[field.key] = input;
    return wrapper;
  }

  function buildInterface() {
    var host = document.getElementById('memory-task-models');
    if (!host) return false;
    state.host = host;
    if (host.dataset.mtmReady === 'true') return true;
    addStyle();
    host.replaceChildren();
    var shell = makeElement('section', 'mtm-shell');
    shell.setAttribute('aria-labelledby', 'mtm-heading');
    shell.appendChild(makeElement('h3', 'mtm-heading', '记忆任务模型路由'));
    shell.appendChild(makeElement('p', 'mtm-intro', '为 ' + STAGES.length + ' 类记忆任务选择模型与思考模式。模型留空时继承连接；思考强度仅在明确选择时发送，并取决于供应商支持。'));
    shell.appendChild(makeElement('p', 'mtm-note', '使用模型默认表示不发送 thinking 参数。展开“更多设置”可调整连接、思考强度和输出上限；daily 未配置时会显示不可用原因，不会借用其他连接的凭据。'));
    state.status = makeElement('div', 'mtm-status', '正在读取路由配置…');
    state.status.setAttribute('role', 'status');
    state.status.setAttribute('aria-live', 'polite');
    shell.appendChild(state.status);
    state.draftStatus = makeElement('div', 'mtm-draft-state', '');
    state.draftStatus.setAttribute('aria-live', 'polite');
    shell.appendChild(state.draftStatus);
    var grid = makeElement('div', 'mtm-grid');
    STAGES.forEach(function (stage) {
      var card = makeElement('article', 'mtm-stage');
      card.appendChild(makeElement('h4', '', stage.label));
      var fields = makeElement('div', 'mtm-fields mtm-primary-fields');
      var connection = createField(stage.id, 'connection', '连接', 'select', CONNECTIONS);
      var model = createField(stage.id, 'model', '模型', 'text');
      var thinking = createField(stage.id, 'thinking_mode', '思考模式', 'select', THINKING_MODES);
      var effort = createField(stage.id, 'reasoning_effort', '思考强度（可选）', 'select', [
        { value: '', label: '不发送' },
        { value: 'low', label: 'low' },
        { value: 'medium', label: 'medium' },
        { value: 'high', label: 'high' },
      ]);
      var maxTokens = createField(stage.id, 'max_tokens', '最大输出 tokens', 'number');
      [model, thinking].forEach(function (field) {
        fields.appendChild(field.wrapper);
        state.inputs[stage.id + ':' + field.input.dataset.routeField] = field.input;
      });
      card.appendChild(fields);
      var more = document.createElement('details');
      more.className = 'mtm-more-settings';
      more.appendChild(makeElement('summary', '', '更多设置：连接、思考强度、输出上限'));
      var advancedFields = makeElement('div', 'mtm-fields mtm-advanced-fields');
      [connection, effort, maxTokens].forEach(function (field) {
        advancedFields.appendChild(field.wrapper);
        state.inputs[stage.id + ':' + field.input.dataset.routeField] = field.input;
      });
      more.appendChild(advancedFields);
      state.routeNotes[stage.id] = makeElement('p', 'mtm-route-note', '');
      more.appendChild(state.routeNotes[stage.id]);
      card.appendChild(more);
      var effective = makeElement('section', 'mtm-effective');
      effective.setAttribute('aria-label', stage.label + '实际解析结果');
      var effectiveSummary = makeElement('p', 'mtm-effective-summary', '实际生效结果尚未读取');
      effective.appendChild(effectiveSummary);
      var effectiveDetails = document.createElement('details');
      effectiveDetails.className = 'mtm-effective-details';
      effectiveDetails.appendChild(makeElement('summary', '', '来源与详细解析'));
      effectiveDetails.appendChild(makeElement('h5', 'mtm-effective-title', '服务器实际解析结果'));
      var list = makeElement('dl', 'mtm-effective-list');
      state.effectiveNodes[stage.id] = { summary: effectiveSummary, list: list, section: effective };
      effectiveDetails.appendChild(list);
      effective.appendChild(effectiveDetails);
      card.appendChild(effective);
      grid.appendChild(card);
    });
    shell.appendChild(grid);
    var scheduler = makeElement('section', 'mtm-scheduler');
    scheduler.setAttribute('aria-labelledby', 'mtm-scheduler-heading');
    var schedulerHeading = makeElement('h4', '', 'L2 后台任务调度');
    schedulerHeading.id = 'mtm-scheduler-heading';
    scheduler.appendChild(schedulerHeading);
    scheduler.appendChild(makeElement('p', 'mtm-scheduler-note', '设置每轮任务数、不同 Memory 的最大并发数与后台检查间隔；每轮任务数为 0 时暂停准备。相同 Memory 会排队避免重复模型调用；已保存的主人手稿仍由 Memory Authority 保护。'));
    var schedulerFields = makeElement('div', 'mtm-fields');
    TASK_OPTIONS.forEach(function (field) { schedulerFields.appendChild(createTaskOptionField(field)); });
    scheduler.appendChild(schedulerFields);
    shell.appendChild(scheduler);
    var schemaDetails = document.createElement('details');
    schemaDetails.className = 'mtm-schema';
    schemaDetails.appendChild(makeElement('summary', '', '第二遍输出 Schema（高级）'));
    schemaDetails.appendChild(makeElement('p', 'mtm-schema-note', '此 Schema 约束第二遍模型的 JSON 输出结构；程序仍会强制检查完整稿、陈述引用等契约，不会执行 Schema 中的任意代码。'));
    var schemaLabel = makeElement('label', 'mtm-schema-label', '第二遍输出 Schema JSON');
    schemaLabel.htmlFor = 'mtm-memory_l2_output_schema';
    schemaDetails.appendChild(schemaLabel);
    state.outputSchemaInput = document.createElement('textarea');
    state.outputSchemaInput.id = 'mtm-memory_l2_output_schema';
    state.outputSchemaInput.className = 'mtm-schema-input';
    state.outputSchemaInput.rows = 18;
    state.outputSchemaInput.spellcheck = false;
    state.outputSchemaInput.autocapitalize = 'off';
    state.outputSchemaInput.autocomplete = 'off';
    state.outputSchemaInput.disabled = true;
    state.outputSchemaInput.addEventListener('input', onInput);
    state.outputSchemaInput.addEventListener('change', onInput);
    schemaDetails.appendChild(state.outputSchemaInput);
    state.outputSchemaRevisionNode = makeElement('p', 'mtm-schema-revision', 'Schema 修订：尚未读取');
    schemaDetails.appendChild(state.outputSchemaRevisionNode);
    shell.appendChild(schemaDetails);
    var actions = makeElement('div', 'mtm-actions');
    state.rebaseButton = makeElement('button', 'mtm-rebase', '核对服务器新版并保留草稿');
    state.rebaseButton.type = 'button';
    state.rebaseButton.dataset.action = 'rebase-memory-task-models';
    state.rebaseButton.hidden = true;
    state.rebaseButton.addEventListener('click', reconcileServerVersion);
    actions.appendChild(state.rebaseButton);
    state.saveButton = makeElement('button', 'mtm-save', '保存并应用');
    state.saveButton.type = 'button';
    state.saveButton.dataset.action = 'save-memory-task-models';
    state.saveButton.addEventListener('click', saveAndVerify);
    actions.appendChild(state.saveButton);
    shell.appendChild(actions);
    host.appendChild(shell);
    host.dataset.mtmReady = 'true';
    if (state.localDraft === undefined) state.localDraft = readSessionDraft();
    updateSaveButton();
    updateDraftState();
    return true;
  }

  function getInput(stageId, key) {
    return state.inputs[stageId + ':' + key];
  }

  function connectionLabel(value) {
    var match = CONNECTIONS.find(function (item) { return item.value === value; });
    return match ? match.label : textValue(value);
  }

  function thinkingLabel(value) {
    if (value === '') return '未发送 thinking 参数';
    var match = THINKING_MODES.find(function (item) { return item.value === value; });
    return match ? match.label : textValue(value);
  }

  function sourceLabel(value) {
    if (typeof value === 'string' && /(?:[a-z]:\\|\\\\|\/|\.json\b|\.ya?ml\b|\.toml\b)/i.test(value)) {
      return '服务配置来源';
    }
    return textValue(value);
  }

  function updateRouteNote(stageId) {
    if (!state.routeNotes[stageId]) return;
    var connection = getInput(stageId, 'connection').value;
    var model = getInput(stageId, 'model').value;
    var thinking = getInput(stageId, 'thinking_mode').value;
    var effort = getInput(stageId, 'reasoning_effort').value;
    var note = model ? '已指定模型：保存后以此模型请求。' : '模型留空：继承所选连接的模型。';
    if (connection === 'legacy') note += ' legacy 按当前连接优先级解析。';
    if (connection === 'daily') note += ' 若 daily 未配置，将按读回原因显示不可用。';
    if (thinking === 'inherit') note += ' 思考模式继承连接。';
    if (thinking === 'default') note += ' 使用模型默认值，不发送 thinking 参数。';
    if (thinking === 'enabled' || thinking === 'disabled') note += ' 思考模式会显式发送。';
    note += effort ? ' 会显式发送思考强度 ' + effort + '。' : ' 思考强度留空时不发送该参数。';
    state.routeNotes[stageId].textContent = note;
  }

  function formatEffectiveEntry(parent, label, value) {
    var term = makeElement('dt', '', label);
    var description = makeElement('dd', '', textValue(value, '未设置'));
    parent.appendChild(term);
    parent.appendChild(description);
  }

  function renderEffective(effective) {
    STAGES.forEach(function (stage) {
      var nodes = state.effectiveNodes[stage.id];
      if (!nodes) return;
      nodes.list.replaceChildren();
      var item = effective && effective[stage.id];
      if (!item) {
        nodes.summary.textContent = '实际生效结果尚未读取';
        nodes.list.appendChild(makeElement('dt', '', '状态'));
        nodes.list.appendChild(makeElement('dd', '', '尚无有效解析结果'));
        return;
      }
      var oneLine = (item.configured ? '实际' : '连接未就绪') + '：'
        + connectionLabel(item.connection) + ' · ' + textValue(item.model, '模型未解析')
        + ' · ' + thinkingLabel(item.thinking_mode);
      if (item.reasoning_effort) oneLine += ' · 强度 ' + textValue(item.reasoning_effort);
      if (item.max_tokens !== null && item.max_tokens !== undefined) oneLine += ' · 上限 ' + textValue(item.max_tokens) + ' tokens';
      nodes.summary.textContent = oneLine;
      formatEffectiveEntry(nodes.list, '实际连接', connectionLabel(item.connection));
      formatEffectiveEntry(nodes.list, '实际模型', item.model);
      formatEffectiveEntry(nodes.list, '实际思考模式', thinkingLabel(item.thinking_mode));
      formatEffectiveEntry(nodes.list, '实际思考强度', item.reasoning_effort);
      formatEffectiveEntry(nodes.list, '实际最大输出 tokens', item.max_tokens);
      formatEffectiveEntry(nodes.list, '连接可用', item.configured ? '可用' : '不可用');
      formatEffectiveEntry(nodes.list, '模型来源', sourceLabel(item.model_source));
      formatEffectiveEntry(nodes.list, '思考设置来源', sourceLabel(item.thinking_source));
      formatEffectiveEntry(nodes.list, '输出上限来源', sourceLabel(item.max_tokens_source));
      if (!item.configured || item.unavailable_reason) {
        var reason = makeElement('dd', 'mtm-unavailable', '不可用原因：' + textValue(item.unavailable_reason, '服务未提供原因'));
        nodes.list.appendChild(reason);
      }
    });
  }

  function updateInputs(routes) {
    STAGES.forEach(function (stage) {
      var route = routes[stage.id];
      ROUTE_FIELDS.forEach(function (key) {
        var input = getInput(stage.id, key);
        if (!input) return;
        input.value = key === 'max_tokens' ? (route[key] === null ? '' : String(route[key])) : route[key];
      });
      updateRouteNote(stage.id);
    });
  }

  function updateInputsFromDraft(routes) {
    STAGES.forEach(function (stage) {
      var route = routes[stage.id];
      ROUTE_FIELDS.forEach(function (key) {
        var input = getInput(stage.id, key);
        if (input) input.value = route[key];
      });
      updateRouteNote(stage.id);
    });
  }

  function updateTaskOptionInputs(values) {
    TASK_OPTIONS.forEach(function (field) {
      var input = state.taskOptionInputs[field.key];
      if (input) input.value = String(values[field.key]);
    });
  }

  function updateTaskOptionInputsFromDraft(values) {
    if (!values) return;
    TASK_OPTIONS.forEach(function (field) {
      var input = state.taskOptionInputs[field.key];
      if (input) input.value = values[field.key];
    });
  }

  function captureLatestServer(routes, revision, effective, taskOptions, outputSchema, outputSchemaRevision) {
    if (!routes || !revision) return;
    state.latestServer = {
      routes: cloneRoutes(routes),
      revision: revision,
      effective: effective || null,
    };
    state.latestServerTaskOptions = taskOptions ? cloneTaskOptions(taskOptions) : null;
    state.latestServerOutputSchema = isObject(outputSchema) ? cloneJson(outputSchema) : null;
    state.latestServerOutputSchemaRevision = typeof outputSchemaRevision === 'string'
      ? outputSchemaRevision : null;
    if (state.outputSchemaRevisionNode && state.latestServerOutputSchemaRevision) {
      state.outputSchemaRevisionNode.textContent = '服务器新版 Schema 修订（只读）：' + state.latestServerOutputSchemaRevision;
    }
    if (state.rebaseButton) state.rebaseButton.hidden = false;
    setStatus('服务器配置有新版本；本地草稿和原编辑版本已保留。核对新版后可合并未编辑字段。', 'error');
  }

  function setStatus(message, tone) {
    if (!state.status) return;
    state.status.textContent = message;
    state.status.classList.remove('is-error', 'is-success');
    if (tone === 'error') state.status.classList.add('is-error');
    if (tone === 'success') state.status.classList.add('is-success');
  }

  function updateDraftState() {
    if (!state.draftStatus) return;
    if (state.saving) {
      state.draftStatus.textContent = '正在保存；本地草稿保留至读回核对完成。';
    } else if (state.dirty) {
      state.draftStatus.textContent = state.localDraft
        ? '有未保存更改；路由、调度和 Schema 草稿只保存在此标签页会话中，核对状态和失败不会清除。'
        : '有未保存更改；当前页面会保留这份草稿。';
    } else if (state.loaded && state.routeContractValid) {
      state.draftStatus.textContent = '当前设置与最近一次服务器读回一致。';
    } else {
      state.draftStatus.textContent = '';
    }
  }

  function updateSaveButton() {
    if (!state.saveButton) return;
    state.saveButton.disabled = state.saving || !state.routeContractValid || !state.revisionValid;
    state.saveButton.textContent = state.saving ? '保存并核对中…' : '保存并应用';
  }

  function applyConfig(config, options) {
    var reflection = getReflectionConfig(config);
    var parsedRoutes = parseRoutes(reflection && reflection.memory_task_routes);
    var parsedTaskOptions = parseTaskOptions(reflection);
    var parsedOutputSchema = parseOutputSchema(reflection);
    var revision = isObject(reflection) && typeof reflection.memory_task_revision === 'string'
      && reflection.memory_task_revision.length > 0 ? reflection.memory_task_revision : null;
    var effective = parseEffective(reflection && reflection.memory_task_effective);
    state.outputSchemaContractValid = parsedOutputSchema.valid;
    if (state.outputSchemaInput) state.outputSchemaInput.disabled = !parsedOutputSchema.valid;
    if (effective) renderEffective(effective);
    else if (!state.dirty && !(options && options.keepError)) renderEffective(null);

    if (!parsedRoutes.valid || !parsedTaskOptions.valid) {
      if (!state.dirty) {
        state.routeContractValid = false;
        state.revisionValid = false;
        state.loaded = false;
        setStatus('服务尚未返回完整的 ' + STAGES.length + ' 阶段路由或有效的 L2 后台频率；为避免覆盖不完整设置，暂不能保存。', 'error');
      }
      updateSaveButton();
      updateDraftState();
      return { routes: null, taskOptions: null, effective: effective, routeValid: false };
    }

    state.routeContractValid = true;
    if (state.dirty || state.saving) {
      if (revision && (revision !== state.baselineRevision
          || !routesEqual(parsedRoutes.routes, state.baseline)
          || !taskOptionsEqual(parsedTaskOptions.values, state.baselineTaskOptions)
          || (parsedOutputSchema.valid && (parsedOutputSchema.revision !== state.baselineOutputSchemaRevision
            || !schemaEqual(parsedOutputSchema.value, state.baselineOutputSchema))))) {
        captureLatestServer(parsedRoutes.routes, revision, effective, parsedTaskOptions.values,
          parsedOutputSchema.value, parsedOutputSchema.revision);
      }
      updateSaveButton();
      updateDraftState();
      return { routes: parsedRoutes.routes, taskOptions: parsedTaskOptions.values, effective: effective, routeValid: true };
    }

    if (!revision) {
      state.revisionValid = false;
      state.loaded = false;
      setStatus('服务尚未返回配置版本，无法保护并发修改；暂不能保存。', 'error');
      updateSaveButton();
      updateDraftState();
      return { routes: parsedRoutes.routes, effective: effective, routeValid: true, revisionValid: false };
    }

    state.baseline = cloneRoutes(parsedRoutes.routes);
    state.baselineRevision = revision;
    state.baselineTaskOptions = cloneTaskOptions(parsedTaskOptions.values);
    state.baselineOutputSchema = parsedOutputSchema.valid ? cloneJson(parsedOutputSchema.value) : null;
    state.baselineOutputSchemaRevision = parsedOutputSchema.revision;
    state.revisionValid = true;
    state.loaded = true;
    updateInputs(parsedRoutes.routes);
    updateTaskOptionInputs(parsedTaskOptions.values);
    if (parsedOutputSchema.valid) updateOutputSchemaInput(parsedOutputSchema.value);
    if (state.outputSchemaRevisionNode) {
      state.outputSchemaRevisionNode.textContent = parsedOutputSchema.valid
        ? 'Schema 修订（只读）：' + parsedOutputSchema.revision
        : 'Schema 修订：服务尚未返回有效 Schema 与修订号';
    }
    if (state.localDraft) {
      var localTaskOptions = state.localDraft.taskOptions;
      var taskOptionsMatch = !localTaskOptions || TASK_OPTIONS.every(function (field) {
        return localTaskOptions[field.key] === String(parsedTaskOptions.values[field.key]);
      });
      var localSchemaText = state.localDraft.outputSchemaText;
      var outputSchemaMatch = localSchemaText === null
        || (parsedOutputSchema.valid && outputSchemaTextMatches(localSchemaText, parsedOutputSchema.value));
      if (draftMatchesRoutes(state.localDraft.routes, parsedRoutes.routes)
          && taskOptionsMatch && outputSchemaMatch) {
        clearSessionDraft();
      } else {
        state.baseline = state.localDraft.baseRoutes || cloneRoutes(parsedRoutes.routes);
        state.baselineRevision = state.localDraft.baseRevision;
        state.baselineTaskOptions = state.localDraft.baseTaskOptions
          || cloneTaskOptions(parsedTaskOptions.values);
        state.baselineOutputSchema = state.localDraft.baseOutputSchema
          || (parsedOutputSchema.valid ? cloneJson(parsedOutputSchema.value) : null);
        state.baselineOutputSchemaRevision = state.localDraft.baseOutputSchemaRevision
          || parsedOutputSchema.revision;
        state.revisionValid = !!state.baselineRevision;
        state.dirty = true;
        updateInputsFromDraft(state.localDraft.routes);
        if (localTaskOptions) updateTaskOptionInputsFromDraft(localTaskOptions);
        if (typeof localSchemaText === 'string') state.outputSchemaInput.value = localSchemaText;
        if (revision !== state.baselineRevision || !routesEqual(state.baseline, parsedRoutes.routes)
            || !taskOptionsEqual(state.baselineTaskOptions, parsedTaskOptions.values)
            || (parsedOutputSchema.valid && (parsedOutputSchema.revision !== state.baselineOutputSchemaRevision
              || !schemaEqual(parsedOutputSchema.value, state.baselineOutputSchema)))) {
          captureLatestServer(parsedRoutes.routes, revision, effective, parsedTaskOptions.values,
            parsedOutputSchema.value, parsedOutputSchema.revision);
        } else {
          setStatus('已恢复此标签页的未保存草稿；编辑版本保持原样，保存后会读回核对。');
        }
        updateSaveButton();
        updateDraftState();
        return { routes: parsedRoutes.routes, taskOptions: parsedTaskOptions.values, effective: effective, routeValid: true, revisionValid: true };
      }
    }
    state.latestServer = null;
    if (state.rebaseButton) state.rebaseButton.hidden = true;
    state.latestServerOutputSchema = null;
    state.latestServerOutputSchemaRevision = null;
    if (!(options && options.keepStatus)) {
      setStatus(parsedOutputSchema.valid
        ? '记忆任务路由、L2 后台选项与第二遍输出 Schema 已从服务器读取。编辑后保存会再次读取并核对。'
        : '路由配置已读取；服务尚未返回有效的第二遍输出 Schema，暂不能编辑该项。');
    }
    updateSaveButton();
    updateDraftState();
    state.baselineTaskOptions = cloneTaskOptions(parsedTaskOptions.values);
    return { routes: parsedRoutes.routes, taskOptions: parsedTaskOptions.values, effective: effective, routeValid: true };
  }

  function onInput(event) {
    var input = event.currentTarget;
    var stageId = input.dataset.stage;
    var taskOption = input.dataset.taskOption;
    var outputSchema = input === state.outputSchemaInput;
    if (!stageId && !taskOption && !outputSchema) return;
    state.dirty = true;
    persistSessionDraft();
    if (stageId) updateRouteNote(stageId);
    updateDraftState();
  }

  function responseErrorLabel(response, action) {
    if (!response) return action + '未完成：认证失效或连接中断。草稿已保留。';
    return action + '失败（HTTP ' + Number(response.status || 0) + '）。草稿已保留。';
  }

  async function requestConfig() {
    var fetchWithAuth = getAuthFetch();
    var baseUrl = getBaseUrl();
    if (!fetchWithAuth || baseUrl === null) {
      throw new Error('dashboard_auth_unavailable');
    }
    return fetchWithAuth(baseUrl + '/api/config');
  }

  async function loadConfig() {
    var sequence = ++state.readSequence;
    try {
      var response = await requestConfig();
      if (!response || !response.ok) {
        if (sequence === state.readSequence && !state.dirty) setStatus(responseErrorLabel(response, '读取配置'), 'error');
        return;
      }
      var config = await response.json();
      if (sequence !== state.readSequence) return;
      applyConfig(config);
    } catch (_) {
      if (sequence === state.readSequence && !state.dirty) setStatus('读取配置失败；如果已有草稿，草稿仍保留。', 'error');
    }
  }

  function collectRoutes() {
    var routes = {};
    for (var i = 0; i < STAGES.length; i += 1) {
      var stageId = STAGES[i].id;
      var maxInput = getInput(stageId, 'max_tokens');
      var maxRaw = maxInput.value.trim();
      var maxTokens = null;
      if (maxRaw !== '') {
        maxTokens = Number(maxRaw);
        if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 1000000) {
          return { valid: false, error: '最大输出 tokens 必须是 1 到 1,000,000 之间的整数；当前草稿已保留。' };
        }
      }
      routes[stageId] = {
        connection: getInput(stageId, 'connection').value,
        model: getInput(stageId, 'model').value,
        thinking_mode: getInput(stageId, 'thinking_mode').value,
        reasoning_effort: getInput(stageId, 'reasoning_effort').value,
        max_tokens: maxTokens,
      };
    }
    var validated = parseRoutes(routes);
    if (!validated.valid) return { valid: false, error: '路由草稿格式无效；修改内容仍保留。' };
    return { valid: true, routes: validated.routes };
  }

  function collectTaskOptions() {
    var values = {};
    for (var i = 0; i < TASK_OPTIONS.length; i += 1) {
      var field = TASK_OPTIONS[i];
      var input = state.taskOptionInputs[field.key];
      var raw = input ? input.value.trim() : '';
      var value = raw === '' ? NaN : Number(raw);
      if (!Number.isInteger(value) || value < field.minimum || value > field.maximum) {
        return {
          valid: false,
          error: field.label + '必须是 ' + field.minimum + ' 到 ' + field.maximum + ' 之间的整数；草稿已保留。',
        };
      }
      values[field.key] = value;
    }
    return { valid: true, values: values };
  }

  function effectiveMismatch(routes, effective) {
    for (var i = 0; i < STAGES.length; i += 1) {
      var stageId = STAGES[i].id;
      var route = routes[stageId];
      var actual = effective[stageId];
      if (!actual.configured) continue;
      if (route.model && actual.model !== route.model) return stageId + ' 的实际模型与设置不一致';
      var expectedThinkingMode = route.thinking_mode === 'default' ? '' : route.thinking_mode;
      if (route.thinking_mode !== 'inherit' && actual.thinking_mode !== expectedThinkingMode) {
        return stageId + ' 的实际思考模式与设置不一致';
      }
      if (route.reasoning_effort && actual.reasoning_effort !== route.reasoning_effort) {
        return stageId + ' 的实际思考强度与设置不一致';
      }
      if (route.max_tokens !== null && actual.max_tokens !== route.max_tokens) {
        return stageId + ' 的实际最大输出 tokens 与设置不一致';
      }
    }
    return '';
  }

  function readbackError(config, expected) {
    var reflection = getReflectionConfig(config);
    var routes = parseRoutes(reflection && reflection.memory_task_routes);
    var options = parseTaskOptions(reflection);
    var outputSchema = parseOutputSchema(reflection);
    var effective = parseEffective(reflection && reflection.memory_task_effective);
    var revision = isObject(reflection) && typeof reflection.memory_task_revision === 'string'
      && reflection.memory_task_revision.length > 0 ? reflection.memory_task_revision : null;
    if (!routes.valid || !options.valid || !outputSchema.valid) {
      return { message: '保存请求已确认，但服务器未返回完整的任务设置或第二遍输出 Schema；不能确认已应用。草稿已保留。', effective: effective, routes: routes.valid ? routes.routes : null, taskOptions: options.valid ? options.values : null, revision: revision, outputSchema: outputSchema.valid ? outputSchema.value : null, outputSchemaRevision: outputSchema.revision };
    }
    if (expected.routes && !routesEqual(routes.routes, expected.routes)) {
      return { message: '保存请求已确认，但服务器路由读回与草稿不一致；不能确认已应用。草稿已保留。', effective: effective, routes: routes.routes, taskOptions: options.values, revision: revision, outputSchema: outputSchema.value, outputSchemaRevision: outputSchema.revision };
    }
    if (!taskOptionsMatchSubset(expected.taskOptionsPatch, options.values)) {
      return { message: '保存请求已确认，但 L2 后台选项读回与草稿不一致；不能确认已应用。草稿已保留。', effective: effective, routes: routes.routes, taskOptions: options.values, revision: revision, outputSchema: outputSchema.value, outputSchemaRevision: outputSchema.revision };
    }
    if (expected.taskChanged && !revision) {
      return { message: '保存请求已确认，但服务器未返回任务配置版本；应用状态无法确认。草稿已保留。', effective: effective, routes: routes.routes, taskOptions: options.values, revision: revision, outputSchema: outputSchema.value, outputSchemaRevision: outputSchema.revision };
    }
    if (expected.routes) {
      if (!effective) {
        return { message: '路由已写入，但服务器未返回完整的实际解析结果；应用状态无法确认。草稿已保留。', effective: null, routes: routes.routes, taskOptions: options.values, revision: revision, outputSchema: outputSchema.value, outputSchemaRevision: outputSchema.revision };
      }
      var mismatch = effectiveMismatch(routes.routes, effective);
      if (mismatch) {
        return { message: '路由已写入，但' + mismatch + '；应用状态无法确认。草稿已保留。', effective: effective, routes: routes.routes, taskOptions: options.values, revision: revision, outputSchema: outputSchema.value, outputSchemaRevision: outputSchema.revision };
      }
    }
    if (expected.outputSchema && !schemaEqual(outputSchema.value, expected.outputSchema)) {
      return { message: '保存请求已确认，但第二遍输出 Schema 读回与草稿不一致；不能确认已应用。草稿已保留。', effective: effective, routes: routes.routes, taskOptions: options.values, revision: revision, outputSchema: outputSchema.value, outputSchemaRevision: outputSchema.revision };
    }
    if (expected.schemaChanged && outputSchema.revision === expected.baseOutputSchemaRevision) {
      return { message: '第二遍输出 Schema 内容读回一致，但修订号未变化；不能确认新版本已保存。草稿已保留。', effective: effective, routes: routes.routes, taskOptions: options.values, revision: revision, outputSchema: outputSchema.value, outputSchemaRevision: outputSchema.revision };
    }
    return { message: '', effective: effective, routes: routes.routes, taskOptions: options.values, revision: revision,
      outputSchema: outputSchema.value, outputSchemaRevision: outputSchema.revision };
  }

  function verifiedSaveMessage(check, changes) {
    var saved = [];
    if (changes.taskChanged) saved.push(changes.routesChanged ? '任务路由与 L2 后台选项' : 'L2 后台选项');
    if (changes.schemaChanged) saved.push('第二遍输出 Schema（修订 ' + check.outputSchemaRevision + '）');
    var unavailable = changes.routesChanged && check.effective
      ? STAGES.filter(function (stage) { return !check.effective[stage.id].configured; }) : [];
    if (unavailable.length) {
      return '已保存并应用，并从服务器读回' + saved.join('、') + '；部分任务连接未就绪（' + unavailable.length + '/' + STAGES.length + '），实际解析已读回，详细原因见对应阶段。';
    }
    return '已保存并应用，并从服务器读回：' + saved.join('、') + (changes.routesChanged
      ? '；' + STAGES.length + ' 阶段的实际解析结果也已核对。' : '。');
  }

  async function saveAndVerify() {
    if (state.saving || !state.routeContractValid || !state.revisionValid || !state.baselineRevision) return;
    var collected = collectRoutes();
    if (!collected.valid) {
      setStatus(collected.error, 'error');
      return;
    }
    var routes = collected.routes;
    var collectedOptions = collectTaskOptions();
    if (!collectedOptions.valid) {
      setStatus(collectedOptions.error, 'error');
      return;
    }
    var taskOptions = collectedOptions.values;
    var taskOptionsPatch = changedTaskOptions(state.baselineTaskOptions, taskOptions);
    var routesChanged = !routesEqual(state.baseline, routes);
    var taskChanged = routesChanged || Object.keys(taskOptionsPatch).length > 0;
    var schemaInput = null;
    var schemaChanged = false;
    if (state.outputSchemaContractValid && state.outputSchemaInput && !state.outputSchemaInput.disabled) {
      schemaInput = parseOutputSchemaInput();
      if (!schemaInput.valid) {
        setStatus(schemaInput.error, 'error');
        return;
      }
      schemaChanged = outputSchemaChanged(schemaInput.value);
    }
    if (schemaChanged && !state.baselineOutputSchemaRevision) {
      setStatus('未读取第二遍输出 Schema 修订，不能安全保存；本地草稿已保留。', 'error');
      return;
    }
    if (!taskChanged && !schemaChanged) {
      state.dirty = false;
      clearSessionDraft();
      setStatus('当前路由、L2 后台选项与第二遍输出 Schema 均与服务器读回一致，无待保存更改。', 'success');
      updateDraftState();
      return;
    }
    var changes = { taskChanged: taskChanged, routesChanged: routesChanged, schemaChanged: schemaChanged };
    var schemaSnapshot = schemaInput && schemaInput.valid ? cloneJson(schemaInput.value) : null;
    state.dirty = true;
    persistSessionDraft();
    state.saving = true;
    updateSaveButton();
    updateDraftState();
    setStatus('正在保存并读取服务器结果…');
    var readbackStarted = false;
    try {
      var fetchWithAuth = getAuthFetch();
      var baseUrl = getBaseUrl();
      if (!fetchWithAuth || baseUrl === null) {
        throw new Error('dashboard_auth_unavailable');
      }
      if (Object.keys(taskOptionsPatch).length || schemaChanged) {
        var preflightResponse = await requestConfig();
        if (!preflightResponse || !preflightResponse.ok) {
          setStatus(responseErrorLabel(preflightResponse, '核对并发配置版本'), 'error');
          return;
        }
        var preflightConfig = await preflightResponse.json();
        var preflightReflection = getReflectionConfig(preflightConfig);
        var preflightRoutes = parseRoutes(preflightReflection && preflightReflection.memory_task_routes);
        var preflightOptions = parseTaskOptions(preflightReflection);
        var preflightSchema = parseOutputSchema(preflightReflection);
        var preflightRevision = isObject(preflightReflection) && typeof preflightReflection.memory_task_revision === 'string'
          ? preflightReflection.memory_task_revision : null;
        var changedOptionConflict = preflightOptions.valid && Object.keys(taskOptionsPatch).some(function (key) {
          return preflightOptions.values[key] !== state.baselineTaskOptions[key];
        });
        var taskConflict = taskChanged && (!preflightRevision || preflightRevision !== state.baselineRevision
          || (routesChanged && (!preflightRoutes.valid || !routesEqual(preflightRoutes.routes, state.baseline)))
          || changedOptionConflict);
        var schemaConflict = schemaChanged && (!preflightSchema.valid
          || preflightSchema.revision !== state.baselineOutputSchemaRevision
          || !schemaEqual(preflightSchema.value, state.baselineOutputSchema));
        if (!preflightRoutes.valid || !preflightOptions.valid || taskConflict || schemaConflict) {
          if (preflightRoutes.valid && preflightOptions.valid && preflightRevision) {
            captureLatestServer(preflightRoutes.routes, preflightRevision,
              parseEffective(preflightReflection.memory_task_effective), preflightOptions.values,
              preflightSchema.value, preflightSchema.revision);
          }
          setStatus('服务器任务配置或第二遍输出 Schema 已有更新；本次未提交，草稿和原编辑版本已保留。请核对服务器新版后再保存。', 'error');
          return;
        }
      }
      var reflectionPatch = {};
      if (routesChanged) reflectionPatch.memory_task_routes = routes;
      Object.keys(taskOptionsPatch).forEach(function (key) { reflectionPatch[key] = taskOptionsPatch[key]; });
      if (schemaChanged) reflectionPatch.memory_l2_output_schema = schemaSnapshot;
      var requestBody = { reflection: reflectionPatch, persist: true };
      if (taskChanged) requestBody.expected_memory_task_revision = state.baselineRevision;
      if (schemaChanged) requestBody.expected_memory_l2_schema_revision = state.baselineOutputSchemaRevision;
      var response = await fetchWithAuth(baseUrl + '/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });
      if (response && response.status === 409) {
        try {
          var conflictResponse = await requestConfig();
          if (conflictResponse && conflictResponse.ok) {
            var conflictConfig = await conflictResponse.json();
            var conflictReflection = getReflectionConfig(conflictConfig);
            var latestRoutes = parseRoutes(conflictReflection && conflictReflection.memory_task_routes);
            var latestTaskOptions = parseTaskOptions(conflictReflection);
            var latestSchema = parseOutputSchema(conflictReflection);
            var latestRevision = isObject(conflictReflection) && typeof conflictReflection.memory_task_revision === 'string'
              && conflictReflection.memory_task_revision.length > 0 ? conflictReflection.memory_task_revision : null;
            var latestEffective = parseEffective(conflictReflection && conflictReflection.memory_task_effective);
            if (latestRoutes.valid && latestTaskOptions.valid && latestRevision) {
              if (latestEffective) renderEffective(latestEffective);
              captureLatestServer(latestRoutes.routes, latestRevision, latestEffective, latestTaskOptions.values,
                latestSchema.value, latestSchema.revision);
            }
          }
        } catch (_) {}
        setStatus('配置有新版本，保存未应用；草稿和原编辑版本已保留。请核对新版后再保存。', 'error');
        return;
      }
      if (!response || !response.ok) {
        setStatus(responseErrorLabel(response, '保存'), 'error');
        return;
      }
      var result = await response.json();
      if (!isObject(result) || result.ok !== true) {
        setStatus('保存未获服务器确认；草稿已保留。', 'error');
        return;
      }
      readbackStarted = true;
      var readResponse = await requestConfig();
      if (!readResponse || !readResponse.ok) {
        setStatus('保存请求已确认，但读回失败，无法确认应用状态。草稿已保留。', 'error');
        return;
      }
      var readConfig = await readResponse.json();
      var check = readbackError(readConfig, {
        routes: routesChanged ? routes : null,
        taskOptionsPatch: taskOptionsPatch,
        taskChanged: taskChanged,
        outputSchema: schemaChanged ? schemaSnapshot : null,
        schemaChanged: schemaChanged,
        baseOutputSchemaRevision: state.baselineOutputSchemaRevision,
      });
      if (check.effective) renderEffective(check.effective);
      if (check.message) {
        if (check.routes && check.revision) captureLatestServer(check.routes, check.revision, check.effective,
          check.taskOptions, check.outputSchema, check.outputSchemaRevision);
        setStatus(check.message, 'error');
        return;
      }
      state.baseline = cloneRoutes(check.routes);
      state.baselineRevision = check.revision || state.baselineRevision;
      state.baselineTaskOptions = cloneTaskOptions(check.taskOptions);
      state.baselineOutputSchema = cloneJson(check.outputSchema);
      state.baselineOutputSchemaRevision = check.outputSchemaRevision;
      state.outputSchemaContractValid = true;
      if (state.outputSchemaRevisionNode) {
        state.outputSchemaRevisionNode.textContent = 'Schema 修订（只读）：' + check.outputSchemaRevision;
      }
      state.loaded = true;
      state.routeContractValid = true;
      state.revisionValid = true;
      var liveRoutes = collectRoutes();
      var liveOptions = collectTaskOptions();
      var liveSchema = state.outputSchemaInput && !state.outputSchemaInput.disabled
        ? parseOutputSchemaInput() : { valid: true, value: check.outputSchema };
      var routeEditedDuringSave = !liveRoutes.valid || !routesEqual(liveRoutes.routes, routes);
      var optionsEditedDuringSave = !liveOptions.valid || !taskOptionsEqual(liveOptions.values, taskOptions);
      var schemaEditedDuringSave = !liveSchema.valid || !schemaEqual(liveSchema.value, schemaSnapshot || check.outputSchema);
      state.dirty = routeEditedDuringSave || optionsEditedDuringSave || schemaEditedDuringSave;
      state.latestServer = null;
      state.latestServerTaskOptions = null;
      state.latestServerOutputSchema = null;
      state.latestServerOutputSchemaRevision = null;
      if (state.rebaseButton) state.rebaseButton.hidden = true;
      if (!routeEditedDuringSave) updateInputs(check.routes);
      if (!optionsEditedDuringSave) updateTaskOptionInputs(check.taskOptions);
      if (!schemaEditedDuringSave) updateOutputSchemaInput(check.outputSchema);
      if (state.dirty) {
        persistSessionDraft();
        setStatus(verifiedSaveMessage(check, changes) + ' 保存期间的新修改仍保留为未提交草稿。', 'success');
      } else {
        clearSessionDraft();
        setStatus(verifiedSaveMessage(check, changes), 'success');
      }
    } catch (_) {
      setStatus(readbackStarted
        ? '保存请求已确认，但读回失败，无法确认应用状态。草稿已保留。'
        : '保存时连接中断；无法确认应用状态。草稿已保留。', 'error');
    } finally {
      state.saving = false;
      updateSaveButton();
      updateDraftState();
    }
  }

  async function reconcileServerVersion() {
    if (state.saving || !state.dirty) return;
    state.saving = true;
    updateSaveButton();
    updateDraftState();
    setStatus('正在读取服务器新版；本地草稿仍保留…');
    try {
      var response = await requestConfig();
      if (!response || !response.ok) {
        setStatus(responseErrorLabel(response, '读取服务器新版'), 'error');
        return;
      }
      var config = await response.json();
      var reflection = getReflectionConfig(config);
      var routes = parseRoutes(reflection && reflection.memory_task_routes);
      var taskOptions = parseTaskOptions(reflection);
      var outputSchema = parseOutputSchema(reflection);
      var revision = isObject(reflection) && typeof reflection.memory_task_revision === 'string'
        && reflection.memory_task_revision.length > 0 ? reflection.memory_task_revision : null;
      var effective = parseEffective(reflection && reflection.memory_task_effective);
      if (!routes.valid || !taskOptions.valid || !outputSchema.valid || !revision || !effective) {
        setStatus('服务器新版缺少完整路由、L2 后台选项、输出 Schema、版本或实际解析结果；草稿仍保留，暂不能合并。', 'error');
        return;
      }
      if (!state.baseline || !state.baselineRevision || !state.baselineOutputSchema
          || !state.baselineOutputSchemaRevision) {
        setStatus('无法确认草稿的原编辑版本；草稿已保留，请先重新读取并核对。', 'error');
        return;
      }
      var currentDraft = collectRoutes();
      if (!currentDraft.valid) {
        setStatus(currentDraft.error + '修正后再核对服务器新版。', 'error');
        return;
      }
      var currentTaskDraft = collectTaskOptions();
      if (!currentTaskDraft.valid) {
        setStatus(currentTaskDraft.error + '修正后再核对服务器新版。', 'error');
        return;
      }
      var currentSchemaText = outputSchemaDraftFromInput();
      var currentSchema = parseOutputSchemaInput();
      var localSchemaChanged = state.localDraft
        ? typeof state.localDraft.outputSchemaText === 'string'
        : !currentSchema.valid || outputSchemaChanged(currentSchema.value);
      var merged = {};
      var preservedChanges = 0;
      STAGES.forEach(function (stage) {
        merged[stage.id] = {};
        ROUTE_FIELDS.forEach(function (key) {
          var before = state.baseline[stage.id][key];
          var local = currentDraft.routes[stage.id][key];
          if (local === before) {
            merged[stage.id][key] = routes.routes[stage.id][key];
          } else {
            merged[stage.id][key] = local;
            preservedChanges += 1;
          }
        });
      });
      var mergedTaskOptions = {};
      TASK_OPTIONS.forEach(function (field) {
        var before = (state.baselineTaskOptions || defaultTaskOptions())[field.key];
        var local = currentTaskDraft.values[field.key];
        if (local === before) {
          mergedTaskOptions[field.key] = taskOptions.values[field.key];
        } else {
          mergedTaskOptions[field.key] = local;
          preservedChanges += 1;
        }
      });
      var mergedOutputSchema = localSchemaChanged
        ? currentSchemaText : JSON.stringify(outputSchema.value, null, 2);
      if (localSchemaChanged) preservedChanges += 1;
      renderEffective(effective);
      state.baseline = cloneRoutes(routes.routes);
      state.baselineRevision = revision;
      state.baselineTaskOptions = cloneTaskOptions(taskOptions.values);
      state.baselineOutputSchema = cloneJson(outputSchema.value);
      state.baselineOutputSchemaRevision = outputSchema.revision;
      state.outputSchemaContractValid = true;
      if (state.outputSchemaInput) {
        state.outputSchemaInput.disabled = false;
        state.outputSchemaInput.value = mergedOutputSchema;
      }
      if (state.outputSchemaRevisionNode) {
        state.outputSchemaRevisionNode.textContent = 'Schema 修订（只读）：' + outputSchema.revision;
      }
      state.routeContractValid = true;
      state.revisionValid = true;
      state.loaded = true;
      state.latestServer = null;
      state.latestServerTaskOptions = null;
      state.latestServerOutputSchema = null;
      state.latestServerOutputSchemaRevision = null;
      if (state.rebaseButton) state.rebaseButton.hidden = true;
      updateInputs(merged);
      updateTaskOptionInputs(mergedTaskOptions);
      var mergedSchemaParsed = parseOutputSchemaInput();
      state.dirty = !routesEqual(merged, routes.routes)
        || !taskOptionsEqual(mergedTaskOptions, taskOptions.values)
        || !mergedSchemaParsed.valid || outputSchemaChanged(mergedSchemaParsed.value);
      if (state.dirty) persistSessionDraft();
      else clearSessionDraft();
      if (state.dirty) {
        setStatus('已核对服务器新版：保留了 ' + preservedChanges + ' 项本地修改，并采用服务器对未编辑路由、L2 选项和输出 Schema 的更新。再次保存并应用以提交。');
      } else {
        setStatus('已核对服务器新版；路由、L2 后台选项、第二遍输出 Schema 与当前草稿一致，实际解析结果也已从服务器读回。', 'success');
      }
    } catch (_) {
      setStatus('读取或合并服务器新版失败；草稿与原编辑版本已保留。', 'error');
    } finally {
      state.saving = false;
      updateSaveButton();
      updateDraftState();
    }
  }

  function renderMemoryTaskModels(config) {
    if (!buildInterface()) return;
    if (arguments.length > 0 && isObject(config)) {
      state.readSequence += 1;
      applyConfig(config);
      return;
    }
    if (!state.loaded) void loadConfig();
  }

  window.renderMemoryTaskModels = renderMemoryTaskModels;
})(window);
