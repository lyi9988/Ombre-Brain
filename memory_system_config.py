"""Owner Memory configuration over the existing runtime overlay.

Pure schema/expressions plus a locked atomic overlay update. No Memory bodies,
model calls, executable tool policy, or additional configuration authority.
"""
from __future__ import annotations

import ast
import copy
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from contextvars import ContextVar

import yaml

SCHEMA_VERSION = "memory-config-v1"
# These are public resource bounds, not owner presets or silent clamps.
LIMITS = {
    "inject_max_cards": ("gateway", "inject_max_cards", 0, 200, 2, "最终经历条数"),
    "diffusion_inject_max_items": ("gateway", "diffusion_inject_max_items", 0, 200, 2, "扩散经历条数"),
    "inject_total_budget": ("gateway", "inject_total_budget", 0, 1000000, 1800, "总注入 token 预算"),
    "recalled_memory_budget": ("gateway", "recalled_memory_budget", 0, 1000000, 900, "召回内容 token 预算"),
    "related_memory_budget": ("gateway", "related_memory_budget", 0, 1000000, 220, "相关内容 token 预算"),
    "memory_detail_recall_max_ids": ("gateway", "memory_detail_recall_max_ids", 0, 200, 3, "详情读取条数"),
    "memory_detail_recall_budget": ("gateway", "memory_detail_recall_budget", 0, 1000000, 1200, "详情 token 预算"),
    "semantic_candidate_top_k": ("gateway", "semantic_candidate_top_k", 1, 200, 50, "向量候选数"),
    "moment_search_limit": ("gateway", "moment_search_limit", 1, 200, 50, "场景候选数"),
    "dynamic_top_k": ("gateway", "dynamic_top_k", 1, 200, 10, "本地候选数"),
    "rerank_candidate_limit": ("reranker", "candidate_limit", 1, 100, 20, "重排窗口条数"),
}
VALUE_TYPES = {"number", "boolean", "string"}
_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_FUNCTIONS = {"min", "max", "clamp", "coalesce", "exp", "abs"}
CATALOG_AXES = ("subjects", "kinds", "topics", "stabilities", "modalities", "bases",
                "entity_types", "relation_types")
EXTENDED_CATALOG_AXES = ("stabilities", "modalities", "bases", "entity_types", "relation_types", "definitions")
DEFINITION_AXES = frozenset(CATALOG_AXES)
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
REQUEST_CONFIGURATION = ContextVar("memory_request_configuration", default=None)


class RequestLimit:
    """A numeric setting frozen for one request, with normal runtime setters."""
    def __init__(self, attribute, field=None):
        self.attribute, self.field = attribute, field or attribute

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        snapshot = REQUEST_CONFIGURATION.get()
        if snapshot and snapshot.get("service") == id(instance):
            value = snapshot.get("values", {}).get("limits", {}).get(self.field)
            if value is not None:
                return value
        if self.attribute not in instance.__dict__:
            raise AttributeError(self.attribute)
        return instance.__dict__[self.attribute]

    def __set__(self, instance, value):
        instance.__dict__[self.attribute] = value


def runtime_overlay_path(config):
    configured = config.get("_runtime_config_path") or os.environ.get("OMBRE_RUNTIME_CONFIG_PATH")
    if configured:
        return str(configured)
    base = os.environ.get("OMBRE_CONFIG_PATH")
    parent = config.get("state_dir") or (str(Path(base).parent) if base else "state")
    return str(Path(parent) / "config.runtime.yaml")


class ConfigurationError(ValueError):
    def __init__(self, code, *, field="", status=400, **details):
        super().__init__(code)
        self.code, self.field, self.status, self.details = code, field, status, details

    def payload(self):
        return {"error": {"code": self.code, "field": self.field, **self.details}}


def _fail(code, field="", **details):
    raise ConfigurationError(code, field=field, **details)


def _object(value, allowed, field, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        _fail("invalid_configuration_fields", field)
    return value


def _text(value, field, *, empty=False, maximum=2000):
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        _fail("invalid_text", field)
    return value


def _id(value, field):
    if not isinstance(value, str) or not _ID.fullmatch(value) or value in _FUNCTIONS:
        _fail("invalid_identifier", field)
    return value


def _number(value, field):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except (OverflowError, TypeError):
        valid = False
    if not valid:
        _fail("finite_number_required", field)
    return value


def validate_limit(name, value):
    if name not in LIMITS:
        _fail("unknown_limit", name)
    _section, _key, low, high, _default, _label = LIMITS[name]
    if type(value) is not int or not low <= value <= high:
        _fail("limit_out_of_range", "limits." + name, minimum=low, maximum=high)
    return value


def defaults():
    return {
        "catalog": {
            "subjects": ["user", "assistant", "third_party", "object"],
            "kinds": ["experience", "state", "fact", "preference", "boundary", "intention", "interpretation", "procedure"],
            "topics": [], "attributes": [],
            "stabilities": ["episode", "phase", "stable"],
            "modalities": ["assertion", "request", "wish", "plan", "commitment", "hypothesis", "question"],
            "bases": ["direct_expression", "record", "reported", "interpretation", "hypothesis"],
            "entity_types": ["user", "assistant", "third_party", "object"],
            "relation_types": ["same_event", "supplements", "revises", "limits", "supersedes",
                               "possible_conflict", "split_from", "merged_from"],
            "definitions": [
                {"axis": "stabilities", "id": "episode", "label": "单次", "description": "限于一次事件或短时状态。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "stabilities", "id": "phase", "label": "阶段", "description": "在明确阶段内持续。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "stabilities", "id": "stable", "label": "相对稳定", "description": "跨阶段相对稳定，仍可修订。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "modalities", "id": "assertion", "label": "陈述", "description": "表达一个主张或情况。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "modalities", "id": "request", "label": "请求", "description": "向他人提出请求。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "modalities", "id": "wish", "label": "愿望", "description": "表达希望但未必计划执行。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "modalities", "id": "plan", "label": "计划", "description": "表达拟采取的行动。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "modalities", "id": "commitment", "label": "承诺", "description": "表达承担履行义务。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "modalities", "id": "hypothesis", "label": "假设", "description": "尚待确认的解释或推测。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "modalities", "id": "question", "label": "疑问", "description": "提出问题或待确认事项。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "bases", "id": "direct_expression", "label": "当事人表达", "description": "来源中的当事人直接表达。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "bases", "id": "record", "label": "实际记录", "description": "来源直接记录的事件或行为。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "bases", "id": "reported", "label": "转述", "description": "由来源转述的信息。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "bases", "id": "interpretation", "label": "理解", "description": "基于材料形成的理解。", "aliases": [], "examples": [], "parent_id": None},
                {"axis": "bases", "id": "hypothesis", "label": "假设", "description": "尚待材料支持或确认的假设。", "aliases": [], "examples": [], "parent_id": None},
            ],
        },
        "features": [{"id": "relevance", "label": "文本相关性", "value_type": "number",
                      "source": "runtime", "missing": "unknown", "expression": "",
                      "description": "归一化文本相关性；P5接运行输入，本地试算由主人提供。",
                      "minimum": 0, "maximum": 1}],
        "formulas": [{"id": "final", "label": "最终用途评分（待 P5 消费）",
                      "expression": "relevance", "weights": {}}],
        "time_profiles": [],
    }


def merge_config(base, overlay):
    result = copy.deepcopy(base or {})
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_overlay(path):
    target = Path(path)
    if not target.exists():
        return {}
    result = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(result, dict):
        _fail("invalid_runtime_overlay", status=409)
    return result


def values_from_config(config):
    values = defaults()
    configured = config.get("memory_system", {})
    if isinstance(configured, dict):
        for key in values:
            if key in configured:
                if key == "catalog" and isinstance(configured[key], dict):
                    values[key] = merge_config(values[key], configured[key])
                else:
                    values[key] = copy.deepcopy(configured[key])
    limits = {}
    for name, (section, key, _lo, _hi, fallback, _label) in LIMITS.items():
        source = config.get(section, {})
        limits[name] = source.get(key, fallback) if isinstance(source, dict) else fallback
    # Preserve the legacy dynamic defaults when no explicit channel size exists.
    gateway = config.get("gateway", {})
    if isinstance(gateway, dict):
        size = limits["dynamic_top_k"]
        if type(size) is int:
            if "semantic_candidate_top_k" not in gateway:
                limits["semantic_candidate_top_k"] = max(50, size)
            if "moment_search_limit" not in gateway:
                limits["moment_search_limit"] = min(200, max(50, size * 2))
    values["limits"] = limits
    return values


def revision(values):
    raw = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return "mc-" + hashlib.sha256(raw.encode()).hexdigest()


def _expression(expression, allowed, field):
    _text(expression, field)
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError):
        _fail("invalid_expression", field)
    nodes = list(ast.walk(tree))
    if len(nodes) > 160:
        _fail("expression_too_complex", field)
    types = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
             ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
             ast.Call, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
             ast.BoolOp, ast.And, ast.Or, ast.Not, ast.IfExp)
    for node in nodes:
        if not isinstance(node, types):
            _fail("unsupported_expression", field)
        if isinstance(node, ast.Name) and node.id not in allowed | _FUNCTIONS:
            _fail("unknown_expression_variable", field, variable=node.id)
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                or node.func.id not in _FUNCTIONS or node.keywords or len(node.args) > 16):
            _fail("unsupported_expression_call", field)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (str, int, float, bool, type(None))):
                _fail("unsupported_expression_literal", field)
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                _number(node.value, field)
    return tree


def validate_values(values):
    sections = {"limits", "catalog", "features", "formulas", "time_profiles"}
    _object(values, sections, "values", sections)
    limits = _object(values["limits"], LIMITS, "limits", LIMITS)
    for key, value in limits.items():
        validate_limit(key, value)
    catalog = _object(values["catalog"], set(CATALOG_AXES) | {"attributes", "definitions"}, "catalog",
                      set(CATALOG_AXES) | {"attributes", "definitions"})
    for name in CATALOG_AXES:
        rows = catalog[name]
        if not isinstance(rows, list) or len(rows) > 512:
            _fail("invalid_catalog_list", "catalog." + name)
        seen = set()
        for item in rows:
            if name in EXTENDED_CATALOG_AXES:
                _id(item, "catalog." + name)
            else:
                _text(item, "catalog." + name, maximum=120)
            if item in seen:
                _fail("duplicate_catalog_value", "catalog." + name)
            seen.add(item)
    for key in ("attributes",):
        if not isinstance(catalog[key], list) or len(catalog[key]) > 256:
            _fail("invalid_attribute_list", "catalog." + key)
        seen = set()
        for item in catalog[key]:
            _object(item, {"id", "label", "value_type", "description", "options", "scope"}, "attributes", {"id", "value_type"})
            ident = _id(item["id"], "attributes.id")
            if ident in seen:
                _fail("duplicate_identifier", "attributes.id")
            seen.add(ident)
            if not isinstance(item["value_type"], str) or item["value_type"] not in VALUE_TYPES | {"integer", "enum", "entity_ref"}:
                _fail("invalid_value_type", "attributes.value_type")
            if "scope" in item and (not isinstance(item["scope"], str) or item["scope"] not in {"memory", "scene", "statement", "any"}):
                _fail("invalid_attribute_scope", "attributes.scope")
            if "options" in item:
                if not isinstance(item["options"], list) or any(not isinstance(v, str) or not v for v in item["options"]):
                    _fail("invalid_enum_options", "attributes.options")
                if len(set(item["options"])) != len(item["options"]):
                    _fail("duplicate_enum_option", "attributes.options")
            if item["value_type"] == "enum" and not item.get("options"):
                _fail("enum_options_required", "attributes.options")
            for name in ("label", "description"):
                if name in item:
                    _text(item[name], "attributes." + name, empty=True)
    definitions = catalog["definitions"]
    if not isinstance(definitions, list) or len(definitions) > 2048:
        _fail("invalid_catalog_definitions", "catalog.definitions")
    definition_ids = {}
    parents = {}
    for item in definitions:
        _object(item, {"id", "axis", "label", "description", "aliases", "examples", "parent_id"},
                "catalog.definitions", {"id", "axis", "label", "description", "aliases", "examples", "parent_id"})
        axis = item["axis"]
        if axis not in DEFINITION_AXES:
            _fail("invalid_definition_axis", "catalog.definitions.axis")
        ident = _id(item["id"], "catalog.definitions.id")
        key = (axis, ident)
        if key in definition_ids:
            _fail("duplicate_definition", "catalog.definitions.id", axis=axis, id=ident)
        if ident not in catalog[axis]:
            _fail("definition_value_missing", "catalog.definitions.id", axis=axis, id=ident)
        _text(item["label"], "catalog.definitions.label", empty=True)
        _text(item["description"], "catalog.definitions.description", empty=True)
        for field in ("aliases", "examples"):
            rows = item[field]
            if not isinstance(rows, list) or len(rows) > 128:
                _fail("invalid_definition_list", "catalog.definitions." + field)
            for value in rows:
                _text(value, "catalog.definitions." + field, empty=True, maximum=500)
        parent = item["parent_id"]
        if parent is not None:
            parent = _id(parent, "catalog.definitions.parent_id")
            parents[key] = (axis, parent)
        definition_ids[key] = True
    for key, parent in parents.items():
        if parent[1] not in catalog[parent[0]]:
            _fail("definition_parent_missing", "catalog.definitions.parent_id", axis=key[0], id=parent[1])
    visited, active = set(), set()
    def visit_definition(key):
        if key in active:
            _fail("definition_parent_cycle", "catalog.definitions.parent_id", axis=key[0], id=key[1])
        if key in visited:
            return
        active.add(key)
        parent = parents.get(key)
        if parent and parent in definition_ids:
            visit_definition(parent)
        active.remove(key)
        visited.add(key)
    for key in definition_ids:
        visit_definition(key)
    features = values["features"]
    if not isinstance(features, list) or len(features) > 256:
        _fail("invalid_features", "features")
    names, dependencies = set(), {}
    for item in features:
        _object(item, {"id", "label", "value_type", "source", "missing", "expression", "description", "minimum", "maximum"}, "features",
                {"id", "value_type", "source", "missing"})
        ident = _id(item["id"], "features.id")
        if ident in names:
            _fail("duplicate_identifier", "features.id")
        names.add(ident)
        if item["value_type"] not in VALUE_TYPES or item["source"] not in {"runtime", "annotation", "computed"}:
            _fail("invalid_feature_type", "features")
        if item["missing"] != "unknown":
            _fail("unknown_missing_policy", "features.missing")
        if "label" in item:
            _text(item["label"], "features.label", empty=True)
        if "description" in item:
            _text(item["description"], "features.description", empty=True)
        for bound in ("minimum", "maximum"):
            if item.get(bound) is not None:
                if item["value_type"] != "number":
                    _fail("non_numeric_feature_bound", "features." + bound)
                _number(item[bound], "features." + bound)
        if item.get("minimum") is not None and item.get("maximum") is not None and item["minimum"] > item["maximum"]:
            _fail("invalid_feature_range", "features." + ident)
        if item["source"] != "computed" and item.get("expression"):
            _fail("noncomputed_expression", "features.expression")
    for item in features:
        if item["source"] == "computed":
            tree = _expression(item.get("expression"), names, "features." + item["id"])
            dependencies[item["id"]] = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in names}
    visited, active = set(), set()
    def visit(name):
        if name in active:
            _fail("feature_dependency_cycle", "features." + name)
        if name in visited:
            return
        active.add(name)
        for dep in dependencies.get(name, ()):
            visit(dep)
        active.remove(name)
        visited.add(name)
    for name in names:
        visit(name)
    formulas = values["formulas"]
    if not isinstance(formulas, list) or len(formulas) > 64:
        _fail("invalid_formulas", "formulas")
    formula_names = set()
    for item in formulas:
        _object(item, {"id", "label", "expression", "weights"}, "formulas", {"id", "expression", "weights"})
        ident = _id(item["id"], "formulas.id")
        if ident in formula_names:
            _fail("duplicate_identifier", "formulas.id")
        formula_names.add(ident)
        if "label" in item:
            _text(item["label"], "formulas.label", empty=True)
        weights = item["weights"]
        if not isinstance(weights, dict) or len(weights) > 256:
            _fail("invalid_weights", "formulas.weights")
        for name, value in weights.items():
            _id(name, "formulas.weights")
            if name in names:
                _fail("weight_shadows_feature", "formulas.weights", variable=name)
            _number(value, "formulas.weights." + name)
        _expression(item["expression"], names | set(weights), "formulas." + ident)
    profiles = values["time_profiles"]
    if not isinstance(profiles, list) or len(profiles) > 256:
        _fail("invalid_time_profiles", "time_profiles")
    seen = set()
    for item in profiles:
        _object(item, {"id", "label", "selector", "function", "half_life_days", "review_after_hours", "weight"},
                "time_profiles", {"id", "selector", "function"})
        ident = _id(item["id"], "time_profiles.id")
        if ident in seen:
            _fail("duplicate_identifier", "time_profiles.id")
        seen.add(ident)
        if "label" in item:
            _text(item["label"], "time_profiles.label", empty=True)
        selector = _object(item["selector"], {"kind", "topic", "stability", "query_purpose"}, "time_profiles.selector")
        for name, value in selector.items():
            _text(value, "time_profiles.selector." + name, maximum=120)
        if item["function"] not in {"none", "exponential", "target_time"}:
            _fail("unknown_time_function", "time_profiles.function")
        for name in ("half_life_days", "review_after_hours", "weight"):
            value = item.get(name)
            if value is not None and (_number(value, "time_profiles." + name) < 0
                    or (name == "half_life_days" and value == 0)):
                _fail("invalid_time_parameter", "time_profiles." + name)
        if item["function"] == "exponential" and item.get("half_life_days") is None:
            _fail("half_life_required", "time_profiles.half_life_days")
    return copy.deepcopy(values)


def _typed(value, kind, field):
    if value is None:
        return None
    if kind == "number":
        return _number(value, field)
    if kind == "boolean" and type(value) is bool:
        return value
    if kind == "string" and isinstance(value, str) and len(value) <= 2000:
        return value
    _fail("input_type_mismatch", field)


def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Call):
        name = node.func.id
        if name == "coalesce":
            for arg in node.args:
                value = _eval(arg, env)
                if value is not None:
                    return value
            return None
        args = [_eval(arg, env) for arg in node.args]
        if any(value is None for value in args):
            return None
        args = [_number(value, "expression") for value in args]
        if name in {"min", "max"} and args:
            return (min if name == "min" else max)(args)
        if name == "clamp" and len(args) == 3 and args[1] <= args[2]:
            return max(args[1], min(args[2], args[0]))
        if name == "abs" and len(args) == 1:
            return abs(args[0])
        if name == "exp" and len(args) == 1:
            return math.exp(args[0])
        _fail("invalid_function_arguments", "expression")
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        if a is None or b is None:
            return None
        a, b = _number(a, "expression"), _number(b, "expression")
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div): return a / b
        if isinstance(node.op, ast.Mod): return a % b
        if isinstance(node.op, ast.Pow) and abs(b) <= 8: return a ** b
        _fail("unsupported_power", "expression")
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, env)
        if value is None: return None
        if isinstance(node.op, ast.Not):
            return not _typed(value, "boolean", "expression")
        value = _number(value, "expression")
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Compare):
        a = _eval(node.left, env)
        result = True
        for op, other in zip(node.ops, node.comparators):
            b = _eval(other, env)
            if a is None or b is None:
                result = None
            else:
                if type(a) is not type(b) and not (type(a) in (int, float) and type(b) in (int, float)):
                    _fail("comparison_type_mismatch", "expression")
                match = ((a == b) if isinstance(op, ast.Eq) else (a != b) if isinstance(op, ast.NotEq)
                         else (a < b) if isinstance(op, ast.Lt) else (a <= b) if isinstance(op, ast.LtE)
                         else (a > b) if isinstance(op, ast.Gt) else (a >= b))
                if not match: return False
            a = b
        return result
    if isinstance(node, ast.BoolOp):
        args = [_typed(_eval(arg, env), "boolean", "expression") for arg in node.values]
        if isinstance(node.op, ast.And):
            return False if False in args else None if None in args else True
        return True if True in args else None if None in args else False
    if isinstance(node, ast.IfExp):
        condition = _typed(_eval(node.test, env), "boolean", "expression")
        return None if condition is None else _eval(node.body if condition else node.orelse, env)
    _fail("unsupported_expression", "expression")


def preview(values, inputs):
    values = validate_values(values)
    if not isinstance(inputs, dict):
        _fail("invalid_preview_inputs", "inputs")
    definitions = {item["id"]: item for item in values["features"]}
    if set(inputs) - set(definitions):
        _fail("unknown_preview_input", "inputs")
    env = {}
    def feature(name):
        if name in env: return env[name]
        item = definitions[name]
        if item["source"] == "computed":
            if name in inputs:
                _fail("computed_input_override", "inputs." + name)
            tree = _expression(item["expression"], set(definitions), name)
            deps = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in definitions}
            value = _eval(tree, {dep: feature(dep) for dep in deps})
        else:
            value = inputs.get(name)
        env[name] = _typed(value, item["value_type"], name)
        if env[name] is not None and item["value_type"] == "number":
            if ((item.get("minimum") is not None and env[name] < item["minimum"])
                    or (item.get("maximum") is not None and env[name] > item["maximum"])):
                _fail("feature_out_of_range", name)
        return env[name]
    try:
        for name in definitions: feature(name)
        results = {}
        for item in values["formulas"]:
            tree = _expression(item["expression"], set(definitions) | set(item["weights"]), item["id"])
            value = _eval(tree, {**env, **item["weights"]})
            if value is not None: _number(value, item["id"])
            results[item["id"]] = {"value": value, "state": "unknown" if value is None else "computed"}
    except (ZeroDivisionError, OverflowError, ArithmeticError, TypeError, RecursionError):
        _fail("expression_evaluation_failed", "expression")
    return {"features": env, "formulas": results, "provider_requests": 0, "saved": False}


def configuration(config, *, last_request=None):
    values = values_from_config(config)
    rev = revision(values)
    try:
        validate_values(values)
        validation = {"valid": True, "errors": []}
    except ConfigurationError as exc:
        validation = {"valid": False, "errors": [exc.payload()["error"]]}
    return {
        "schema_version": SCHEMA_VERSION, "revision": rev, "values": values,
        "bindings": {"limits": {name: {"section": row[0], "key": row[1], "minimum": row[2],
                      "maximum": row[3], "default": row[4], "label": row[5]} for name, row in LIMITS.items()}},
        "application": {
            "limits": {"state": "active", "revision": rev},
            "catalog": {"state": "awaiting_consumer", "consumer": "P2"},
            "features": {"state": "awaiting_consumer", "consumer": "P5"},
            "formulas": {"state": "awaiting_consumer", "consumer": "P5"},
            "time_profiles": {"state": "awaiting_consumer", "consumer": "P3/P5"},
            "last_request": last_request,
        }, "validation": validation,
    }


@contextmanager
def overlay_lock(path):
    key = str(Path(path).resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    with lock:
        target = Path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(str(target) + ".lock", "a+b") as handle:
            if os.name == "nt":
                import msvcrt
                handle.seek(0, 2)
                if handle.tell() == 0:
                    handle.write(b"0"); handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield target
            finally:
                if os.name == "nt":
                    handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)


def update_overlay(path, updater):
    if not str(path or "").strip():
        _fail("runtime_config_unavailable", status=503)
    with overlay_lock(path) as target:
        current = read_overlay(target)
        updated = updater(copy.deepcopy(current))
        if not isinstance(updated, dict):
            _fail("invalid_runtime_overlay")
        if updated != current:
            fd, temporary = tempfile.mkstemp(prefix=".memory-config-", dir=str(target.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    yaml.safe_dump(updated, handle, allow_unicode=True, sort_keys=False)
                    handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary): os.unlink(temporary)
        return updated


def save_configuration(path, base_config, payload):
    _object(payload, {"expected_revision", "request_id", "values"}, "request",
            {"expected_revision", "request_id", "values"})
    expected = _text(payload["expected_revision"], "expected_revision", maximum=100)
    if not re.fullmatch(r"mc-[0-9a-f]{64}", expected):
        _fail("invalid_configuration_revision", "expected_revision")
    request_id = _text(payload["request_id"], "request_id", maximum=128)
    submitted_values = copy.deepcopy(payload["values"])
    if not isinstance(submitted_values, dict):
        _fail("invalid_configuration_fields", "values")
    fingerprint = revision(submitted_values)
    outcome = {}
    def change(overlay):
        effective = merge_config(base_config, overlay)
        current = configuration(effective)
        values = copy.deepcopy(submitted_values)
        catalog = values.get("catalog")
        if not isinstance(catalog, dict):
            _fail("invalid_configuration_fields", "catalog")
        supplied_extended = set(catalog) & set(EXTENDED_CATALOG_AXES)
        if supplied_extended and supplied_extended != set(EXTENDED_CATALOG_AXES):
            _fail("incomplete_extended_catalog", "catalog", required=list(EXTENDED_CATALOG_AXES))
        effective_values = values_from_config(effective)
        for name in EXTENDED_CATALOG_AXES:
            if name not in catalog:
                catalog[name] = copy.deepcopy(effective_values["catalog"][name])
        values = validate_values(values)
        metadata = overlay.setdefault("memory_system_meta", {})
        receipts = metadata.setdefault("receipts", {})
        prior = receipts.get(request_id)
        if prior:
            if prior.get("fingerprint") != fingerprint:
                _fail("request_id_conflict", status=409)
            # The caller supplies the original values, verified by digest.
            # Keep only a small durable receipt, not a copy of the entire
            # catalogue and UI projection for every save.
            original = copy.deepcopy(effective)
            for name, value in values["limits"].items():
                section, key, *_ = LIMITS[name]
                original.setdefault(section, {})[key] = value
            original["memory_system"] = {key: value for key, value in values.items() if key != "limits"}
            outcome.update(configuration(original))
            outcome.update(saved_revision=prior["revision"], request_id=request_id, replayed=True)
            return overlay
        if current["revision"] != expected:
            _fail("configuration_revision_conflict", status=409, current_revision=current["revision"])
        for name, value in values["limits"].items():
            section, key, *_ = LIMITS[name]
            overlay.setdefault(section, {})[key] = value
        overlay["memory_system"] = {key: copy.deepcopy(value) for key, value in values.items() if key != "limits"}
        saved = configuration(merge_config(base_config, overlay))
        saved.update(saved_revision=saved["revision"], request_id=request_id, replayed=False)
        receipts[request_id] = {"fingerprint": fingerprint, "revision": saved["revision"]}
        metadata["saved_revision"] = saved["revision"]
        outcome.update(saved)
        return overlay
    overlay = update_overlay(path, change)
    return outcome, merge_config(base_config, overlay)
