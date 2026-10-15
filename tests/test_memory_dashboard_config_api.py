"""Execute the actual route function without booting Brain or its providers."""
import ast
import asyncio
import copy
import json
import os
from pathlib import Path
import re

import pytest

from config_activation import activation_receipt, config_save_result
from memory_system_config import configuration, read_overlay, update_overlay


class Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return copy.deepcopy(self.body)


def route_for(tmp_path):
    source = Path(__file__).resolve().parents[1] / "server.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "api_config_update")
    node.decorator_list = []
    config = {"state_dir": str(tmp_path), "_runtime_config_path": str(tmp_path / "runtime.yaml")}
    pushes = []
    async def hot(payload, **kwargs):
        pushes.append(copy.deepcopy(payload))
        return activation_receipt(state="confirmed")
    namespace = {"config": config, "__file__": str(source), "os": os, "re": re,
                 "_require_dashboard_auth": lambda request: None,
                 "_hot_update_gateway_config": hot,
                 "config_save_result": config_save_result, "activation_receipt": activation_receipt}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), namespace)
    return namespace["api_config_update"], config, pushes


def test_dashboard_persist_and_hot_push_keep_large_limits_and_other_overlay_sections(tmp_path):
    route, config, pushes = route_for(tmp_path)
    path = tmp_path / "runtime.yaml"
    update_overlay(path, lambda data: {"untouched": {"keep": 19}})
    revision = configuration(config)["revision"]
    values = {"inject_max_cards": 7, "diffusion_inject_max_items": 5, "inject_total_budget": 14000,
              "memory_detail_recall_max_ids": 10, "memory_detail_recall_budget": 8000,
              "dynamic_top_k": 20, "semantic_candidate_top_k": 5}
    response = asyncio.run(route(Request({"gateway": values, "persist": True,
                                         "expected_memory_revision": revision})))
    assert response.status_code == 200, response.body
    assert json.loads(response.body)["gateway_activation"]["state"] == "confirmed"
    saved = read_overlay(path)
    assert saved["untouched"] == {"keep": 19}
    for key, value in values.items():
        assert saved["gateway"][key] == value
        assert pushes[-1]["gateway"][key] == value
    before = path.read_bytes()
    stale = asyncio.run(route(Request({"gateway": {"inject_max_cards": 6}, "persist": True,
                                      "expected_memory_revision": revision})))
    assert stale.status_code == 409
    assert path.read_bytes() == before


def test_missing_revision_and_bad_limits_do_not_mutate_runtime_or_persist(tmp_path):
    route, config, pushes = route_for(tmp_path)
    before = copy.deepcopy(config)
    for body in ({"gateway": {"inject_max_cards": 7}, "persist": True},
                 {"gateway": {"inject_max_cards": 201}, "persist": True,
                  "expected_memory_revision": configuration(config)["revision"]}):
        response = asyncio.run(route(Request(body)))
        assert response.status_code == 400
        assert config == before and pushes == []
    assert not (tmp_path / "runtime.yaml").exists()
