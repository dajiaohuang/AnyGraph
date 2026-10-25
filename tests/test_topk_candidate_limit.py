import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


def _load_main():
    names = ["main", "params", "model", "data_handler", "setproctitle", "Utils", "Utils.TimeLogger"]
    previous = {name: sys.modules.get(name) for name in names}
    params = types.ModuleType("params")
    params.args = SimpleNamespace(topk=20)
    model = types.ModuleType("model")
    for name in ("Expert", "Feat_Projector", "Adj_Projector", "AnyGraph"):
        setattr(model, name, object)
    data_handler = types.ModuleType("data_handler")
    data_handler.MultiDataHandler = object
    data_handler.DataHandler = object
    setproctitle = types.ModuleType("setproctitle")
    logger = types.ModuleType("Utils.TimeLogger")
    logger.log = lambda *args, **kwargs: None
    logger.saveDefault = False
    utils = types.ModuleType("Utils")
    utils.__path__ = []
    utils.TimeLogger = logger
    sys.modules.update({
        "params": params,
        "model": model,
        "data_handler": data_handler,
        "setproctitle": setproctitle,
        "Utils": utils,
        "Utils.TimeLogger": logger,
    })
    try:
        spec = importlib.util.spec_from_file_location("anygraph_main_topk_test", MAIN_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, previous
    except Exception:
        _restore_modules(previous)
        raise


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_topk_is_capped_to_available_candidates():
    module, previous = _load_main()
    try:
        scores = torch.tensor([[0.9, 0.5, 0.1]])
        with pytest.raises(RuntimeError):
            torch.topk(scores, 20)

        values, indices = module._topk_predictions(scores, 20)
        assert indices.tolist() == [[0, 1, 2]]
        assert torch.allclose(values, scores)
    finally:
        _restore_modules(previous)


def test_topk_rejects_nonpositive_limits():
    module, previous = _load_main()
    try:
        with pytest.raises(ValueError, match="topk must be positive"):
            module._topk_predictions(torch.ones((1, 3)), 0)
    finally:
        _restore_modules(previous)
