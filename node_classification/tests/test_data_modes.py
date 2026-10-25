import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import coo_matrix


NODE_DIR = Path(__file__).resolve().parents[1]


def _load_data_handler(args):
    names = [
        "data_handler", "params", "model", "Utils", "Utils.TimeLogger",
        "torch_geometric", "torch_geometric.transforms",
    ]
    previous = {name: sys.modules.get(name) for name in names}
    node_dir_str = str(NODE_DIR)
    sys.path.insert(0, node_dir_str)

    params = types.ModuleType("params")
    params.args = args
    logger = types.ModuleType("Utils.TimeLogger")
    logger.log = lambda *args, **kwargs: None
    utils = types.ModuleType("Utils")
    utils.__path__ = []
    utils.TimeLogger = logger
    model = types.ModuleType("model")
    model.Feat_Projector = object
    model.Adj_Projector = object
    model.TopoEncoder = object
    torch_geometric = types.ModuleType("torch_geometric")
    torch_geometric.__path__ = []
    transforms = types.ModuleType("torch_geometric.transforms")
    torch_geometric.transforms = transforms

    sys.modules.update({
        "params": params,
        "Utils": utils,
        "Utils.TimeLogger": logger,
        "model": model,
        "torch_geometric": torch_geometric,
        "torch_geometric.transforms": transforms,
    })
    try:
        module = importlib.import_module("data_handler")
    except Exception:
        _restore_modules(previous, node_dir_str)
        raise
    return module, previous, node_dir_str


def _restore_modules(previous, node_dir_str):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    sys.path.remove(node_dir_str)


def _make_handler(module, args):
    full_train = coo_matrix((np.ones(3), ([0, 1, 2], [0, 1, 0])), shape=(3, 2))
    fewshot = coo_matrix((np.ones(2), ([0, 1], [0, 1])), shape=(3, 2))
    validation = coo_matrix((np.ones(1), ([2], [1])), shape=(3, 2))
    test = coo_matrix((np.ones(1), ([0], [1])), shape=(3, 2))
    files = {
        "train": full_train,
        "fewshot": fewshot,
        "validation": validation,
        "test": test,
    }
    handler = module.DataHandler.__new__(module.DataHandler)
    handler.data_name = "synthetic"
    handler.feat_file = None
    handler.trnfile = "train"
    handler.fewshotfile = "fewshot"
    handler.valfile = "validation"
    handler.tstfile = "test"
    loaded_files = []

    def load_file(filename):
        loaded_files.append(filename)
        return files[filename]

    handler.load_one_file = load_file
    adj_calls = []

    def make_adj(mat, unidirectional_for_asym=False):
        source = next(name for name, candidate in files.items() if candidate is mat)
        adj = (source, unidirectional_for_asym)
        adj_calls.append(adj)
        return adj

    handler.make_torch_adj = make_adj
    handler.make_projectors = lambda: None
    return handler, files, adj_calls, loaded_files


def _args(trn_mode, tst_mode):
    return SimpleNamespace(
        trn_mode=trn_mode,
        tst_mode=tst_mode,
        latdim=4,
        batch=2,
        tst_batch=2,
        proj_trn_steps=1,
    )


def test_fewshot_validation_uses_fewshot_labels_and_adjacency():
    args = _args("fewshot", "val")
    module, previous, node_dir_str = _load_data_handler(args)
    try:
        handler, files, adj_calls, loaded_files = _make_handler(module, args)
        handler.load_data()

        assert handler.trn_mat is files["fewshot"]
        assert handler.trn_loader.dataset.ancs.tolist() == files["fewshot"].row.tolist()
        assert handler.tst_loader.dataset.nodes.tolist() == files["validation"].row.tolist()
        assert handler.tst_loader.dataset.labels.tolist() == files["validation"].col.tolist()
        assert handler.tst_input_adj == ("fewshot", False)
        assert handler.trn_input_adj == handler.tst_input_adj
        assert ("fewshot", True) in adj_calls
        assert loaded_files == ["test", "train", "validation", "fewshot"]
    finally:
        _restore_modules(previous, node_dir_str)


def test_train_all_test_mode_preserves_existing_matrices():
    args = _args("train-all", "tst")
    module, previous, node_dir_str = _load_data_handler(args)
    try:
        handler, files, _, loaded_files = _make_handler(module, args)
        handler.load_data()

        assert handler.trn_mat is files["train"]
        assert handler.trn_loader.dataset.ancs.tolist() == files["train"].row.tolist()
        assert handler.tst_loader.dataset.nodes.tolist() == files["test"].row.tolist()
        assert handler.tst_input_adj == ("train", False)
        assert handler.trn_input_adj == handler.tst_input_adj
        assert loaded_files == ["test", "train"]
    finally:
        _restore_modules(previous, node_dir_str)
