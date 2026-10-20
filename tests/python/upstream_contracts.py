"""Execute upstream forward bodies in isolation, NOT a full DeePMD import.

DeePMD's compiled libraries/e3nn are absent locally. These contract probes use
the checked-out forward code, with module parameters supplied by each test.
The separate opt-in integration test is responsible for real model loading.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


DEEP = Path(__file__).resolve().parents[3] / "deepmd-kit"


def module_with_upstream_forward(relative_path, name, *, strict=False):
    tree = ast.parse((DEEP / relative_path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    forward.decorator_list=[]
    namespace={"torch":torch,"F":F,"env":SimpleNamespace(DP_DTYPE_PROMOTION_STRICT=strict)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[forward],type_ignores=[])),str(DEEP/relative_path),"exec"),namespace)
    qualified=relative_path.removesuffix(".py").replace("/", ".")
    return type(name,(nn.Module,),{"__module__":qualified,"forward":namespace["forward"]})()
