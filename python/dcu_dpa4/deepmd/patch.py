"""Reversible instance-local operator installation; no DeePMD global patches."""
from dataclasses import dataclass
from functools import wraps
import inspect
from types import MethodType

import torch
from torch import nn

from .adapters import ADAPTERS


@dataclass
class _Patch:
    path:str
    module:nn.Module
    operator:str
    original:object
    had_override:bool
    override:object
    replacement:object=None
    calls:int=0


class DeepMDOperatorHandle:
    """Own installed forwards; restore before exporting/deepcopying a model.

    Installation/restoration must not race with model execution. The caller
    owns the model's thread lifecycle. Context entry does not reinstall a
    previously restored handle.
    """
    def __init__(self,model,backend,patches):
        self.model,self.backend,self._patches=model,backend,patches
        self.active=True

    def report(self):
        return [{"module":p.path or "<root>","operator":p.operator,"backend":self.backend,
            "class":f"{type(p.module).__module__}.{type(p.module).__name__}",
            "calls":p.calls,"installed":self.active} for p in self._patches]

    def restore(self):
        if not self.active:
            return
        # Preflight avoids overwriting another owner's later changes.
        for p in self._patches:
            if p.module.__dict__.get("forward") is not p.replacement:
                raise RuntimeError(f"forward changed externally at {p.path}; refusing to overwrite")
        for p in reversed(self._patches):
            if p.had_override:
                p.module.forward=p.override
            else:
                del p.module.forward
            del p.module._dcu_dpa4_operator_owner
        self.active=False

    def __enter__(self):
        if not self.active:
            raise RuntimeError("operator handle has already been restored")
        return self

    def __exit__(self,*exc):
        self.restore()


def _check_eager(model):
    for path,module in model.named_modules():
        if isinstance(module,torch.jit.ScriptModule) or type(module).__module__.startswith("torch._dynamo"):
            raise TypeError("operator adapters require an eager model, not TorchScript/torch.compile")
        if getattr(module,"_dcu_dpa4_operator_owner",None) is not None:
            raise RuntimeError(f"operators already installed at {path or '<root>'}")
        flags=("use_triton_infer","use_cute_infer","use_cutile_infer","cuda_infer_level","triton_infer_level","_compile_infer","_act_infer")
        if any(bool(getattr(module,flag,False)) for flag in flags):
            raise RuntimeError(f"fused inference enabled at {path}; configure_eager before constructing the model")
        should_compile=getattr(module,"should_use_compile",None)
        if callable(should_compile) and should_compile():
            raise RuntimeError("DeePMD compile path enabled; configure_eager before loading the model")


def install_operators(model,*,backend="reference",operators=("radial","rmsnorm","linear")):
    """Install primitive adapters in an existing DeePMD PyTorch eager model.

    Use ``potential.get_model()`` to obtain the model from DeepPot. Parameters,
    buffers, state_dict names and module hooks remain owned by DeePMD. Only
    exact supported classes are matched. No numerical placeholder is used.
    """
    if not isinstance(model,nn.Module):
        raise TypeError("expected a PyTorch nn.Module from DeepPot.get_model()")
    if backend not in ("reference","hip"):
        raise ValueError("operator backend must be reference or hip")
    if isinstance(operators,str):
        raise TypeError("operators must be a sequence, not a string")
    selected=set(operators)
    unknown=selected-{"linear","rmsnorm","radial"}
    if unknown:
        raise ValueError(f"unsupported adapter operators: {sorted(unknown)}")
    _check_eager(model)
    if backend=="hip":
        from ..native import _extension
        _extension()  # require a real extension before changing any forward
    candidates=[]
    for path,module in model.named_modules():
        adapter=ADAPTERS.get(f"{type(module).__module__}.{type(module).__name__}")
        if adapter and adapter[0] in selected:
            original=module.forward
            signature=inspect.signature(original)
            if len(signature.parameters)!=1:
                raise RuntimeError(f"upstream forward signature changed at {path}: {signature}")
            if backend=="hip":
                tensors=list(module.parameters(recurse=False))+list(module.buffers(recurse=False))
                if any(t.is_floating_point() and (t.dtype!=torch.float32 or t.device.type!="cuda") for t in tensors):
                    raise RuntimeError(f"HIP adapter at {path} requires DCU FP32 parameters/buffers; no fallback")
                if adapter[0]=="radial" and module.basis_family!="bessel":
                    raise NotImplementedError("HIP radial adapter only supports Bessel")
            candidates.append((_Patch(path,module,adapter[0],original,"forward" in module.__dict__,module.__dict__.get("forward")),adapter[1],signature))
    if not candidates:
        raise ValueError("no supported operators matched the eager model")
    handle=DeepMDOperatorHandle(model,backend,[p for p,_,_ in candidates])
    for p,adapter,signature in candidates:
        def build_wrapper(p,adapter,signature):
            @wraps(getattr(p.original,"__func__",p.original))
            def forward(module,*args,**kwargs):
                if backend=="hip" and (model.training or module.training):
                    raise RuntimeError("HIP adapters support eval inference, not force-loss training; call model.eval()")
                bound=signature.bind(*args,**kwargs)
                x=next(iter(bound.arguments.values()))
                # Preserve nn.Linear autocast output semantics (BF16/FP16) on
                # reference; native FP32 kernel cannot silently change these.
                if backend=="hip" and torch.is_autocast_enabled(x.device.type):
                    raise RuntimeError("HIP adapters require autocast disabled for FP32 inference")
                y=adapter(module,x,backend)
                p.calls+=1
                return y
            # MethodType removes the leading owner parameter when inspecting
            # a bound method, including when the original was an override.
            forward.__signature__=signature.replace(parameters=[inspect.Parameter("_owner",inspect.Parameter.POSITIONAL_ONLY),*signature.parameters.values()])
            return MethodType(forward,p.module)
        p.replacement=build_wrapper(p,adapter,signature)
        p.module.forward=p.replacement
        p.module._dcu_dpa4_operator_owner=handle
    return handle
