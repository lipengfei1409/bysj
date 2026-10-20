"""Small forward adapters preserving the checked-out DeePMD module contracts."""
import sys
import math

import torch

from .. import ops


def _linear_last_axis(x, weight, bias, backend):
    width=x.shape[-1]
    y=ops.linear(x.reshape(math.prod(x.shape[:-1]),width),weight,bias,backend=backend)
    return y.reshape(*x.shape[:-1],weight.shape[1])


def torch_linear(module,x,backend):
    # torch nn.Linear stores [C,K]; native ABI and DeePMD use [K,C].
    if backend=="reference" and torch.is_autocast_enabled(x.device.type):
        return torch.nn.functional.linear(x,module.weight,module.bias)
    return _linear_last_axis(x,module.weight.T,module.bias,backend)


def channel_linear(module,x,backend):
    return _linear_last_axis(x,module.weight,module.bias if module.use_bias else None,backend)


def mlp_linear(module,x,backend):
    env=sys.modules.get("deepmd.pt.utils.env")
    strict=getattr(env,"DP_DTYPE_PROMOTION_STRICT",False)
    original_dtype=x.dtype
    xx=x if strict else x.to(module.prec)
    if backend=="reference" and torch.is_autocast_enabled(xx.device.type):
        y=torch.nn.functional.linear(xx,module.matrix.T,module.bias)
    else:
        y=_linear_last_axis(xx,module.matrix,module.bias,backend)
    y=module.activate(y)
    if module.idt is not None:
        y=y*module.idt
    if module.resnet:
        if xx.shape[-1]==y.shape[-1]:
            y=y+xx
        elif 2*xx.shape[-1]==y.shape[-1]:
            y=y+torch.cat((xx,xx),dim=-1)
    return y if strict else y.to(original_dtype)


def rmsnorm(module,x,backend):
    original_dtype=x.dtype
    # Like upstream RMSNorm, normalization is outside autocast in module dtype.
    with torch.autocast(device_type=x.device.type,enabled=False):
        value=x.to(module.dtype)
        y=ops.rmsnorm(value.reshape(-1,value.shape[-1]),module.adam_scale,module.eps,backend=backend)
        return y.reshape(x.shape).to(original_dtype)


def radial(module,r,backend):
    if backend=="hip":
        if module.basis_family!="bessel":
            raise NotImplementedError("HIP radial adapter only supports Bessel; Gaussian stays unsupported")
        return ops.radial(r,module.adam_freqs,module.rcut,module.exponent,0,backend=backend)[0]
    # Preserve upstream non-persistent constants after dtype conversions;
    # recomputing pi/gaussian width in a new dtype subtly changes derivatives.
    if module.basis_family=="bessel":
        raw=module.adam_freqs*torch.sinc((r*module.adam_freqs)/module.pi_tensor)
    else:
        dr=r-module.adam_freqs
        raw=torch.exp(dr*dr*module.gaussian_coeff)
    return raw if module.envelope is None else raw*module.envelope(r)


# Exact classes only: do not accidentally replace specialized equivariant or
# per-focus linear/norm subclasses with plain scalar operations.
ADAPTERS={
    "torch.nn.modules.linear.Linear":("linear",torch_linear),
    "deepmd.pt.model.network.mlp.MLPLayer":("linear",mlp_linear),
    "deepmd.pt.model.descriptor.sezm_nn.so3.ChannelLinear":("linear",channel_linear),
    "deepmd.pt.model.descriptor.sezm_nn.norm.RMSNorm":("rmsnorm",rmsnorm),
    "deepmd.pt.model.descriptor.sezm_nn.radial.RadialBasis":("radial",radial),
}
