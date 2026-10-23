"""Read DPA4 checkpoint metadata without freezing or importing DeePMD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class CheckpointSpec:
    path: Path
    model_type: str
    type_map: tuple[str, ...]
    rcut: float
    channels: int
    n_radial: int
    n_blocks: int
    lmax: int
    mmax: int
    basis_type: str
    env_exp: tuple[int, int]
    use_env_seed: bool
    radial_frequencies_shape: tuple[int, ...]
    model_params: dict


def load_checkpoint_spec(path: str | Path, head: str | None = None) -> CheckpointSpec:
    """Extract shape-driving parameters from a trusted DeePMD ``.pt`` file.

    ``torch.load`` can deserialize Python objects; only open checkpoints from
    a source you trust. ``weights_only=True`` narrows the accepted object set.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict) or "_extra_state" not in state:
        raise ValueError("checkpoint has no DeePMD model metadata")
    model_params = state["_extra_state"]["model_params"]
    if "model_dict" in model_params:
        branches = model_params["model_dict"]
        if head is None:
            raise ValueError(f"multitask checkpoint requires head; choices: {list(branches)}")
        if head not in branches:
            raise ValueError(f"unknown head {head!r}; choices: {list(branches)}")
        model_params = branches[head]
    descriptor = model_params["descriptor"]
    from .modeling.config import DPA4Config
    config = DPA4Config.from_model_params(model_params)
    model_type = str(model_params.get("type", descriptor.get("type", ""))).lower()
    if model_type not in ("dpa4", "sezm", "se_zm"):
        raise ValueError(f"checkpoint is not DPA4/SeZM: {model_type!r}")
    freq_keys = [key for key in state if key.endswith(".radial_basis.adam_freqs")]
    if head is not None:
        freq_keys = [key for key in freq_keys if f".{head}." in key]
    if len(freq_keys) != 1:
        raise ValueError(f"expected one radial frequency tensor, found {freq_keys}")
    frequencies_shape = tuple(state[freq_keys[0]].shape)
    return CheckpointSpec(
        path=path,
        model_type=model_type,
        type_map=tuple(model_params["type_map"]),
        rcut=float(descriptor["rcut"]),
        channels=int(descriptor["channels"]),
        n_radial=int(descriptor["n_radial"]),
        n_blocks=config.n_blocks,
        lmax=config.lmax,
        mmax=config.orders[0] if config.orders else (config.lmax if config.mmax is None else config.mmax),
        basis_type=str(descriptor.get("basis_type", "bessel")),
        env_exp=config.env_exp,
        use_env_seed=bool(descriptor.get("use_env_seed", False)),
        radial_frequencies_shape=frequencies_shape,
        model_params=model_params,
    )
