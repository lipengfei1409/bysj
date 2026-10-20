from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DPA4Config:
    type_map: tuple[str, ...] = ("O", "H")
    rcut: float = 6.0
    channels: int = 16
    n_radial: int = 6
    lmax: int = 1
    mmax: int | None = 1
    kmax: int = 1
    n_blocks: int = 1
    n_focus: int = 1
    focus_dim: int = 0
    n_heads: int = 1
    mixing_layers: int = 2
    ffn_blocks: int = 1
    ffn_hidden: int = 0
    use_env_seed: bool = True
    env_exp: tuple[int, int] = (7, 5)
    l_schedule: tuple[int, ...] = ()
    m_schedule: tuple[int, ...] = ()
    extra_node_l: int = 0
    radial_so2_mode: str = "none"
    atten_f_mix: bool = False
    sandwich_norm: tuple[bool, bool, bool, bool] = (False, True, True, False)
    optional: dict[str, Any] = field(default_factory=dict)
    radial_so2_rank: int = 0
    so2_norm: bool = False
    ffn_so3_grid: bool = False
    grid_branch: tuple[int, int, int] = (0, 0, 0)
    message_node_so3: bool = False
    focus_compete: bool = True
    focus_norm: bool = True
    charge_spin_enabled: bool = False
    default_charge_spin: tuple[float, float] | None = None
    precision: str = "float32"
    so3_readout: str = "mlp"
    readout_layers: int = 1
    mlp_bias: bool = False
    s2_activation: tuple[bool, bool] = (False, False)

    def __post_init__(self):
        if self.rcut <= 0 or min(self.channels, self.n_radial, self.n_focus) <= 0:
            raise ValueError("rcut, channels, n_radial and n_focus must be positive")
        if min(self.n_blocks, self.lmax, self.n_heads, self.ffn_blocks, self.extra_node_l, self.focus_dim, self.ffn_hidden, self.kmax) < 0 or (self.mmax is not None and self.mmax < 0):
            raise ValueError("model dimensions must be nonnegative")
        if self.mixing_layers < 0:
            raise ValueError("mixing_layers must be nonnegative")
        if self.radial_so2_rank < 0:
            raise ValueError("radial_so2_rank must be nonnegative")
        if len(self.env_exp) != 2 or any(value < 0 for value in self.env_exp):
            raise ValueError("env_exp must contain two nonnegative exponents")
        if len(self.sandwich_norm) != 4:
            raise ValueError("sandwich_norm requires four branch flags")
        if len(self.grid_branch) != 3 or any(value < 0 for value in self.grid_branch):
            raise ValueError("grid_branch must contain three nonnegative values")
        if self.so3_readout not in ("none", "glu", "mlp") or self.readout_layers < 1:
            raise ValueError("invalid SO(3) readout configuration")
        if len(self.s2_activation) != 2:
            raise ValueError("s2_activation must contain SO(2) and FFN flags")
        if self.default_charge_spin is not None:
            if len(self.default_charge_spin) != 2 or not all(
                math.isfinite(value) for value in self.default_charge_spin
            ):
                raise ValueError("default_charge_spin must contain two finite values")
        for schedule in (self.l_schedule, self.m_schedule):
            if schedule and len(schedule) != self.n_blocks:
                raise ValueError("schedule length must equal n_blocks")
        if any(a < b for a, b in zip(self.degrees, self.degrees[1:])):
            raise ValueError("l_schedule must be non-increasing")
        for l, m in zip(self.degrees, self.orders):
            if not 0 <= m <= l:
                raise ValueError("each m must satisfy 0 <= m <= l")
        attention_width = self.focus_channels * (self.n_focus if self.atten_f_mix else 1)
        if self.n_heads and attention_width % self.n_heads:
            raise ValueError("attention heads must divide the attention channel width")

    @property
    def degrees(self):
        return self.l_schedule or (self.lmax,) * self.n_blocks

    @property
    def orders(self):
        return self.m_schedule or tuple(l if self.mmax is None else min(self.mmax, l) for l in self.degrees)

    @property
    def initial_l(self):
        return (self.degrees[0] if self.degrees else self.lmax) + self.extra_node_l

    @property
    def focus_channels(self):
        return self.focus_dim or self.channels

    @classmethod
    def from_model_params(cls, params: dict) -> "DPA4Config":
        d = params["descriptor"]
        schedule = d.get("l_schedule")
        degrees = tuple(schedule or ())
        nblocks = len(degrees) if schedule is not None else int(d.get("n_blocks", 1))
        grid_branch = d.get("grid_branch")
        if grid_branch is None:
            grid_branch = (0, 0, 0)
        elif isinstance(grid_branch, int):
            grid_branch = (grid_branch,) * 3
        else:
            grid_branch = tuple(int(value) for value in grid_branch)
        default_charge_spin = d.get("default_chg_spin")
        raw_env_exp = d.get("env_exp")
        if raw_env_exp is None:
            raw_env_exp = (7, 5)
        env_exp = (0, raw_env_exp) if isinstance(raw_env_exp, int) else tuple(int(value) for value in raw_env_exp)
        sandwich_norm = d.get("sandwich_norm")
        if sandwich_norm is None:
            sandwich_norm = (False, True, True, False)
        else:
            sandwich_norm = tuple(sandwich_norm)
        edge_norm = d.get("edge_norm", True)
        if isinstance(edge_norm, (list, tuple)):
            if len(edge_norm) != 3:
                raise ValueError("edge_norm must be a bool or three branch flags")
            focus_norm = bool(edge_norm[2])
        else:
            focus_norm = bool(edge_norm)
        raw_s2_activation = d.get("s2_activation")
        if raw_s2_activation is None:
            s2_activation = (False, False)
        elif isinstance(raw_s2_activation, bool):
            s2_activation = (raw_s2_activation, raw_s2_activation)
        else:
            s2_activation = tuple(bool(value) for value in raw_s2_activation)
        return cls(
            type_map=tuple(params["type_map"]), rcut=float(d["rcut"]),
            channels=int(d["channels"]), n_radial=int(d["n_radial"]),
            lmax=int(degrees[0] if degrees else d.get("lmax", 1)), mmax=int(d["mmax"]) if d.get("mmax") is not None else None,
            kmax=int(d.get("kmax", 1)),
            n_blocks=nblocks, n_focus=int(d.get("n_focus", 1)), focus_dim=int(d.get("focus_dim", 0)),
            n_heads=int(d.get("n_atten_head", 0)), mixing_layers=int(d.get("mixing_layers", 2)),
            ffn_blocks=int(d.get("ffn_blocks", 1)), ffn_hidden=int(d.get("ffn_neurons", 0)),
            use_env_seed=bool(d.get("use_env_seed", False)), env_exp=env_exp,
            l_schedule=degrees, m_schedule=tuple(d.get("m_schedule") or ()),
            extra_node_l=int(d.get("extra_node_l", 0)), radial_so2_mode=d.get("radial_so2_mode", "none"),
            atten_f_mix=bool(d.get("atten_f_mix", False)), optional=dict(d),
            sandwich_norm=sandwich_norm,
            radial_so2_rank=int(d.get("radial_so2_rank", 0)),
            so2_norm=bool(d.get("so2_norm", False)),
            ffn_so3_grid=bool(d.get("ffn_so3_grid", False)),
            grid_branch=grid_branch,
            message_node_so3=bool(d.get("message_node_so3", False)),
            focus_compete=bool(d.get("focus_compete", True)),
            focus_norm=focus_norm,
            charge_spin_enabled=bool(d.get("add_chg_spin_ebd", False)),
            default_charge_spin=(
                None
                if default_charge_spin is None
                else tuple(float(value) for value in default_charge_spin)
            ),
            precision=str(d.get("precision", "float32")),
            so3_readout=str(d.get("so3_readout", "mlp")).lower(),
            readout_layers=int(d.get("readout_layers", 1)),
            mlp_bias=bool(d.get("mlp_bias", False)),
            s2_activation=s2_activation,
        )
