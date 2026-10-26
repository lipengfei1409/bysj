"""Coarse DPA4 execution manifest for operator replacement planning."""

from __future__ import annotations

from dataclasses import dataclass

from .checkpoint import CheckpointSpec


@dataclass(frozen=True)
class Stage:
    name: str
    owner: str
    note: str


def build_stage_plan(spec: CheckpointSpec) -> tuple[Stage, ...]:
    """List top-level stages; this is not yet a tile-level executable DAG."""
    stages = [
        Stage("neighbor_graph", "deepmd", "periodic neighbors and edge indices"),
        Stage("edge_geometry", "src", "edge vectors, distances, periodic shifts"),
        Stage("radial_basis", "src", "learned radial basis and C3 cutoff"),
        Stage("type_embedding", "deepmd", "element embeddings"),
    ]
    if spec.use_env_seed:
        stages.append(Stage("environment_seed", "deepmd", "environment embedding"))
    stages.append(Stage("geometric_initial_embedding", "deepmd", "SO(3) edge/node seeds"))
    for index in range(spec.n_blocks):
        stages.append(
            Stage(f"interaction_block_{index}", "deepmd", "SO(2) convolution and FFN")
        )
    stages.extend(
        [
            Stage("output_ffn", "deepmd", "SO(3) readout to invariant channels"),
            Stage("fitting_energy", "deepmd", "per-atom energy fitting"),
            Stage("energy_reduce", "deepmd", "per-frame energy sum"),
            Stage("force_virial", "src", "edge-gradient scatter and virial"),
        ]
    )
    return tuple(stages)
