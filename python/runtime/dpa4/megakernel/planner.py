"""Checkpoint-shaped DPA4 task DAGs for later HIP scheduling.

The planner describes work and synchronization epochs; it does not execute
kernels or claim that the registered operators are implemented.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math


@dataclass(frozen=True)
class TaskTemplate:
    name: str
    operator: str
    domain: str
    epoch: int
    work_items: int
    tile_size: int
    depends_on: tuple[str, ...] = ()
    variant: str | None = None

    def __post_init__(self):
        if not self.name or not self.operator:
            raise ValueError("task name and operator must be nonempty")
        if self.domain not in ("edge", "node", "frame"):
            raise ValueError("task domain must be edge, node or frame")
        if self.epoch < 0 or self.work_items < 0 or self.tile_size <= 0:
            raise ValueError("invalid task epoch or tiling contract")

    @property
    def grid_size(self) -> int:
        if self.work_items == 0:
            return 0
        return math.ceil(self.work_items / self.tile_size)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["grid_size"] = self.grid_size
        return result


@dataclass(frozen=True)
class DPA4TaskGraph:
    """Validated DAG of operator templates with explicit launch epochs."""

    tasks: tuple[TaskTemplate, ...]
    version: int = 1

    def __post_init__(self):
        if self.version != 1:
            raise ValueError("unsupported task-graph version")
        by_name = {task.name: task for task in self.tasks}
        if len(by_name) != len(self.tasks):
            raise ValueError("task names must be unique")
        for task in self.tasks:
            for dependency in task.depends_on:
                if dependency not in by_name:
                    raise ValueError(f"unknown dependency {dependency!r} for {task.name!r}")
                if by_name[dependency].epoch > task.epoch:
                    raise ValueError("task dependencies cannot point to a later epoch")
        self.topological_order()

    def topological_order(self) -> tuple[TaskTemplate, ...]:
        by_name = {task.name: task for task in self.tasks}
        pending = {task.name: set(task.depends_on) for task in self.tasks}
        ordered = []
        while pending:
            ready = sorted(
                (name for name, deps in pending.items() if not deps),
                key=lambda name: (by_name[name].epoch, name),
            )
            if not ready:
                raise ValueError("task graph contains a dependency cycle")
            for name in ready:
                ordered.append(by_name[name])
                del pending[name]
            for deps in pending.values():
                deps.difference_update(ready)
        return tuple(ordered)

    def epochs(self) -> tuple[tuple[TaskTemplate, ...], ...]:
        if not self.tasks:
            return ()
        groups = []
        for epoch in range(max(task.epoch for task in self.tasks) + 1):
            groups.append(tuple(task for task in self.tasks if task.epoch == epoch))
        return tuple(groups)

    def to_dict(self) -> dict:
        ordered = self.topological_order()
        task_ids = {task.name: index for index, task in enumerate(ordered)}
        consumer_counts = {task.name: 0 for task in ordered}
        for task in ordered:
            for dependency in task.depends_on:
                consumer_counts[dependency] += 1
        records = []
        for task in ordered:
            record = task.to_dict()
            record["task_id"] = task_ids[task.name]
            record["dependency_ids"] = [task_ids[name] for name in task.depends_on]
            record["completion_consumers"] = consumer_counts[task.name]
            records.append(record)
        return {"version": self.version, "tasks": records}

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def build_dpa4_task_graph(
    config,
    *,
    n_atoms: int,
    edge_capacity: int,
    n_frames: int,
    edge_tile: int = 256,
    node_tile: int = 128,
    frame_tile: int = 1,
    fuse_edge_features: bool = False,
) -> DPA4TaskGraph:
    """Build an auditable epoch DAG for one fixed model and shape bucket.

    ``edge_capacity`` is the allocation bound, not the dynamic valid-edge
    count. Device kernels must still apply the graph mask and valid length.
    Cross-tile destination reductions and frame reductions advance the epoch;
    no implicit grid-wide barrier is represented here.
    """
    if min(n_atoms, edge_capacity) < 0 or n_frames <= 0:
        raise ValueError("invalid atom, edge-capacity or frame count")
    if min(edge_tile, node_tile, frame_tile) <= 0:
        raise ValueError("tile sizes must be positive")

    tasks: list[TaskTemplate] = []

    def add(name, operator, domain, epoch, deps=(), variant=None):
        work_items, tile_size = {
            "edge": (edge_capacity, edge_tile),
            "node": (n_atoms, node_tile),
            "frame": (n_frames, frame_tile),
        }[domain]
        task = TaskTemplate(name, operator, domain, epoch, work_items, tile_size, tuple(deps), variant)
        tasks.append(task)
        return name

    if fuse_edge_features:
        add("edge_features", "edge_features", "edge", 0)
        edge_task = radial_task = "edge_features"
    else:
        add("edge_geometry", "edge_geometry", "edge", 0)
        add("radial_basis", "radial_basis", "edge", 0, ("edge_geometry",))
        edge_task, radial_task = "edge_geometry", "radial_basis"
    add("type_embedding", "type_embedding", "node", 0)
    type_task = "type_embedding"
    if config.charge_spin_enabled:
        add("charge_spin_embedding", "charge_spin_embedding", "frame", 0)
        add("condition_type", "optional_conditioning", "node", 0,
            ("type_embedding", "charge_spin_embedding"))
        type_task = "condition_type"
    add("pair_embedding", "pair_embedding", "edge", 0, (type_task, edge_task))
    add("degree_reduce", "degree_reduce", "node", 1, (radial_task, edge_task))
    add("radial_mlp", "radial_mlp", "edge", 1, (radial_task,))
    add("radial_fuse", "radial_fuse", "edge", 1, ("radial_mlp", "pair_embedding"))
    add("wigner_rotation", "wigner_rotation", "edge", 1, (edge_task,))
    scalar_task = type_task
    if config.use_env_seed:
        add("environment_seed", "environment_seed", "node", 1,
            (type_task, radial_task, "degree_reduce"))
        add("film", "film", "node", 1, (type_task, "environment_seed"))
        scalar_task = "film"
    seed_deps = [scalar_task, "radial_mlp", "degree_reduce", "wigner_rotation"]
    add("geometric_seed", "geometric_seed", "node", 1, tuple(seed_deps))

    node_state = "geometric_seed"
    epoch = 2
    for block in range(config.n_blocks):
        prefix = f"block_{block}"
        block_deps = [node_state, "wigner_rotation", "radial_fuse"]
        if config.sandwich_norm[0]:
            add(f"{prefix}_pre_norm", "equivariant_norm", "node", epoch, tuple(block_deps))
            block_deps = [f"{prefix}_pre_norm"]
        add(f"{prefix}_focus_mix", "focus_mix", "node", epoch, tuple(block_deps))
        add(f"{prefix}_edge_message", "gather_rotate", "edge", epoch,
            (f"{prefix}_focus_mix", "wigner_rotation"))
        modulation = "radial_so2" if config.radial_so2_mode != "none" else "radial_modulation"
        add(
            f"{prefix}_radial_modulation",
            modulation,
            "edge",
            epoch,
            (f"{prefix}_edge_message", "radial_fuse"),
            variant=f"{config.radial_so2_mode}:rank={config.radial_so2_rank}",
        )
        previous = f"{prefix}_radial_modulation"
        for layer in range(config.mixing_layers):
            name = f"{prefix}_so2_{layer}"
            add(name, "so2_linear", "edge", epoch, (previous,))
            previous = name
            if layer < config.mixing_layers - 1:
                activation_name = f"{prefix}_so2_activation_{layer}"
                add(activation_name, "gated_activation", "edge", epoch, (previous,))
                previous = activation_name
        if config.focus_compete and config.n_focus > 1:
            focus_name = f"{prefix}_focus_compete"
            add(focus_name, "focus_compete", "edge", epoch, (previous,))
            previous = focus_name
        add(f"{prefix}_inverse_rotate", "inverse_rotate", "edge", epoch,
            (previous, "wigner_rotation"))
        if config.n_heads:
            add(f"{prefix}_attention_logits", "attention_logits", "edge", epoch,
                (f"{prefix}_focus_mix", "radial_fuse", edge_task))
            attention_epoch = epoch + 1
            add(f"{prefix}_attention_softmax", "attention_softmax", "edge", attention_epoch,
                (f"{prefix}_attention_logits", "degree_reduce"))
            reduce_epoch = attention_epoch + 1
            reduce_deps = (f"{prefix}_inverse_rotate", f"{prefix}_attention_softmax")
        else:
            reduce_epoch = epoch + 1
            reduce_deps = (f"{prefix}_inverse_rotate", edge_task)
        add(f"{prefix}_message_reduce", "message_reduce", "node", reduce_epoch, reduce_deps)
        message_state = f"{prefix}_message_reduce"
        if config.message_node_so3:
            project = f"{prefix}_message_grid_project"
            activation = f"{prefix}_message_grid_activation"
            integrate = f"{prefix}_message_grid_integrate"
            add(project, "grid_project", "node", reduce_epoch,
                (message_state, node_state), variant=f"so3:kmax={config.kmax}")
            add(activation, "grid_activation", "node", reduce_epoch, (project,),
                variant=f"message_node_branches={config.grid_branch[1]};kmax={config.kmax}")
            add(integrate, "grid_integrate", "node", reduce_epoch, (activation,),
                variant=f"so3:kmax={config.kmax}")
            message_state = integrate
        update_deps = [message_state, node_state]
        if config.sandwich_norm[1]:
            post_norm = f"{prefix}_post_norm"
            add(post_norm, "equivariant_norm", "node", reduce_epoch, (message_state,))
            update_deps = [post_norm, node_state]
        add(f"{prefix}_residual", "residual", "node", reduce_epoch, tuple(update_deps))
        node_state = f"{prefix}_residual"
        for layer in range(config.ffn_blocks):
            name = f"{prefix}_ffn_{layer}"
            residual_input = node_state
            ffn_input = residual_input
            if config.sandwich_norm[2]:
                ffn_norm = f"{name}_pre_norm"
                add(ffn_norm, "equivariant_norm", "node", reduce_epoch, (ffn_input,))
                ffn_input = ffn_norm
            if config.ffn_so3_grid:
                project = f"{name}_grid_project"
                activation = f"{name}_grid_activation"
                update = f"{name}_grid_integrate"
                add(project, "grid_project", "node", reduce_epoch, (ffn_input,),
                    variant=f"so3:kmax={config.kmax}")
                add(activation, "grid_activation", "node", reduce_epoch, (project,),
                    variant=f"ffn_branches={config.grid_branch[2]};kmax={config.kmax}")
                add(update, "grid_integrate", "node", reduce_epoch, (activation,),
                    variant=f"so3:kmax={config.kmax}")
            else:
                update = name
                add(update, "equivariant_ffn", "node", reduce_epoch, (ffn_input,))
            if config.sandwich_norm[3]:
                post_norm = f"{name}_post_norm"
                add(post_norm, "equivariant_norm", "node", reduce_epoch, (update,))
                update = post_norm
            residual = f"{name}_residual"
            add(residual, "residual", "node", reduce_epoch, (residual_input, update))
            node_state = residual
        epoch = reduce_epoch + 1

    add("scalar_readout", "scalar_readout", "node", epoch, (node_state,),
        variant=f"{config.so3_readout}:layers={config.readout_layers}")
    add("fitting_energy", "fitting_energy", "node", epoch, ("scalar_readout",))
    add("energy_reduce", "energy_reduce", "frame", epoch + 1, ("fitting_energy",))
    add("edge_gradient", "edge_gradient", "edge", epoch + 2,
        ("energy_reduce", edge_task))
    add("force_virial", "force_virial", "edge", epoch + 3,
        ("edge_gradient", edge_task))
    return DPA4TaskGraph(tuple(tasks))
