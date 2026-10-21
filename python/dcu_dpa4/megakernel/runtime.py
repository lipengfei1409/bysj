"""Host-side lowering for the opt-in HIP persistent task runtime."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import secrets


RUNTIME_PLAN_VERSION = 1
OPERATOR_IDS = {"edge_features": 1}


@dataclass(frozen=True)
class RuntimeTask:
    task_id: int
    name: str
    operator: str
    operator_id: int
    epoch: int
    work_items: int
    tile_size: int
    depends_on: tuple[int, ...]

    @property
    def tile_count(self) -> int:
        if self.work_items == 0:
            return 0
        return (self.work_items + self.tile_size - 1) // self.tile_size

    def to_dict(self) -> dict:
        result = asdict(self)
        result["tile_count"] = self.tile_count
        return result


@dataclass(frozen=True)
class PersistentTaskPlan:
    """Versioned, device-oriented subset of a static task DAG."""

    tasks: tuple[RuntimeTask, ...]
    generation: int
    queue_capacity: int
    version: int = RUNTIME_PLAN_VERSION
    max_queue_tiles: int = 1 << 20

    def __post_init__(self):
        if self.version != RUNTIME_PLAN_VERSION:
            raise ValueError(f"unsupported runtime-plan version {self.version}")
        if not 0 < self.generation < 2**31:
            raise ValueError("runtime generation must be in [1,2^31)")
        if self.queue_capacity < 0:
            raise ValueError("queue capacity must be nonnegative")
        if self.max_queue_tiles <= 0 or self.queue_capacity > self.max_queue_tiles:
            raise ValueError("runtime queue capacity exceeds its configured limit")
        if sum(task.tile_count for task in self.tasks) != self.queue_capacity:
            raise ValueError("queue capacity must equal the total number of tile tickets")
        ids = {task.task_id for task in self.tasks}
        if ids != set(range(len(self.tasks))):
            raise ValueError("runtime task ids must be contiguous from zero")
        for task in self.tasks:
            if task.operator_id != OPERATOR_IDS.get(task.operator):
                raise ValueError(f"operator id mismatch for task {task.name!r}")
            if task.epoch < 0 or task.work_items < 0 or task.tile_size <= 0:
                raise ValueError(f"invalid runtime task dimensions for {task.name!r}")
            if len(set(task.depends_on)) != len(task.depends_on):
                raise ValueError(f"runtime task {task.name!r} repeats a dependency")
            if any(dep not in ids or dep >= task.task_id for dep in task.depends_on):
                raise ValueError(f"runtime task {task.name!r} has an invalid dependency id")
        if len(self.tasks) > 1:
            if self.tasks[0].depends_on:
                raise ValueError("edge-feature runtime plans require one root task")
            for index, task in enumerate(self.tasks[1:], start=1):
                if task.depends_on != (index - 1,):
                    raise NotImplementedError(
                        "the shared edge-feature payload supports only a linear task chain"
                    )

    @classmethod
    def from_task_graph(
        cls,
        graph,
        *,
        target_tasks,
        max_queue_tiles: int = 1 << 20,
        generation: int | None = None,
    ) -> "PersistentTaskPlan":
        """Lower selected tasks and their transitive prerequisites.

        The initial device dispatcher supports only ``edge_features``. Unsupported
        operators are rejected before any tensor is submitted to the HIP runtime.
        """
        if max_queue_tiles <= 0:
            raise ValueError("max_queue_tiles must be positive")
        if isinstance(target_tasks, str):
            targets = (target_tasks,)
        else:
            targets = tuple(target_tasks)
        if not targets:
            raise ValueError("at least one target task is required")
        if len(set(targets)) != len(targets):
            raise ValueError("target task names must be unique")

        by_name = {task.name: task for task in graph.tasks}
        missing = [name for name in targets if name not in by_name]
        if missing:
            raise ValueError(f"unknown target task names: {missing}")

        closure = set()
        pending = list(targets)
        while pending:
            name = pending.pop()
            if name in closure:
                continue
            closure.add(name)
            pending.extend(by_name[name].depends_on)

        ordered = [task for task in graph.topological_order() if task.name in closure]
        wrong_domains = [task.name for task in ordered if task.domain != "edge"]
        if wrong_domains:
            raise ValueError(f"edge-feature runtime tasks must use the edge domain: {wrong_domains}")
        unsupported = sorted({task.operator for task in ordered if task.operator not in OPERATOR_IDS})
        if unsupported:
            raise NotImplementedError(
                "persistent HIP runtime does not support selected operators: "
                + ", ".join(unsupported)
            )

        task_ids = {task.name: index for index, task in enumerate(ordered)}
        tasks = tuple(
            RuntimeTask(
                task_id=task_ids[task.name],
                name=task.name,
                operator=task.operator,
                operator_id=OPERATOR_IDS[task.operator],
                epoch=task.epoch,
                work_items=task.work_items,
                tile_size=task.tile_size,
                depends_on=tuple(task_ids[name] for name in task.depends_on if name in closure),
            )
            for task in ordered
        )
        queue_capacity = sum(task.tile_count for task in tasks)
        if queue_capacity > max_queue_tiles:
            raise OverflowError(
                f"persistent plan needs {queue_capacity} tile tickets; "
                f"limit is {max_queue_tiles}"
            )
        generation = secrets.randbelow(2**31 - 1) + 1 if generation is None else int(generation)
        return cls(tasks, generation, queue_capacity, max_queue_tiles=max_queue_tiles)

    def to_native(self) -> dict:
        """Return primitive host arrays consumed by the native binding."""
        dependency_offsets = [0]
        dependency_ids = []
        for task in self.tasks:
            dependency_ids.extend(task.depends_on)
            dependency_offsets.append(len(dependency_ids))
        return {
            "version": self.version,
            "generation": self.generation,
            "queue_capacity": self.queue_capacity,
            "max_queue_tiles": self.max_queue_tiles,
            "task_names": [task.name for task in self.tasks],
            "opcodes": [task.operator_id for task in self.tasks],
            "epochs": [task.epoch for task in self.tasks],
            "work_items": [task.work_items for task in self.tasks],
            "tile_sizes": [task.tile_size for task in self.tasks],
            "dependency_offsets": dependency_offsets,
            "dependency_ids": dependency_ids,
        }

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "generation": self.generation,
            "queue_capacity": self.queue_capacity,
            "max_queue_tiles": self.max_queue_tiles,
            "tasks": [task.to_dict() for task in self.tasks],
        }

_RUNTIME_ERRORS = {
    1: "stale runtime generation",
    2: "unsupported runtime opcode",
    3: "runtime queue overflow",
    4: "invalid runtime task descriptor",
    5: "runtime graph stalled with unfinished tasks",
}


def check_runtime_status(status) -> None:
    """Synchronize the status scalar and raise for an asynchronous scheduler error."""
    if not hasattr(status, "item") or not hasattr(status, "shape") or tuple(status.shape) != ():
        raise TypeError("runtime status must be a scalar tensor")
    code = int(status.item())
    if code:
        raise RuntimeError(_RUNTIME_ERRORS.get(code, f"unknown runtime error code {code}"))


def execute_persistent_edge_features(
    task_graph,
    *,
    target_tasks,
    positions,
    edge_index,
    shifts,
    cells,
    edge_frame,
    mask,
    frequencies,
    cutoff,
    basis_exponent=7,
    edge_exponent=5,
    eps=1e-7,
    tile_limit=1 << 20,
    generation=None,
):
    """Run a selected edge-feature dependency closure and return outputs/status.

    This executes only runtime-plan tasks lowered to the ``edge_features``
    opcode. Other DPA4 stages remain outside the persistent worker path.
    """
    plan = PersistentTaskPlan.from_task_graph(
        task_graph,
        target_tasks=target_tasks,
        max_queue_tiles=tile_limit,
        generation=generation,
    )
    edge_count = edge_index.shape[1]
    if any(task.work_items != edge_count for task in plan.tasks):
        raise ValueError("selected edge-feature tasks must cover the supplied edge capacity")
    from dcu_dpa4.ops import edge_features_persistent

    return edge_features_persistent(
        positions, edge_index, shifts, cells, edge_frame, mask, frequencies,
        cutoff, basis_exponent, edge_exponent, eps, runtime_plan=plan,
    )
