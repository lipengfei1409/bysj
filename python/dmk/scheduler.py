"""Assign instructions to CUs -- the reference scheduler's job, on our units.

Mirrors `Megakernels/megakernels/scheduler.py`:

* a :class:`DagNode` wraps one instruction and the nodes it depends on;
* :func:`assign_dag_to_cus` walks the DAG in readiness order and hands each
  instruction to whichever CU is free soonest (a heap keyed by finish time), so
  a long GEMM tile does not sit behind a queue of short elementwise work when
  another CU is idle;
* :func:`round_robin_assign` is the trivial alternative, kept because it is the
  right answer for a flow that is already balanced and costs nothing to reason
  about.

Differences from the reference, deliberately:

* "SM" is called a CU here (a DCU workgroup, one block);
* cost is per instruction (:meth:`Instruction.cost`), not measured kernel time;
* there is no ``ScheduleBuilder`` subclass per model -- `shard.py` builds the
  nodes for a flat flow, and a model that needs a real DAG can call
  :func:`assign_dag_to_cus` directly.

Nothing here is device-side: the output is one list of instructions per CU, and
`instructions.InstructionStream` turns that into the int32 array.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .instructions import Instruction, NoOp


@dataclass(eq=False)
class DagNode:
    """One instruction plus the instructions it must follow."""

    instruction: Instruction
    dependencies: list["DagNode"] = field(default_factory=list)

    children: set["DagNode"] = field(default_factory=set)
    remaining: set["DagNode"] = field(default_factory=set)
    start_time: float = float("inf")
    end_time: float = float("inf")

    def register_with_parents(self) -> None:
        for dependency in self.dependencies:
            dependency.children.add(self)

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"DagNode({self.instruction!r})"


def linear_chain(instructions: list[Instruction]) -> tuple[list[DagNode], DagNode]:
    """A DAG where each instruction depends on the previous one.

    That is the shape a flat operator flow has: the ops of one model run in
    order.  A model with independent branches builds its own nodes.
    """
    if not instructions:
        raise ValueError("a schedule needs at least one instruction")
    nodes: list[DagNode] = []
    previous: DagNode | None = None
    for instruction in instructions:
        node = DagNode(instruction=instruction)
        if previous is not None:
            node.dependencies = [previous]
        nodes.append(node)
        previous = node
    assert previous is not None
    return nodes, previous


def assign_dag_to_cus(nodes: list[DagNode], cus: int) -> list[list[Instruction]]:
    """Hand each node to the CU that frees up first, respecting dependencies.

    Ready nodes are taken longest-first, so a heavy instruction is not starved by
    a stream of cheap ones -- the same ordering the reference uses.
    """
    if cus < 1:
        raise ValueError(f"cus must be positive, got {cus}")
    for node in nodes:
        node.register_with_parents()
        node.remaining = set(node.dependencies)

    queues: list[list[Instruction]] = [[] for _ in range(cus)]
    cu_heap = [(0.0, index) for index in range(cus)]
    heapq.heapify(cu_heap)

    index_of = {node: index for index, node in enumerate(nodes)}
    ready: list[tuple[float, int]] = [
        (-node.instruction.cost(), index_of[node])
        for node in nodes
        if not node.remaining
    ]
    heapq.heapify(ready)

    scheduled = 0
    while ready:
        _, index = heapq.heappop(ready)
        node = nodes[index]
        finish, cu = heapq.heappop(cu_heap)
        node.start_time = finish
        node.end_time = finish + node.instruction.cost()
        queues[cu].append(node.instruction)
        heapq.heappush(cu_heap, (node.end_time, cu))
        scheduled += 1
        for child in node.children:
            child.remaining.discard(node)
            if not child.remaining:
                heapq.heappush(ready, (-child.instruction.cost(), index_of[child]))

    if scheduled != len(nodes):
        raise ValueError(
            f"{len(nodes) - scheduled} instructions could not be scheduled: "
            "the dependency graph has a cycle"
        )
    return queues


def round_robin_assign(instructions: list[Instruction],
                       cus: int) -> list[list[Instruction]]:
    """Deal the instructions out in order, one per CU in turn."""
    if cus < 1:
        raise ValueError(f"cus must be positive, got {cus}")
    queues: list[list[Instruction]] = [[] for _ in range(cus)]
    for index, instruction in enumerate(instructions):
        queues[index % cus].append(instruction)
    return queues


def pad_with_noops(queues: list[list[Instruction]]) -> list[list[Instruction]]:
    """Equalise row lengths with NoOps, which is what the runtime requires."""
    width = max((len(row) for row in queues), default=0)
    return [row + [NoOp() for _ in range(width - len(row))] for row in queues]


__all__ = [
    "DagNode",
    "assign_dag_to_cus",
    "linear_chain",
    "pad_with_noops",
    "round_robin_assign",
]
