"""Diagnostic ROCm bridge: CPU vesin search, GPU model arithmetic.

This is a process-global temporary patch and is deliberately not a production
neighbor-list implementation. It is needed while the installed vesin_torch
extension tries to load NVIDIA libcuda.so on the target ROCm cluster.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Iterator

import torch


_patch_lock = RLock()


@contextmanager
def cpu_vesin_search() -> Iterator[None]:
    """Keep only non-differentiable neighbor search on CPU for one inference."""
    import deepmd.pt_expt.utils.vesin_graph_builder as graph_builder

    with _patch_lock:
        original = graph_builder.vesin_search_ijs

        def bridged_search(positions, cell, periodic, rcut, device):
            if positions.device.type == "cpu":
                return original(positions, cell, periodic, rcut, device)
            target = torch.device(device)
            ii, jj, ss = original(
                positions.detach().to("cpu"),
                None if cell is None else cell.detach().to("cpu"),
                periodic,
                rcut,
                torch.device("cpu"),
            )
            return ii.to(target), jj.to(target), ss.to(target)

        graph_builder.vesin_search_ijs = bridged_search
        try:
            yield
        finally:
            graph_builder.vesin_search_ijs = original
