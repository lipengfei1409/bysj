"""Execution helpers for model backends that need coordinate autograd."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import torch


def run_model(model, batch, graph=None, *, compute_forces=True, **kwargs):
    """Run a DPA4 dataflow with the backend's required gradient context.

    Model definitions stay independent of numerical backends; runtime supplies
    coordinate-gradient setup for implementations that request it.
    """
    backend = model.backend
    tracks_coordinates = compute_forces and bool(
        getattr(backend, "requires_coordinate_gradients", False)
    )
    context = torch.enable_grad() if tracks_coordinates else nullcontext()
    if tracks_coordinates and not batch.positions.requires_grad:
        batch = replace(
            batch,
            positions=batch.positions.detach().requires_grad_(True),
        )
    with context:
        return model(
            batch,
            graph,
            compute_forces=compute_forces,
            **kwargs,
        )
