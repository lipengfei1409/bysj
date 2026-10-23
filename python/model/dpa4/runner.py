"""End-to-end DPA4 reference runner backed by DeePMD's own model.

The project-owned operators in ``python/dcu_dpa4`` are independently testable; this runner
retains DeePMD for the not-yet-ported SO(3)/SO(2), FFN and fitting stages.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .checkpoint import CheckpointSpec, load_checkpoint_spec
from .neighbor_bridge import cpu_vesin_search
from .system import DeepmdSystem
from dcu_dpa4.deepmd import configure_eager,install_operators


def configure_device(device: str) -> None:
    """Configure the DeePMD process before any DeePMD backend import."""
    if device not in ("cpu","dcu"):
        raise ValueError("device must be cpu or dcu")
    configure_eager(device)


def compute_metrics(
    predicted_energy: np.ndarray,
    predicted_force: np.ndarray,
    reference_energy: np.ndarray,
    reference_force: np.ndarray,
) -> dict[str, float]:
    """Return per-system energy and componentwise force RMSE."""
    predicted_energy = np.asarray(predicted_energy)
    predicted_force = np.asarray(predicted_force)
    reference_energy = np.asarray(reference_energy)
    reference_force = np.asarray(reference_force)
    if predicted_energy.shape != reference_energy.shape:
        raise ValueError("energy prediction and labels have different shapes")
    if predicted_force.shape != reference_force.shape or predicted_force.ndim != 3:
        raise ValueError("force prediction and labels must have shape [F, N, 3]")
    n_atoms = predicted_force.shape[1]
    if n_atoms == 0:
        raise ValueError("cannot compute per-atom metrics for an empty system")
    energy_rmse = float(
        np.sqrt(np.mean((predicted_energy - reference_energy).astype(np.float64) ** 2))
    )
    force_rmse = float(
        np.sqrt(np.mean((predicted_force - reference_force).astype(np.float64) ** 2))
    )
    return {
        "energy_rmse_ev": energy_rmse,
        "energy_rmse_per_atom_ev": energy_rmse / n_atoms,
        "force_rmse_ev_per_angstrom": force_rmse,
    }


@dataclass(frozen=True)
class Prediction:
    energy: np.ndarray  # [F, 1], eV
    force: np.ndarray  # [F, N, 3], eV/Å
    virial: np.ndarray  # [F, 3, 3], eV


class DPA4Runner:
    """Run original DPA4 or opt-in instance-local primitive adapters.

    DeePMD owns the whole numerical model in either case. The independent
    shape scaffold is not used in this path. Set device before backend import.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        head: str | None = None,
        neighbor_backend: str = "native",
        operator_backend: str = "deepmd",
        operators: tuple[str,...] = ("radial","rmsnorm","linear"),
        device: str | None = None,
    ) -> None:
        if neighbor_backend not in ("native", "cpu-bridge"):
            raise ValueError("neighbor_backend must be 'native' or 'cpu-bridge'")
        if operator_backend not in ("deepmd","reference","hip"):
            raise ValueError("operator_backend must be deepmd, reference or hip")
        if sys.version_info<(3,10):
            raise RuntimeError("DeePMD integration requires Python >=3.10; shape/reference primitives can still run on 3.9")
        if device is not None:
            configure_device(device)
        elif operator_backend!="deepmd":
            configure_eager()
        self.spec: CheckpointSpec = load_checkpoint_spec(checkpoint, head=head)
        self.neighbor_backend = neighbor_backend
        self.operator_backend=operator_backend
        self._operator_handle=None
        self._closed=False
        try:
            from deepmd.infer import DeepPot
        except ImportError as exc:
            raise RuntimeError("full-model inference requires installed DeePMD-kit with compiled libraries and DPA4 dependencies") from exc
        self._potential = DeepPot(str(self.spec.path), head=head,no_jit=operator_backend!="deepmd")
        if operator_backend!="deepmd":
            try:
                self._operator_handle=install_operators(self.torch_model,backend=operator_backend,operators=operators)
            except Exception:
                self._potential.close()
                raise

    @property
    def potential(self):
        """Original DeepPot: eval/forward_lower semantics remain upstream-owned."""
        return self._potential

    @property
    def torch_model(self):
        return self._potential.get_model()

    @property
    def operator_report(self):
        return [] if self._operator_handle is None else self._operator_handle.report()

    def close(self):
        if not self._closed:
            if self._operator_handle is not None:
                self._operator_handle.restore()
            self._potential.close()
            self._closed=True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("runner is closed")
        return self

    def __exit__(self,*exc):
        self.close()

    def eval(self, system: DeepmdSystem, **eval_kwargs) -> Prediction:
        """Forward optional fparam/aparam/charge_spin arguments to DeepPot.eval."""
        if self._closed:
            raise RuntimeError("runner is closed")
        if system.coords.shape[0] == 0:
            raise ValueError("system contains no frames")
        if system.type_map:
            try:
                remap = np.asarray(
                    [self.spec.type_map.index(element) for element in system.type_map],
                    dtype=np.int64,
                )
            except ValueError as exc:
                raise ValueError("system contains an element absent from the checkpoint") from exc
            atype = remap[system.atype]
        else:
            atype = system.atype
        if np.any(atype < 0) or np.any(atype >= len(self.spec.type_map)):
            raise ValueError("atom type index is outside the checkpoint type map")
        cells = None if system.cells is None else system.cells.reshape(-1, 9)
        context = cpu_vesin_search() if self.neighbor_backend == "cpu-bridge" else nullcontext()
        with context:
            energy, force, virial = self._potential.eval(system.coords, cells, atype,**eval_kwargs)
        nframes, n_atoms, _ = system.coords.shape
        return Prediction(
            np.asarray(energy).reshape(nframes, 1),
            np.asarray(force).reshape(nframes, n_atoms, 3),
            np.asarray(virial).reshape(nframes, 3, 3),
        )
