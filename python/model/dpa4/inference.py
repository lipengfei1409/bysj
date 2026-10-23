"""Generic-framework adapter for the full DeePMD-backed DPA4 path."""
from __future__ import annotations


class DPA4DeepMDAdapter:
    """Adapt a DPA4Runner without changing its system/E/F/V contract."""

    def __init__(
        self,
        *,
        checkpoint,
        head=None,
        device=None,
        neighbor_backend="native",
        operator_backend="deepmd",
        operators=("radial", "rmsnorm", "linear"),
    ):
        from .runner import DPA4Runner

        self._runner = DPA4Runner(
            checkpoint,
            head=head,
            device=device,
            neighbor_backend=neighbor_backend,
            operator_backend=operator_backend,
            operators=tuple(operators),
        )

    def predict(self, request, **options):
        from .system import DeepmdSystem

        if not isinstance(request, DeepmdSystem):
            raise TypeError("DPA4 inference expects a DeepmdSystem request")
        return self._runner.eval(request, **options)

    def close(self):
        self._runner.close()


def create_adapter(**config):
    return DPA4DeepMDAdapter(**config)
