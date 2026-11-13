"""Opt-in native GIE parity test against DeepMD's PyTorch reference.

Run on a supported DTK/DCU host after building the HIP extension:

    source /opt/dtk/env.sh
    # DeepMD-kit must be built/installed with its PyTorch backend first.
    export PYTHONPATH="$PWD/python:$PWD/../deepmd-kit${PYTHONPATH:+:$PYTHONPATH}"
    MAX_JOBS=4 python setup.py build_ext --inplace
    DP_RUN_GIE_DEEPMD_PARITY=1 python tests/csrc/accuracy/test_gie_deepmd_parity.py -v

This compares the standalone native zonal-scatter stage with DeepMD's direct
GeometricInitialEmbedding module on sparse, all-valid edges. Spin, source
gates, and padded-edge masks are not part of this primitive-level check.
It does not compare the full descriptor.
"""
import os
import unittest

import torch

from kernels.gie import gie_available, zonal_scatter


_RUN_PARITY = os.environ.get("DP_RUN_GIE_DEEPMD_PARITY") == "1"


@unittest.skipUnless(
    _RUN_PARITY,
    "set DP_RUN_GIE_DEEPMD_PARITY=1 to run native-vs-DeepMD GIE parity",
)
class GIEDeepMDParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DP_RUN_GIE_DEEPMD_PARITY=1 requires a visible supported DCU"
            )
        if not gie_available():
            raise RuntimeError(
                "DP_RUN_GIE_DEEPMD_PARITY=1 requires the built GIE HIP extension "
                "and a supported DCU"
            )
        try:
            from deepmd.pt.model.descriptor.sezm_nn.edge_cache import EdgeFeatureCache
            from deepmd.pt.model.descriptor.sezm_nn.embedding import (
                GeometricInitialEmbedding,
            )
        except Exception as exc:
            raise RuntimeError(
                "DeepMD-kit must be built/installed with its PyTorch backend "
                "and importable (including deepmd.lib); see the test docstring"
            ) from exc

        cls.EdgeFeatureCache = EdgeFeatureCache
        cls.GeometricInitialEmbedding = GeometricInitialEmbedding
        cls.device = torch.device("cuda")

    @staticmethod
    def _inputs(module, *, lmax, channels, seed):
        device = torch.device("cuda")
        n_nodes = 11
        n_edges = 37
        n_rows = (lmax + 1) ** 2 - 1
        generator = torch.Generator(device=device).manual_seed(seed)

        # This deterministic pattern creates many colliding destinations while
        # keeping every edge valid and every index in range.
        dst = (torch.arange(n_edges, device=device, dtype=torch.int64) % n_nodes)
        src = (dst + 1) % n_nodes
        zonal = torch.randn(
            n_edges, n_rows, device=device, dtype=torch.float32, generator=generator
        ).contiguous()
        radial = torch.randn(
            n_edges, lmax, channels,
            device=device,
            dtype=torch.float32,
            generator=generator,
        ).contiguous()
        edge_vec = torch.randn(
            n_edges, 3, device=device, dtype=torch.float32, generator=generator
        ).contiguous()
        edge_env = torch.ones(n_edges, 1, device=device, dtype=torch.float32)
        deg = torch.zeros(n_nodes, device=device, dtype=torch.float32)
        deg.index_add_(0, dst, edge_env[:, 0].square())
        inv_sqrt_deg = torch.rsqrt(deg + 1.0).reshape(n_nodes, 1, 1).contiguous()

        radial_slot = module.radial_slot_index_for_row.to(
            device=device, dtype=torch.int64
        ).contiguous()
        edge_cache = GIEDeepMDParityTests.EdgeFeatureCache(
            src=src.contiguous(),
            dst=dst.contiguous(),
            edge_type_feat=torch.zeros(
                n_edges, channels, device=device, dtype=torch.float32
            ),
            edge_vec=edge_vec,
            edge_rbf=torch.zeros(n_edges, 1, device=device, dtype=torch.float32),
            edge_env=edge_env,
            deg=deg,
            inv_sqrt_deg=inv_sqrt_deg,
            # The explicit zonal values below make Wigner-D blocks unnecessary.
            D_full=None,
            Dt_full=None,
            csr_cache=None,
            edge_src_gate=None,
        )
        return zonal, radial, dst.contiguous(), radial_slot, inv_sqrt_deg, edge_cache

    def test_forward_output_matches_deepmd(self):
        channels = 8
        for lmax in (1, 2):
            with self.subTest(lmax=lmax):
                module = self.GeometricInitialEmbedding(
                    lmax=lmax, channels=channels, dtype=torch.float32
                ).to(self.device)
                # Keep DeepMD on its reference composition, not an optional fused
                # path; csr_cache=None also prevents the fused branch.
                module.train()
                zonal, radial, dst, radial_slot, inv_sqrt_deg, edge_cache = (
                    self._inputs(
                        module,
                        lmax=lmax,
                        channels=channels,
                        seed=20261113 + lmax,
                    )
                )
                with torch.no_grad():
                    expected = module(
                        n_nodes=inv_sqrt_deg.shape[0],
                        edge_cache=edge_cache,
                        radial_feat=radial,
                        zonal_coupling=zonal,
                    )
                    actual = zonal_scatter(
                        zonal, radial, dst, radial_slot, inv_sqrt_deg
                    )

                self.assertEqual(
                    tuple(actual.shape),
                    (inv_sqrt_deg.shape[0], (lmax + 1) ** 2, channels),
                )
                self.assertEqual(
                    torch.count_nonzero(actual[:, 0, :]).item(), 0,
                    "native GIE must leave the scalar row reserved and zero",
                )
                self.assertEqual(
                    torch.count_nonzero(expected[:, 0, :]).item(), 0,
                    "DeepMD GIE must leave the scalar row reserved and zero",
                )

                difference = (actual - expected).abs()
                max_abs = difference.max().item()
                rms = difference.square().mean().sqrt().item()
                max_rel = (
                    difference / expected.abs().clamp_min(2e-6)
                ).max().item()
                try:
                    torch.testing.assert_close(
                        actual, expected, rtol=2e-5, atol=2e-6
                    )
                except AssertionError as exc:
                    self.fail(
                        f"lmax={lmax}: GIE output differs from DeepMD; "
                        f"max_abs={max_abs:.8g}, rms={rms:.8g}, "
                        f"max_rel={max_rel:.8g}\n{exc}"
                    )
                print(
                    f"lmax={lmax}: max_abs={max_abs:.8g}, "
                    f"rms={rms:.8g}, max_rel={max_rel:.8g}"
                )


if __name__ == "__main__":
    unittest.main()
