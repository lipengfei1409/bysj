"""Run the fused-operator validation ladder in one command.

The ladder has five rungs, each answering a different question (see
``docs/operators.md`` §4):

===== ============================== =========================================
rung  question                       needs
===== ============================== =========================================
host  do the references and contracts hold?                nothing (any host)
build is the HIP extension built?                          DCU + hipcc
acc   is the operator numerically right?                   DCU
lat   is the operator faster than the ATen chain?          DCU
par   does the seam match DeepMD's own implementation?      DCU (no checkpoint)
e2e   is the model faster?                                  DCU + model + input
===== ============================== =========================================

End-to-end is the only rung that decides whether a fusion is worth keeping: the
project's standing lesson is that `segment_softmax` is 9.09x as an operator and
0.0 ms end to end.  This runner therefore prints the three e2e criteria (seam
hits, launch count, device time) and fails if any of them is not met.

Usage::

    # what would run?
    python tests/deepmd/profile/verify_fused.py --dry-run

    # the cheap rungs on any host
    python tests/deepmd/profile/verify_fused.py --stage host

    # the full ladder on a DCU
    MODEL=/models/dpa4_air.pt NPZ=/tmp/water_3000.npz \\
    python tests/deepmd/profile/verify_fused.py --stage all \\
        --model-file "$MODEL" --input-npz "$NPZ"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
STAGES = ("host", "build", "acc", "lat", "par", "e2e")
DEFAULT_END_TO_END_OUT = Path("/root/private_data/lipengfei/test_result/verify")


def _python() -> str:
    return sys.executable or "python3"


def _base_env(project: Path) -> dict:
    env = dict(os.environ)
    python_dir = str(project / "python")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = python_dir + (os.pathsep + existing if existing else "")
    return env


def build_plan(
    *,
    project: Path = _PROJECT,
    out_dir: Path = DEFAULT_END_TO_END_OUT,
    model_file: str | None = None,
    input_npz: str | None = None,
    device_name: str = "K500SM_AI",
) -> dict[str, list[tuple[list[str], dict, str]]]:
    """Return ``{stage: [(argv, extra_env, label), ...]}``.

    Pure construction on purpose: ``--dry-run`` prints this and the tests assert
    it, so a typo in a unittest pattern shows up on a machine that has no DCU.
    """
    py = _python()
    plan: dict[str, list[tuple[list[str], dict, str]]] = {}

    plan["host"] = [
        ([py, "-m", "unittest", "discover", "-s", "tests/python"], {}, "python contracts"),
    ]
    plan["build"] = [
        ([py, "setup.py", "build_ext", "--inplace"], {"MAX_JOBS": "4"}, "HIP extension"),
    ]
    plan["acc"] = [
        ([py, "-m", "unittest", "discover", "-s", "tests/csrc/accuracy",
          "-p", "test_segment_ops*", "-v"], {}, "segment ops accuracy"),
        ([py, "-m", "unittest", "discover", "-s", "tests/csrc/accuracy",
          "-p", "test_slot_ops*", "-v"], {}, "slot ops accuracy"),
    ]
    plan["lat"] = [
        ([py, "-m", "unittest", "discover", "-s", "tests/csrc/latency",
          "-p", "test_segment_ops*", "-v"],
         {"DPA4_RUN_SEGMENT_LATENCY": "1",
          "DPA4_SEGMENT_LATENCY_JSON": str(out_dir / "seg_latency.json")},
         "segment ops A/B latency"),
        ([py, "-m", "unittest", "discover", "-s", "tests/csrc/latency",
          "-p", "test_slot_ops*", "-v"],
         {"DPA4_RUN_SLOT_LATENCY": "1",
          "DPA4_SLOT_LATENCY_JSON": str(out_dir / "slot_latency.json")},
         "slot ops A/B latency"),
    ]
    plan["par"] = [
        ([py, "-m", "unittest", "discover", "-s", "tests/csrc/accuracy",
          "-p", "test_fused_deepmd_parity.py", "-v"],
         {"DP_RUN_FUSED_DEEPMD_PARITY": "1"}, "seam parity vs DeepMD"),
    ]
    if not model_file or not input_npz:
        plan["e2e"] = []
    else:
        common = [
            "--model-file", model_file,
            "--input-npz", input_npz,
            "--fixed-input",
            "--expected-dcu-name", device_name,
        ]
        plan["e2e"] = [
            ([py, str(_HERE / "test_dpa4_inference_profile.py"), "dpa4", *common,
              "--output-dir", str(out_dir / "off")], {}, "e2e baseline"),
            ([py, str(_HERE / "test_dpa4_inference_profile.py"), "dpa4", *common,
              "--fused", "on", "--output-dir", str(out_dir / "on")], {}, "e2e fused on"),
        ]
    return plan


def judge(baseline: dict, patched: dict) -> tuple[bool, list[str]]:
    """Apply the three end-to-end criteria and return ``(ok, messages)``."""
    from compare_fused_runs import _device_ms

    messages: list[str] = []
    ok = True

    fused = patched.get("fused") or {}
    hits = {key: value for key, value in fused.items() if key.endswith("_hits")}
    fallbacks = {key: value for key, value in fused.items() if key.endswith("_fallbacks")}
    dispatched = {key: value for key, value in hits.items() if value}
    if not dispatched:
        ok = False
        messages.append(f"FAIL no seam dispatched: hits={hits or 'none'}")
    else:
        messages.append("ok   dispatched: " + ", ".join(f"{k}={v}" for k, v in sorted(dispatched.items())))
    broken = {key: value for key, value in fallbacks.items() if value}
    if broken:
        ok = False
        messages.append("FAIL fallbacks: " + ", ".join(f"{k}={v}" for k, v in sorted(broken.items())))
    if fused.get("last_error"):
        ok = False
        messages.append(f"FAIL last_error: {fused['last_error']}")

    base_launches = baseline.get("device_event_count")
    fused_launches = patched.get("device_event_count")
    if isinstance(base_launches, int) and isinstance(fused_launches, int):
        if fused_launches >= base_launches:
            ok = False
            messages.append(f"FAIL launch count did not drop: {base_launches} -> {fused_launches}")
        else:
            messages.append(f"ok   launches {base_launches} -> {fused_launches} ({fused_launches - base_launches:+d})")

    base_ms, fused_ms = _device_ms(baseline), _device_ms(patched)
    if base_ms is None or fused_ms is None:
        ok = False
        messages.append("FAIL no device total in one of the summaries")
    elif fused_ms >= base_ms:
        ok = False
        messages.append(
            f"FAIL device time did not drop: {base_ms:.1f} -> {fused_ms:.1f} ms "
            f"({base_ms / fused_ms if fused_ms else 0:.3f}x)"
        )
    else:
        messages.append(
            f"ok   device {base_ms:.1f} -> {fused_ms:.1f} ms "
            f"({base_ms / fused_ms:.3f}x, {(fused_ms - base_ms):+.1f} ms)"
        )
    return ok, messages


def _latest_summary(root: Path) -> Path | None:
    found = sorted(glob.glob(str(root / "**" / "dpa4_inference_profile_summary.json"), recursive=True))
    return Path(found[-1]) if found else None


def _run(command: list[str], env: dict, cwd: Path) -> int:
    print("$ " + " ".join(command), flush=True)
    return subprocess.call(command, cwd=str(cwd), env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", action="append", choices=(*STAGES, "all"), default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_END_TO_END_OUT)
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--input-npz", default=None)
    parser.add_argument("--expected-dcu-name", default="K500SM_AI")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    stages = args.stage or ["host"]
    if "all" in stages:
        stages = list(STAGES)
    plan = build_plan(
        out_dir=args.out_dir,
        model_file=args.model_file,
        input_npz=args.input_npz,
        device_name=args.expected_dcu_name,
    )

    if args.dry_run:
        for stage in stages:
            print(f"[{stage}]")
            for command, extra, label in plan.get(stage) or []:
                print(f"  # {label}")
                print("  $ " + " ".join(command))
                for key, value in extra.items():
                    print(f"    env {key}={value}")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, bool, str]] = []
    for stage in stages:
        entries = plan.get(stage) or []
        if not entries:
            results.append((stage, False, "no command configured (pass --model-file/--input-npz)"))
            continue
        env = _base_env(_PROJECT)
        ok = True
        for command, extra, label in entries:
            stage_env = dict(env, **extra)
            code = _run(command, stage_env, _PROJECT)
            if code != 0:
                ok = False
                results.append((label, False, f"exit {code}"))
                break
        if ok:
            results.append((stage, True, "all commands exited 0"))

    if "e2e" in stages:
        base = _latest_summary(args.out_dir / "off")
        fused = _latest_summary(args.out_dir / "on")
        if base and fused:
            from compare_fused_runs import compare

            with base.open() as handle:
                baseline_summary = json.load(handle)
            with fused.open() as handle:
                fused_summary = json.load(handle)
            ok, messages = judge(baseline_summary, fused_summary)
            print("\n-- end-to-end verdict --")
            for message in messages:
                print("  " + message)
            print()
            print(json.dumps(compare(baseline_summary, fused_summary), indent=2))
            results.append(("e2e verdict", ok, "see criteria above"))
        else:
            results.append(("e2e verdict", False, "summaries not found"))

    print("\n-- summary --")
    for label, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:20s} {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
