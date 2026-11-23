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


def _find_project(start: Path) -> Path:
    """Return the checkout root: the nearest ancestor with ``setup.py`` and ``tests/``.

    The checkout is not always named ``myproject`` -- on the DCU host the repo
    root is the directory the shell is already in -- so the root is discovered
    instead of counted from this file's depth.  Getting this wrong runs every
    command from the wrong cwd, which is what the first version did.
    """
    for parent in start.parents:
        if (parent / "setup.py").is_file() and (parent / "tests").is_dir():
            return parent
    return start.parents[2]


_PROJECT = _find_project(_HERE)
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

    base_launches = _launch_count(baseline)
    fused_launches = _launch_count(patched)
    if base_launches is None or fused_launches is None:
        messages.append("skip launch criterion: trace_analysis.device_kernel_calls missing")
    elif fused_launches >= base_launches:
        ok = False
        messages.append(f"FAIL launch count did not drop: {base_launches} -> {fused_launches}")
    else:
        messages.append(
            f"ok   launches {base_launches} -> {fused_launches} "
            f"({fused_launches - base_launches:+d})"
        )

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


def _launch_count(summary: dict) -> int | None:
    """Kernel launches per the trace analysis.

    ``device_event_count`` counts *unique* profiler event names, which a fused
    path can raise by adding kernels, so it cannot judge the launch count.
    """
    value = (summary.get("trace_analysis") or {}).get("device_kernel_calls")
    return value if isinstance(value, int) else None


def _status_text(status: str | bool) -> str:
    """Render a stage status; the e2e verdict row carries a bool."""
    if isinstance(status, str):
        return status
    return "PASS" if status else "FAIL"


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

    requested = args.stage or ["host"]
    # Always run the ladder in its canonical order: `--stage acc --stage build`
    # must not measure an extension that has not been built yet.
    stages = list(STAGES) if "all" in requested else [s for s in STAGES if s in requested]
    if not str(args.out_dir).strip():
        args.out_dir = DEFAULT_END_TO_END_OUT
    if "e2e" in stages and not (args.model_file and args.input_npz):
        # Fail before the 30-minute ladder, not after it: an unset $MODEL would
        # otherwise only surface at the last rung.
        print("--stage e2e needs --model-file and --input-npz", file=sys.stderr)
        return 2
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

    print(f"project: {_PROJECT}")
    print(f"out dir: {args.out_dir}\n", flush=True)
    # Only the stages that write artifacts need the directory, so a host-only
    # run stays possible where /root is not writable.
    if {"lat", "e2e"} & set(stages):
        try:
            args.out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(f"cannot create out dir {args.out_dir}: {error}", file=sys.stderr)
            return 2
    # A ladder: a failed rung makes every later rung meaningless (no extension,
    # no numbers, nothing to compare), so stop instead of piling up failures.
    results: list[tuple[str, str, str]] = []
    failed_stage: str | None = None
    for stage in stages:
        if failed_stage is not None:
            results.append((stage, "SKIP", f"not run: {failed_stage} failed"))
            continue
        entries = plan.get(stage) or []
        if not entries:
            results.append((stage, "SKIP", "no command configured (pass --model-file and --input-npz)"))
            continue
        env = _base_env(_PROJECT)
        for command, extra, label in entries:
            code = _run(command, dict(env, **extra), _PROJECT)
            if code != 0:
                results.append((stage, "FAIL", f"{label}: exit {code}"))
                failed_stage = stage
                break
        else:
            results.append((stage, "PASS", "all commands exited 0"))

    if "e2e" in stages and failed_stage is None:
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
            results.append(("e2e verdict", "PASS" if ok else "FAIL",
                            "see criteria above"))
        else:
            results.append(("e2e verdict", "FAIL", "summaries not found"))

    print("\n-- summary --")
    for label, status, detail in results:
        print(f"  {_status_text(status):4s}  {label:20s} {detail}")
    return 1 if any(_status_text(status) == "FAIL" for _, status, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
