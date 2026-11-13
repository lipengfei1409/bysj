"""Compile/run the backend profiler against local libtorch (CPU is sufficient)."""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-dir", type=Path, default=Path(__file__).resolve().parents[4] / "deepmd-kit/source/api_cc/include")
    args = parser.parse_args()
    compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++")
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "smoke"
        command = [compiler, "-std=c++17", "-O0", str(Path(__file__).with_name("backend_profiler_smoke.cc")),
                   "-I" + str(args.include_dir)]
        command.extend("-I" + path for path in include_paths())
        for path in library_paths():
            command.extend(["-L" + path, "-Wl,-rpath," + path])
        command.extend(["-ltorch", "-ltorch_cpu", "-lc10", "-o", str(binary)])
        if hasattr(torch, "compiled_with_cxx11_abi"):
            command.append(f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}")
        subprocess.run(command, check=True)
        stem = Path(tmp) / "trace"
        subprocess.run([str(binary), str(stem)], check=True)
        for suffix, complete, aborted, calls in [
            ("", True, False, 3), ("_partial", False, False, 1),
            ("_aborted", False, True, 0), ("_after", True, False, 1),
        ]:
            trace = Path(str(stem) + suffix + ".json")
            summary = json.loads(Path(str(trace) + ".summary.json").read_text())
            assert summary["complete"] is complete, summary
            assert summary["aborted"] is aborted, summary
            assert summary["captured_calls"] == calls, summary
            events = json.loads(trace.read_text())["traceEvents"]
            ranges = [event for event in events if event.get("name") == "deepmd/pt_expt.compute" and event.get("ph") == "X"]
            assert len(ranges) == (calls if not aborted else 1), len(ranges)
        failed_start = json.loads(Path(str(stem) + "_startup_failure.json.summary.json").read_text())
        assert failed_start["aborted"] and not failed_start["complete"], failed_start
        print("CPU Kineto: disabled, invalid config, nested scopes, complete/partial windows and exception cleanup passed")


if __name__ == "__main__":
    main()
