"""HIP forward operator benchmark. Validate numerical parity BEFORE timing."""
import argparse
import json
import statistics

import torch

from dcu_dpa4 import ops
from dcu_dpa4.native import native_available


def time_call(fn, warmup, iterations, trials):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / iterations)
    return {"median_us": statistics.median(times), "min_us": min(times), "trials_us": times}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=192)
    parser.add_argument("--edges", type=int, default=12288)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--radial", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if min(args.atoms, args.edges, args.channels, args.radial, args.iterations, args.trials) <= 0 or args.warmup < 0:
        parser.error("dimensions/iterations/trials must be positive, warmup nonnegative")
    if not (torch.version.hip and torch.cuda.is_available() and native_available()):
        raise RuntimeError("benchmark requires ROCm PyTorch, DCU and a built dcu_dpa4._C; no fallback")
    torch.manual_seed(42)
    n,e,c,r = args.atoms,args.edges,args.channels,args.radial
    pos=torch.randn(n,3,device="cuda")
    edges=torch.randint(n,(2,e),device="cuda")
    shifts=torch.zeros(e,3,dtype=torch.int64,device="cuda")
    cells=torch.eye(3,device="cuda")[None]*10
    frame=torch.zeros(e,dtype=torch.int64,device="cuda")
    distance=torch.rand(e,1,device="cuda")*6
    frequencies=torch.arange(1,r+1,device="cuda",dtype=torch.float32)[None]*.52
    x=torch.randn(e,c,device="cuda"); scale=torch.ones(c,device="cuda")
    weight=torch.randn(c,c,device="cuda")*.1
    gradient=torch.randn(e,3,device="cuda"); vectors=torch.randn_like(gradient)
    mask=torch.ones(e,dtype=torch.bool,device="cuda")
    cases={
        "edge_geometry": lambda b:ops.edge_geometry(pos,edges,shifts,cells,frame,backend=b),
        "radial": lambda b:ops.radial(distance,frequencies,6.,backend=b),
        "rmsnorm": lambda b:ops.rmsnorm(x,scale,backend=b),
        "linear": lambda b:ops.linear(x,weight,backend=b),
        "segment_sum": lambda b:ops.segment_sum(x,edges[1],n,backend=b),
        "force_virial": lambda b:ops.force_virial(gradient,vectors,edges,frame,mask,n,1,backend=b),
    }
    report={"device":torch.cuda.get_device_name(),"torch":torch.__version__,"hip":torch.version.hip,
        "sizes":vars(args),"timing_scope":"forward wrapper + launches + allocation/zeroing; inputs resident on DCU; no H2D or backward",
        "operators":{}}
    with torch.inference_mode():
        for name,fn in cases.items():
            ref,actual=fn("reference"),fn("hip")
            ref = ref if isinstance(ref,(tuple,list)) else (ref,)
            actual = actual if isinstance(actual,(tuple,list)) else (actual,)
            # Atomic reductions accumulate in FP32; large E needs an absolute error budget.
            for a,b in zip(actual,ref):
                torch.testing.assert_close(a,b,rtol=5e-4,atol=2e-3)
            report["operators"][name] = {"parity":"passed"}
            for backend in ("reference","hip"):
                report["operators"][name][backend] = time_call(lambda:fn(backend),args.warmup,args.iterations,args.trials)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
