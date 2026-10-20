"""Real DCU forward microbenchmark; validates precision before HIP event timing."""
import argparse
import json

import torch
from dcu_dpa4 import blas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=192)
    parser.add_argument('--n', type=int, default=16)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--batch', type=int, default=0, help='0: ordinary GEMM; >0: batched')
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.repeats) <= 0 or args.batch < 0:
        parser.error('m,n,k,repeats must be positive and batch nonnegative')
    blas._rocblas_extension()
    if not torch.cuda.is_available():
        raise RuntimeError('benchmark requires a real DCU')
    torch.set_float32_matmul_precision('highest')
    dtype = getattr(torch, args.dtype)
    prefix = (args.batch,) if args.batch else ()
    a = torch.randn(*prefix, args.m, args.k, device='cuda', dtype=dtype)
    b = torch.randn(*prefix, args.k, args.n, device='cuda', dtype=dtype)
    tolerance = 3e-4 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(blas.gemm(a, b, backend='rocblas'), a @ b,
                               atol=tolerance, rtol=tolerance)
    times = {}
    for name, call in (('torch_matmul', lambda: a @ b),
                       ('rocblas_wrapper', lambda: blas.gemm(a, b, backend='rocblas'))):
        for _ in range(10):
            call()
        torch.cuda.synchronize()
        trials = []
        for _ in range(5):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.repeats):
                result = call()
            stop.record(); stop.synchronize()
            trials.append(start.elapsed_time(stop) * 1000 / args.repeats)
            del result
        times[name] = sorted(trials)[2]
    print(json.dumps({'device': torch.cuda.get_device_name(), 'hip': torch.version.hip,
                      'torch': torch.__version__, 'shape_a': list(a.shape), 'shape_b': list(b.shape),
                      'dtype': args.dtype, 'median_event_us': times,
                      'scope': 'forward wrapper, allocation/dispatch included; no DPA4 speedup claim'},
                     indent=2))


if __name__ == '__main__':
    main()
