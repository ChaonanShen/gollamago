"""Sweep MACA BF16 RoPE block_rows/threads; correctness precedes timing.

Run after building the extension:
  python -m benchmarks.tune_maca_rope --output /data/gollamago-data/rope-tuning.json
CUDA Graph timing excludes allocation/dispatch; call timing includes both.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch

import backends
from operators.maca_cpp import maca_kernels
from operators.torch_ops import rope as reference
from operators.tilelang_ops import rope as tilelang_rope


def gpu_us(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            output = fn()
    samples = []
    for _ in range(3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 400)
    return statistics.median(samples)


def call_us(fn):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(200):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / 200)
    return statistics.median(samples)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-rows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--threads", type=int, nargs="+", default=[64, 128, 256, 512])
    args = parser.parse_args()
    if not getattr(torch.version, "maca", None):
        parser.error("requires MACA PyTorch")
    backends.configure_backend("tilelang", "cuda", "maca")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)
    shapes = [
        (b, s, h, 64) for h in (8, 32)
        for b, s in ((1, 1), (1, 33), (1, 128), (1, 512), (1, 2048),
                     (1, 4906), (4, 129), (8, 1024))
    ] + [(1, s, h, 128) for h in (8, 24, 32) for s in (128, 2048)]
    records = []
    configs = list(itertools.product(args.block_rows, args.threads))
    for shape in shapes:
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        angles = torch.randn(shape[1], shape[-1] // 2, device="cuda")
        sin, cos = angles.sin().bfloat16(), angles.cos().bfloat16()
        saved = [t.clone() for t in (x, sin, cos)]
        expected = reference(x, sin, cos)
        record = {"shape": list(shape), "dtype": "bfloat16", "configs_gpu_us": {}}
        shuffled = configs[:]
        random.shuffle(shuffled)
        for br, threads in shuffled:
            fn = lambda: maca_kernels.rope(x, sin, cos, block_rows=br, threads=threads)
            torch.testing.assert_close(fn(), expected, rtol=.04, atol=.04)
            record["configs_gpu_us"][f"{br}/{threads}"] = gpu_us(fn)
        for value, before in zip((x, sin, cos), saved):
            torch.testing.assert_close(value, before, rtol=0, atol=0)
        best = min(record["configs_gpu_us"], key=record["configs_gpu_us"].get)
        br, threads = map(int, best.split("/"))
        record["best"] = best
        record["best_call_us"] = call_us(lambda: maca_kernels.rope(
            x, sin, cos, block_rows=br, threads=threads))
        torch_fn = lambda: reference(x, sin, cos)
        record["torch_gpu_us"] = gpu_us(torch_fn)
        record["torch_call_us"] = call_us(torch_fn)
        torch.testing.assert_close(tilelang_rope(x, sin, cos), expected, rtol=.04, atol=.04)
        record["tilelang_gpu_us"] = gpu_us(lambda: tilelang_rope(x, sin, cos))
        record["tilelang_call_us"] = call_us(lambda: tilelang_rope(x, sin, cos))
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
    scores = {}
    for dim in (64, 128):
        subset = [r for r in records if r["shape"][-1] == dim]
        scores[str(dim)] = {
            f"{br}/{th}": math.exp(statistics.mean(
                math.log(r["configs_gpu_us"][f"{br}/{th}"] / min(r["configs_gpu_us"].values()))
                for r in subset))
            for br, th in configs
        }
    args.output.with_name(args.output.stem + "-scores.json").write_text(json.dumps(scores, indent=2))
    print("SCORES", json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
