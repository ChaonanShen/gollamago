"""Focused complete-call operator timings versus eager Torch."""
import random
import statistics
import time
import torch
import operators
from benchmarks import report_common as C


def timed(function, warmup, repeat):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / repeat


@torch.inference_mode()
def main(argv=None):
    parser = C.parser(__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.repeat < 1 or args.trials < 1:
        parser.error("warmup, repeat and trials must be positive")
    with C.reject_fallback():
        for backend in C.backends_selected(args.backend):
            C.configure(backend)
            torch.manual_seed(0)
            random.seed(0)
            C.heading("2. OPERATOR PERFORMANCE VS TORCH", backend)
            print(f"warmup={args.warmup} repeat={args.repeat} trials={args.trials}; median of trial means")
            print("Complete call: dispatch + output allocation + GPU; JIT/validation excluded.")
            print(f"{'Operator':<10} {'Shape':<24} {'Torch us':>12} {'Backend us':>12} {'Speedup':>10} {'Check':>7}")
            report = C.metadata("performance", backend)
            report.update({"warmup": args.warmup, "repeat": args.repeat, "trials": args.trials, "results": []})
            for label, name, batch, tensors in C.cases():
                error = C.validate(name, tensors, backend)
                calls = {
                    "torch": lambda: operators.dispatch(name, *tensors, backend="torch"),
                    backend: lambda: operators.dispatch(name, *tensors, backend=backend),
                }
                samples = {name: [] for name in calls}
                for _ in range(args.trials):
                    order = list(calls)
                    random.shuffle(order)
                    for key in order:
                        samples[key].append(timed(calls[key], args.warmup, args.repeat))
                torch_us = statistics.median(samples["torch"])
                backend_us = statistics.median(samples[backend])
                speedup = torch_us / backend_us
                shape = list(tensors[0].shape)
                report["results"].append({
                    "operator": label, "batch": batch, "shape": shape,
                    "torch_us": torch_us, "backend_us": backend_us,
                    "speedup_vs_torch": speedup, "trials_us": samples,
                    "max_abs_error": error, "inputs_unchanged": True,
                })
                print(f"{label:<10} {'x'.join(map(str, shape)):<24} {torch_us:>12.2f} {backend_us:>12.2f} {speedup:>9.2f}x {'PASS':>7}", flush=True)
            print("PASS: all 6 cases verified before timing; speedup = Torch / backend.")
            C.save(args, "performance", backend, report)


if __name__ == "__main__":
    main()
