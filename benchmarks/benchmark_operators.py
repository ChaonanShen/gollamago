"""Compare standalone operator call latency with the registered Torch reference.

Run from the repository root with python -m benchmarks.benchmark_operators.
"""

import argparse
from dataclasses import dataclass
import time
import warnings

import torch

import backends
import operators
from llama import generate_sin_and_cos_tables


@dataclass(frozen=True)
class LlamaProfile:
    hidden_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    source: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    dtype: str = "bfloat16"


# Dimensions and defaults verified against these config.json files.
# Input activations/weights are synthetic; no model weights are downloaded.
MODEL_PROFILES = {
    "llama3.2-1b": LlamaProfile(
        hidden_size=2048, query_heads=32, key_value_heads=8, head_dim=64,
        source="https://modelscope.cn/models/LLM-Research/Llama-3.2-1B/resolve/master/config.json",
    ),
    "llama3.2-3b": LlamaProfile(
        hidden_size=3072, query_heads=24, key_value_heads=8, head_dim=128,
        source="https://modelscope.cn/models/LLM-Research/Llama-3.2-3B/resolve/master/config.json",
    ),
    "llama3.1-8b": LlamaProfile(
        hidden_size=4096, query_heads=32, key_value_heads=8, head_dim=128,
        source="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B/resolve/master/config.json",
    ),
}


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
TOLERANCES = {
    "rms_norm": {
        torch.float32: (1e-5, 1e-5),
        torch.float16: (4e-3, 7e-3),
        torch.bfloat16: (2e-2, 7e-2),
    },
    "rope": {
        torch.float32: (1e-5, 1e-5),
        torch.float16: (4e-3, 4e-3),
        torch.bfloat16: (4e-2, 4e-2),
    },
}


def timed(function, device, warmup, repeat):
    """Average call time in microseconds, including dispatch and allocation."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeat):
        function()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1e6 / repeat



def assert_close_chunked(actual, expected, *, rtol, atol, chunk_elements=1 << 20, max_error=False):
    """Validate all elements with bounded temporary memory, outside timing."""
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} != {expected.shape}")
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    actual_flat, expected_flat = actual.reshape(-1), expected.reshape(-1)
    error = 0.0
    for start in range(0, actual.numel(), chunk_elements):
        a = actual_flat[start:start + chunk_elements]
        e = expected_flat[start:start + chunk_elements]
        torch.testing.assert_close(a, e, rtol=rtol, atol=atol)
        if max_error:
            error = max(error, (a.float() - e.float()).abs().max().item())
    return error


def benchmark_case(name, tensors, backend, device, warmup, repeat, model_name, role=""):
    torch_call = lambda: operators.dispatch(name, *tensors, backend="torch")
    backend_call = lambda: operators.dispatch(name, *tensors, backend=backend)
    before = [value.clone() if isinstance(value, torch.Tensor) else value for value in tensors]

    # The first backend call compiles the kernel before any timed iterations.
    expected = torch_call()
    actual = backend_call()
    torch.cuda.synchronize(device)
    rtol, atol = TOLERANCES[name][tensors[0].dtype]
    max_error = assert_close_chunked(actual, expected, rtol=rtol, atol=atol, max_error=True)
    for value, saved in zip(tensors, before):
        if isinstance(value, torch.Tensor):
            assert_close_chunked(value, saved, rtol=0, atol=0)
    del actual, expected, before

    torch_us = timed(torch_call, device, warmup, repeat)
    backend_us = timed(backend_call, device, warmup, repeat)
    speedup = torch_us / backend_us
    shape = "x".join(str(size) for size in tensors[0].shape)
    label = f"{name}({role})" if role else name
    print(
        f"{model_name:<14} {label:<10} {shape:<20} {'PASS':<7} {backend_us:>13.2f} "
        f"{torch_us:>13.2f} {speedup:>9.2f}x {max_error:>13.3e}",
        flush=True,
    )


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("all", "rms_norm", "rope"), default="all")
    parser.add_argument("--backend", choices=("tilelang", "maca_cpp", "ninetoothed"), default="tilelang")
    parser.add_argument("--target", choices=backends.TARGET_NAMES, default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_PROFILES,
        default=["llama3.2-1b"], help="default: Llama-3.2-1B; other profiles remain selectable",
    )
    parser.add_argument("--dtype", choices=DTYPES, help="override the model's default BF16 dtype")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512, 2048, 4906])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    return parser


@torch.inference_mode()
def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be nonnegative and --repeat must be positive")
    if args.batch_size <= 0 or any(sequence <= 0 for sequence in args.seq_lens):
        parser.error("--batch-size and --seq-lens must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("an available CUDA or MACA accelerator is required")
    selected = ("rms_norm", "rope") if args.operator == "all" else (args.operator,)

    with warnings.catch_warnings(), torch.cuda.device(device):
        warnings.filterwarnings("error", message="(?s).*using torch", category=RuntimeWarning)
        config = backends.configure_backend(args.backend, device, args.target)
        if config.backend != args.backend:
            raise RuntimeError(f"Requested {args.backend}, but configured {config.backend}")
        registered = operators.get_registered_operators(args.backend)
        for name in selected:
            if registered.get(name) != args.backend:
                raise RuntimeError(f"{args.backend} has no native registered {name} operator")
        torch.manual_seed(0)
        print(
            f"device={torch.cuda.get_device_name(device)} target={config.target} "
            f"warmup={args.warmup} repeat={args.repeat}",
            flush=True,
        )
        print("Timing: synchronized batch mean; includes dispatch and output allocation. JIT excluded.")
        print(
            f"{'model':<14} {'operator':<10} {'input shape':<20} {'correct':<7} "
            f"{args.backend + '(us)':>13} {'torch(us)':>13} {'speedup':>10} {'max_abs_error':>13}",
            flush=True,
        )
        for model_name in args.models:
            profile = MODEL_PROFILES[model_name]
            dtype_name = args.dtype or profile.dtype
            dtype = DTYPES[dtype_name]
            print(
                f"{model_name}: hidden_size={profile.hidden_size} "
                f"Q_heads={profile.query_heads} K_heads={profile.key_value_heads} "
                f"head_dim={profile.head_dim} eps={profile.rms_norm_eps:g} "
                f"rope_theta={profile.rope_theta:g} dtype={dtype_name}",
                flush=True,
            )
            for sequence in args.seq_lens:
                if "rms_norm" in selected:
                    input = torch.randn(
                        args.batch_size, sequence, profile.hidden_size, device=device, dtype=dtype,
                    )
                    weight = torch.randn(profile.hidden_size, device=device, dtype=dtype)
                    benchmark_case(
                        "rms_norm", (input, weight, profile.rms_norm_eps), args.backend,
                        device, args.warmup, args.repeat, model_name,
                    )
                if "rope" in selected:
                    # Match this project's model forward path, including table dtype.
                    # The helper implements the project's current RoPE frequencies;
                    # config.json rope_scaling is not applied by this framework.
                    sin, cos = generate_sin_and_cos_tables(
                        sequence, profile.head_dim, profile.rope_theta, dtype, device,
                    )
                    for role, heads in (("Q", profile.query_heads), ("K", profile.key_value_heads)):
                        input = torch.randn(
                            args.batch_size, sequence, heads, profile.head_dim, device=device, dtype=dtype,
                        )
                        benchmark_case(
                            "rope", (input, sin, cos), args.backend,
                            device, args.warmup, args.repeat, model_name, role,
                        )


if __name__ == "__main__":
    main()
