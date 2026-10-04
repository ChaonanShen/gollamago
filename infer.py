import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch
from transformers import AutoTokenizer

import backends


def create_parser():
    parser = argparse.ArgumentParser(description="Text generation with a Llama model.")

    parser.add_argument("--model", type=str, required=True, help="Path to the model.")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="List of prompts for text generation.",
    )
    parser.add_argument(
        "--batch-size", type=int,
        help="Repeat a single prompt to this batch size; multiple prompts must match this count.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--backend",
        choices=backends.BACKEND_NAMES,
        default="torch",
        help="Operator implementation to test.",
    )
    parser.add_argument(
        "--target",
        choices=backends.TARGET_NAMES,
        default="auto",
        help="Accelerator compilation target. Auto detects CUDA or MACA.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device to use for inference (e.g., "cuda", "cpu").',
    )
    parser.add_argument(
        "--num-warmup-iterations",
        type=int,
        default=0,
        help="For profiling. The number of warmup iterations to run before measuring performance.",
    )
    parser.add_argument(
        "--num-profiling-iterations",
        type=int,
        default=1,
        help="For profiling. The number of iterations to run for performance measurement.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Also write the complete result to this JSON file.",
    )

    return parser


def expand_prompts(prompts, batch_size):
    if batch_size is None:
        return prompts
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if len(prompts) == 1:
        return prompts * batch_size
    if len(prompts) != batch_size:
        raise ValueError("--batch-size must match the number of prompts when multiple prompts are given")
    return prompts


def configure_tokenizer(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


@torch.inference_mode()
def main(argv=None):
    parser = create_parser()

    args = parser.parse_args(argv)

    if args.num_warmup_iterations < 0:
        parser.error("--num-warmup-iterations must be non-negative")
    if args.num_profiling_iterations <= 0:
        parser.error("--num-profiling-iterations must be positive")

    model_path = args.model
    try:
        prompts = expand_prompts(args.prompts, args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    max_new_tokens = args.max_new_tokens
    device = args.device
    num_warmup_iterations = args.num_warmup_iterations
    num_profiling_iterations = args.num_profiling_iterations

    torch.manual_seed(args.seed)
    backend = backends.configure_backend(args.backend, device, args.target)

    # Optimized submissions may select a JIT target while llama is imported.
    import llama
    import operators

    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(model_path))

    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    model = llama.ModelForCausalLM.from_pretrained(model_path).to(device)

    for _ in range(num_warmup_iterations):
        model.generate(inputs.input_ids, max_new_tokens=max_new_tokens)

    backends.synchronize(backend.device)
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)

    iteration_times = []

    for _ in range(num_profiling_iterations):
        start_time = time.perf_counter()

        outputs = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens)

        backends.synchronize(backend.device)

        iteration_times.append(time.perf_counter() - start_time)

    average_time = statistics.mean(iteration_times)
    sorted_times = sorted(iteration_times)
    p90_index = max(0, math.ceil(len(sorted_times) * 0.9) - 1)
    num_input_tokens = inputs["input_ids"].size(-1)
    num_output_tokens_per_sequence = outputs.size(-1) - num_input_tokens
    num_generated_tokens = outputs.size(0) * num_output_tokens_per_sequence
    result = {
        "backend": backend.backend,
        "target": backend.target,
        "device": str(backend.device),
        "device_name": (
            torch.cuda.get_device_name(backend.device)
            if backend.device.type == "cuda"
            else platform.processor() or "cpu"
        ),
        "torch_version": torch.__version__,
        "maca_version": getattr(torch.version, "maca", None),
        "registered_operators": operators.get_registered_operators(backend.backend),
        "seed": args.seed,
        "batch_size": outputs.size(0),
        "num_input_tokens_per_sequence": num_input_tokens,
        "num_output_tokens_per_sequence": num_output_tokens_per_sequence,
        "num_generated_tokens": num_generated_tokens,
        "average_latency_ms": average_time * 1000,
        "p50_latency_ms": statistics.median(iteration_times) * 1000,
        "p90_latency_ms": sorted_times[p90_index] * 1000,
        "iteration_latency_ms": [sample * 1000 for sample in iteration_times],
        "tokens_per_second": num_generated_tokens / average_time,
        "generated_token_ids": outputs[:, num_input_tokens:].tolist(),
        "texts": tokenizer.batch_decode(outputs, skip_special_tokens=True),
    }
    if backend.device.type == "cuda":
        result["peak_memory_mib"] = torch.cuda.max_memory_allocated(backend.device) / 2**20

    encoded = json.dumps(result, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return result


if __name__ == "__main__":
    main()
