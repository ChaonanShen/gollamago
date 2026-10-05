"""Fixed B=64/128, 512-token end-to-end validation and timing for screenshots."""
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoTokenizer
import infer
import llama
import operators
from benchmarks import report_common as C


def print_summary(report):
    expected = {(batch, backend) for batch in C.BATCHES for backend in C.backends_selected(report["backend"])}
    observed = {(row["batch"], row["backend"]) for row in report["results"]}
    if observed != expected or len(report["results"]) != len(expected):
        raise ValueError("Incomplete results: both requested batches must be present.")
    if not all(row.get("output_tokens_match") for row in report["results"]):
        raise ValueError("Cannot display PASS: generated tokens do not match Torch.")
    print("\n3. END-TO-END MODEL CORRECTNESS AND PERFORMANCE")
    print(f"model={report['model']}  device={report['device']}  dtype=BF16")
    print(f"batches=64,128  input=512  new_tokens={report['new_tokens']}  KV cache=OFF")
    print(f"warmup={report['warmup']} repeat={report['repeat']}; model load/JIT excluded")
    print("Timing covers complete generation; peak = Torch allocated memory.")
    print(f"{'Batch':>5} {'Backend':<10} {'Tokens':<7} {'Torch ms':>11} {'Backend ms':>11} {'Speedup':>9} {'Peak GiB':>9}")
    for r in report["results"]:
        print(f"{r['batch']:>5} {r['backend']:<10} {'PASS':<7} {r['torch_ms']:>11.2f} {r['backend_ms']:>11.2f} {r['speedup_vs_torch']:>8.3f}x {r['peak_memory_gib']:>9.2f}")
    print("PASS: every generated token matches Torch; both operators are native.")


@torch.inference_mode()
def main(argv=None):
    parser = C.parser(__doc__)
    parser.add_argument("--model", type=Path, default=C.ROOT.parent / (C.ROOT.name + "-data") / "models" / C.MODEL_NAME)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--show-json", type=Path, help="Display a saved result, without running inference.")
    args = parser.parse_args(argv)
    if args.show_json:
        report = json.loads(args.show_json.read_text())
        if report.get("kind") != "e2e" or report["batches"] != [64, 128] or report["sequence"] != 512:
            parser.error("saved JSON must be a result from this report script")
        print("SAVED RESULT DISPLAY (no inference performed):", args.show_json)
        print_summary(report)
        return report
    if min(args.new_tokens, args.warmup, args.repeat) < 1:
        parser.error("new-tokens, warmup and repeat must be positive")
    torch.manual_seed(0)
    with C.reject_fallback():
        C.configure("torch")
        report = C.metadata("e2e", args.backend)
        report.update({"new_tokens": args.new_tokens, "warmup": args.warmup, "repeat": args.repeat,
                       "kv_cache": False, "seed": 0, "results": [], "raw_cases": []})
        print("Loading model once; only generation is timed.", flush=True)
        tokenizer = infer.configure_tokenizer(AutoTokenizer.from_pretrained(str(args.model), local_files_only=True))
        prompt = C.report_prompt(tokenizer)
        report["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        inputs_one = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        assert inputs_one.shape[1] == C.SEQUENCE
        model = llama.ModelForCausalLM.from_pretrained(args.model).cuda().eval()
        report["model_path"] = str(args.model.resolve())
        report["model_config_sha256"] = hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest()
        if next(model.parameters()).dtype != C.DTYPE:
            raise ValueError("The report requires BF16 model weights.")
        for batch in C.BATCHES:
            inputs = inputs_one.repeat(batch, 1)
            baseline = None
            for backend in ("torch", *C.backends_selected(args.backend)):
                print(f"Running batch={batch}, backend={backend} ...", flush=True)
                C.configure(backend)
                gc.collect()
                torch.cuda.empty_cache()
                for _ in range(args.warmup):
                    warmup_output = model.generate(inputs, max_new_tokens=args.new_tokens)
                    del warmup_output
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                samples = []
                for _ in range(args.repeat):
                    start = time.perf_counter()
                    output = model.generate(inputs, max_new_tokens=args.new_tokens)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - start) * 1000)
                assert output.shape == (batch, C.SEQUENCE + args.new_tokens)
                tokens = output[:, C.SEQUENCE:].cpu().tolist()
                raw = {"batch": batch, "backend": backend,
                       "average_latency_ms": statistics.mean(samples),
                       "iteration_latency_ms": samples, "generated_token_ids": tokens,
                       "tokens_per_second": batch * args.new_tokens / (statistics.mean(samples) / 1000),
                       "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "registered_operators": operators.get_registered_operators(backend)}
                report["raw_cases"].append(raw)
                if backend == "torch":
                    baseline = raw
                else:
                    if tokens != baseline["generated_token_ids"]:
                        C.save(args, "e2e-failed", args.backend, report)
                        raise AssertionError(f"Generated tokens differ: batch={batch}, backend={backend}")
                    report["results"].append({
                        "batch": batch, "backend": backend, "output_tokens_match": True,
                        "torch_ms": baseline["average_latency_ms"], "backend_ms": raw["average_latency_ms"],
                        "speedup_vs_torch": baseline["average_latency_ms"] / raw["average_latency_ms"],
                        "peak_memory_gib": raw["peak_memory_gib"],
                    })
                del output
            del inputs
        print_summary(report)
        C.save(args, "e2e", args.backend, report)
        return report


if __name__ == "__main__":
    main()
