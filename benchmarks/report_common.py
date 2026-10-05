"""Shared settings for the report-only B=64/128, S=512 benchmark scripts."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import warnings

import torch
import backends
import operators
from benchmarks.benchmark_operators import assert_close_chunked, TOLERANCES
from llama import generate_sin_and_cos_tables

ROOT = Path(__file__).resolve().parents[1]
BATCHES = (64, 128)
SEQUENCE = 512
DTYPE = torch.bfloat16
EPS = 1e-5
MODEL_NAME = "Llama-3.2-1B"


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--backend", choices=("tilelang", "maca_cpp", "all"), default="all")
    result.add_argument("--output-dir", type=Path, default=ROOT.parent / (ROOT.name + "-data") / "report-screenshots")
    return result


def backends_selected(name):
    return ("tilelang", "maca_cpp") if name == "all" else (name,)


@contextmanager
def reject_fallback():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="(?is).*(using torch|using the PyTorch reference)", category=RuntimeWarning)
        yield


def configure(name):
    if not torch.cuda.is_available() or not getattr(torch.version, "maca", None):
        raise RuntimeError("Run these report scripts in the project's MACA Python environment on C500.")
    logging.getLogger("tilelang").setLevel(logging.ERROR)
    active = backends.configure_backend(name, "cuda", "maca")
    if active.backend != name:
        raise RuntimeError("Requested backend was not activated: " + name)
    registered = operators.get_registered_operators(name)
    if name != "torch" and any(registered.get(op) != name for op in ("rms_norm", "rope")):
        raise RuntimeError("Both RMSNorm and RoPE must be native; Torch fallback is forbidden.")
    return active


def metadata(kind, backend):
    return {
        "kind": kind, "backend": backend, "model": MODEL_NAME,
        "batches": list(BATCHES), "sequence": SEQUENCE, "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "maca_version": getattr(torch.version, "maca", None),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("operators/tilelang_ops.py", "operators/maca_cpp/src/rope/rope.maca",
                         "operators/maca_cpp/src/rms_norm.maca", "llama.py")
        },
    }


def heading(title, backend):
    print("\n" + title)
    print(f"backend={backend}  device={torch.cuda.get_device_name()}  dtype=BF16")
    print("batches=64,128  sequence=512  native-only (no Torch fallback)")


def save(args, kind, backend, report):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{kind}-{backend}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("JSON:", path)
    return path


def cases():
    for batch in BATCHES:
        x = torch.randn(batch, SEQUENCE, 2048, device="cuda", dtype=DTYPE)
        weight = torch.randn(2048, device="cuda", dtype=DTYPE)
        yield "RMSNorm", "rms_norm", batch, (x, weight, EPS)
        del x, weight
        sin, cos = generate_sin_and_cos_tables(SEQUENCE, 64, 500000.0, DTYPE, "cuda")
        for label, heads in (("RoPE(Q)", 32), ("RoPE(K)", 8)):
            x = torch.randn(batch, SEQUENCE, heads, 64, device="cuda", dtype=DTYPE)
            yield label, "rope", batch, (x, sin, cos)
            del x


def validate(name, tensors, backend):
    before = [x.clone() if isinstance(x, torch.Tensor) else x for x in tensors]
    expected = operators.dispatch(name, *tensors, backend="torch")
    actual = operators.dispatch(name, *tensors, backend=backend)
    torch.cuda.synchronize()
    assert actual.shape == tensors[0].shape and actual.dtype == tensors[0].dtype
    assert actual.device == tensors[0].device
    rtol, atol = TOLERANCES[name][DTYPE]
    error = assert_close_chunked(actual, expected, rtol=rtol, atol=atol, max_error=True)
    for x, original in zip(tensors, before):
        if isinstance(x, torch.Tensor):
            assert_close_chunked(x, original, rtol=0, atol=0)
    return error


def special_rotations(tensors, backend):
    x = tensors[0]
    for quarter_turn in (False, True):
        sin = torch.full((SEQUENCE, 32), float(quarter_turn), device=x.device, dtype=x.dtype)
        cos = torch.full_like(sin, float(not quarter_turn))
        expected = operators.dispatch("rope", x, sin, cos, backend="torch")
        actual = operators.dispatch("rope", x, sin, cos, backend=backend)
        assert_close_chunked(actual, expected, rtol=0, atol=0)


def report_prompt(tokenizer):
    paragraph = ("This report describes how computers process information. Software programs consist of instructions and data. "
                 "Engineers measure performance carefully, compare implementations, and check that their outputs remain correct. ")
    suffix = "\nContinue this list of positive integers: 1, 2, 3, 4, 5, 6, 7, 8,"
    prefix = tokenizer.encode(paragraph * 50, add_special_tokens=False)
    for n in range(400, SEQUENCE + 1):
        text = tokenizer.decode(prefix[:n]) + suffix
        if len(tokenizer(text).input_ids) == SEQUENCE:
            return text
    raise RuntimeError("Could not construct the fixed 512-token prompt.")
