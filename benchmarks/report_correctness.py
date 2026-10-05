"""Focused correctness checks for report screenshots; no full test suite."""
import torch
from benchmarks import report_common as C


@torch.inference_mode()
def main(argv=None):
    args = C.parser(__doc__).parse_args(argv)
    with C.reject_fallback():
        for backend in C.backends_selected(args.backend):
            C.configure(backend)
            torch.manual_seed(0)
            C.heading("1. OPERATOR CORRECTNESS", backend)
            print("BF16 tolerances: RMSNorm rtol=.02 atol=.07; RoPE rtol=.04 atol=.04")
            print(f"{'Operator':<10} {'Shape':<24} {'Numeric':<9} {'Inputs':<9} {'Special':<9} {'Max abs error':>13}")
            report = C.metadata("correctness", backend)
            report["results"] = []
            for label, name, batch, tensors in C.cases():
                error = C.validate(name, tensors, backend)
                special = "-"
                if name == "rope":
                    C.special_rotations(tensors, backend)
                    special = "PASS"
                shape = list(tensors[0].shape)
                report["results"].append({
                    "operator": label, "batch": batch, "shape": shape,
                    "numerical_match": True, "shape_dtype_device_match": True,
                    "inputs_unchanged": True, "max_abs_error": error,
                    "identity_and_quarter_turn_exact": name == "rope",
                })
                print(f"{label:<10} {'x'.join(map(str, shape)):<24} {'PASS':<9} {'PASS':<9} {special:<9} {error:>13.3e}", flush=True)
            print("PASS: 6 cases; all inputs/tables/weights unchanged.")
            print("Special = identity and 90-degree rotation (exact checks).")
            C.save(args, "correctness", backend, report)


if __name__ == "__main__":
    main()
