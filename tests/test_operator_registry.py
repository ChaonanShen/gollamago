import pytest
import torch

import operators
from benchmarks.compare_results import compare


def test_torch_operators_match_references():
    input = torch.randn(2, 3, 8)
    weight = torch.randn(8)

    actual = operators.dispatch("rms_norm", input, weight, 1e-5, backend="torch")

    torch.testing.assert_close(
        actual,
        input * torch.rsqrt(input.pow(2).mean(-1, keepdim=True) + 1e-5) * weight,
    )


def test_torch_rope_matches_half_rotation_reference():
    input = torch.randn(2, 3, 4, 8)
    sin = torch.randn(3, 4)
    cos = torch.randn(3, 4)
    actual = operators.get_operator("rope", "torch")(input, sin, cos)
    first, second = input[..., :4], input[..., 4:]
    expected = torch.cat(
        ((first * cos[None, :, None] - second * sin[None, :, None]),
         (first * sin[None, :, None] + second * cos[None, :, None])), dim=-1
    )
    torch.testing.assert_close(actual, expected)


def test_registered_operators_report_backend_implementation():
    assert operators.get_registered_operators("torch") == {
        "rms_norm": "torch",
        "rope": "torch",
    }


def test_missing_operator_reports_no_torch_implementation():
    with pytest.raises(operators.OperatorUnavailableError, match="No torch implementation"):
        operators.get_operator("not_implemented", "torch")


def test_unavailable_backend_falls_back_to_torch():
    with pytest.warns(RuntimeWarning, match="using torch"):
        implementation = operators.get_operator("rms_norm", "missing_backend")
    assert implementation is operators.get_operator("rms_norm", "torch")


def test_dispatch_falls_back_when_backend_operator_fails():
    import operators.registry as registry

    original_get_operator = registry.get_operator

    def failing_operator(name, backend=None):
        if backend == "tilelang":
            return lambda *args: (_ for _ in ()).throw(RuntimeError("kernel failed"))
        return original_get_operator(name, backend)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(registry, "get_operator", failing_operator)
    with pytest.warns(RuntimeWarning, match="failed.*using torch"):
        result = operators.dispatch("rms_norm", torch.randn(1, 2, 4), torch.ones(4), 1e-5, backend="tilelang")
    monkeypatch.undo()
    assert result.shape == (1, 2, 4)


def test_compare_uses_throughput_ratio():
    shared = {
        "target": "maca",
        "device_name": "test-device",
        "torch_version": "test",
        "maca_version": "test",
        "seed": 0,
        "batch_size": 2,
        "num_input_tokens_per_sequence": 8,
        "num_output_tokens_per_sequence": 4,
        "generated_token_ids": [[1, 2], [3, 4]],
        "registered_operators": {"rms_norm": "torch", "rope": "torch"},
    }
    result = compare(
        {**shared, "backend": "torch", "tokens_per_second": 100},
        {
            **shared,
            "backend": "tilelang",
            "registered_operators": {
                "rms_norm": "tilelang",
                "rope": "torch_fallback",
            },
            "tokens_per_second": 180,
        },
    )

    assert result["speedup"] == 1.8
    assert result["improvement_percent"] == pytest.approx(80)
    assert result["modified_operators"] == {
        "rms_norm": {"baseline": "torch", "candidate": "tilelang"}
    }


def test_compare_rejects_different_outputs():
    baseline = {
        "target": "maca", "device_name": "d", "torch_version": "t",
        "maca_version": "m", "seed": 0, "batch_size": 1,
        "num_input_tokens_per_sequence": 1, "num_output_tokens_per_sequence": 1,
        "generated_token_ids": [[1]], "backend": "torch", "tokens_per_second": 1,
    }
    with pytest.raises(ValueError, match="token IDs differ"):
        compare(baseline, {**baseline, "generated_token_ids": [[2]]})
