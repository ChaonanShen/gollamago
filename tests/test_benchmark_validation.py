import pytest
import torch

from benchmarks.benchmark_operators import assert_close_chunked, create_parser


def test_chunked_validation_checks_tail_and_max_error():
    expected = torch.arange(11, dtype=torch.float32)
    actual = expected.clone()
    actual[-1] += 0.125
    assert assert_close_chunked(actual, expected, rtol=0, atol=0.125,
                                chunk_elements=4, max_error=True) == 0.125
    with pytest.raises(AssertionError):
        assert_close_chunked(actual, expected, rtol=0, atol=0, chunk_elements=4)


def test_chunked_validation_rejects_shape_and_dtype_mismatch():
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_close_chunked(torch.ones(2, 3), torch.ones(6), rtol=0, atol=0)
    with pytest.raises(AssertionError):
        assert_close_chunked(torch.ones(6).half(), torch.ones(6), rtol=0, atol=0)


def test_operator_benchmark_defaults_and_optional_models():
    parser = create_parser()
    defaults = parser.parse_args([])
    assert defaults.models == ["llama3.2-1b"]
    assert defaults.seq_lens == [128, 512, 2048, 4906]
    assert parser.parse_args(["--models", "llama3.2-3b", "llama3.1-8b"]).models == [
        "llama3.2-3b", "llama3.1-8b",
    ]
