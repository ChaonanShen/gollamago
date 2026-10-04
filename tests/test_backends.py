from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import backends
import infer


def test_torch_backend_accepts_cpu():
    config = backends.configure_backend("torch", "cpu")

    assert config.backend == "torch"
    assert config.name == "torch"
    assert config.device == torch.device("cpu")
    assert config.target is None
    assert config.tilelang_target is None


@pytest.mark.parametrize("legacy_target_path", [False, True], ids=["current", "maca-0.1.9"])
def test_tilelang_backend_resolves_requested_target(legacy_target_path):
    target_utils = SimpleNamespace(determine_target=mock.Mock(return_value="maca"))

    def import_target(module):
        if legacy_target_path and module == "tilelang.backend.target":
            raise ModuleNotFoundError("No module named tilelang.backend", name="tilelang.backend")
        return target_utils

    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.version, "maca", "3.0", create=True),
        mock.patch.object(backends.importlib, "import_module", side_effect=import_target) as import_module,
    ):
        config = backends.configure_backend("tilelang", "cuda:0", "maca")

    expected_imports = [mock.call("tilelang.backend.target")]
    if legacy_target_path:
        expected_imports.append(mock.call("tilelang.utils.target"))
    assert import_module.call_args_list == expected_imports
    target_utils.determine_target.assert_called_once_with("maca")
    assert config.target == "maca"
    assert config.tilelang_target == "maca"


def test_tilelang_missing_dependency_does_not_try_legacy_target_path():
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.version, "maca", "3.0", create=True),
        mock.patch.object(
            backends.importlib, "import_module",
            side_effect=ModuleNotFoundError("No module named missing_dependency", name="missing_dependency"),
        ) as import_module,
        pytest.warns(RuntimeWarning, match="TileLang is unavailable.*using torch"),
    ):
        config = backends.configure_backend("tilelang", "cuda", "maca")

    assert config.backend == "torch"
    import_module.assert_called_once_with("tilelang.backend.target")


@pytest.mark.parametrize("name", ["tilelang", "maca_cpp", "ninetoothed"])
def test_accelerator_backends_fall_back_to_torch_on_cpu(name):
    with pytest.warns(RuntimeWarning, match="using torch"):
        config = backends.configure_backend(name, "cpu")
    assert config.backend == "torch"


def test_maca_target_falls_back_without_maca_pytorch():
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.version, "maca", None, create=True),
        pytest.warns(RuntimeWarning, match="using torch"),
    ):
        config = backends.configure_backend("tilelang", "cuda", "maca")
    assert config.backend == "torch"


def test_maca_cpp_falls_back_for_wrong_target():
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.version, "maca", "3.0", create=True),
        pytest.warns(RuntimeWarning, match="using torch"),
    ):
        config = backends.configure_backend("maca_cpp", "cuda", "cuda")
    assert config.backend == "torch"


def test_synchronize_handles_indexed_cuda_device():
    with mock.patch.object(torch.cuda, "synchronize") as synchronize:
        backends.synchronize("cuda:1")

    synchronize.assert_called_once_with(torch.device("cuda:1"))


def test_cli_exposes_all_backends_and_targets():
    parser = infer.create_parser()
    common = ["--model", "model", "--prompts", "hello"]

    for backend in backends.BACKEND_NAMES:
        assert parser.parse_args([*common, "--backend", backend]).backend == backend
    for target in backends.TARGET_NAMES:
        assert parser.parse_args([*common, "--target", target]).target == target


def test_configure_tokenizer_uses_eos_for_left_padding():
    tokenizer = SimpleNamespace(
        pad_token_id=None, pad_token=None, eos_token="<eos>", padding_side="right"
    )

    assert infer.configure_tokenizer(tokenizer) is tokenizer
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.padding_side == "left"


def test_main_runs_in_inference_mode():
    assert hasattr(infer.main, "__wrapped__")
