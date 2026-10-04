"""Runtime implementation and accelerator-target selection."""

from __future__ import annotations

import dataclasses
import importlib
import warnings

import torch


BACKEND_NAMES = ("torch", "tilelang", "maca_cpp", "ninetoothed")
TARGET_NAMES = ("auto", "cuda", "maca")


@dataclasses.dataclass(frozen=True)
class BackendConfig:
    backend: str
    device: torch.device
    target: str | None

    @property
    def name(self) -> str:
        """Backward-compatible alias for older submissions."""
        return self.backend

    @property
    def tilelang_target(self) -> str | None:
        """Target passed to TileLang JIT kernels."""
        return self.target if self.backend == "tilelang" else None


_active_backend: BackendConfig | None = None


def _detect_hardware_target(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    return "maca" if getattr(torch.version, "maca", None) else "cuda"


def configure_backend(
    backend: str,
    device: str | torch.device,
    target: str = "auto",
) -> BackendConfig:
    """Validate and activate an operator implementation and hardware target."""
    global _active_backend

    if backend not in BACKEND_NAMES:
        warnings.warn(f"Unknown backend {backend!r}; using torch", RuntimeWarning)
        backend = "torch"
    if target not in TARGET_NAMES:
        warnings.warn(f"Unknown target {target!r}; using torch target", RuntimeWarning)
        target = "auto"

    torch_device = torch.device(device)
    if backend != "torch" and torch_device.type != "cuda":
        warnings.warn(
            f"The {backend} backend requires an accelerator; using torch on {torch_device}",
            RuntimeWarning,
        )
        backend = "torch"
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA device is unavailable; using torch on CPU", RuntimeWarning)
        backend = "torch"
        torch_device = torch.device("cpu")

    detected_target = _detect_hardware_target(torch_device)
    resolved_target = detected_target if target == "auto" else target

    if resolved_target == "maca" and not getattr(torch.version, "maca", None):
        warnings.warn("MACA PyTorch is unavailable; using torch", RuntimeWarning)
        backend = "torch"
        resolved_target = None
    if backend == "maca_cpp" and resolved_target != "maca":
        warnings.warn("maca_cpp requires the maca target; using torch", RuntimeWarning)
        backend = "torch"
        resolved_target = None

    if backend == "tilelang":
        try:
            try:
                target_utils = importlib.import_module("tilelang.backend.target")
            except ModuleNotFoundError as error:
                # MACA TileLang 0.1.9 keeps this helper under utils.
                if error.name not in ("tilelang.backend", "tilelang.backend.target"):
                    raise
                target_utils = importlib.import_module("tilelang.utils.target")
            resolved_target = str(target_utils.determine_target(resolved_target))
        except (ImportError, OSError) as error:
            warnings.warn(f"TileLang is unavailable ({error}); using torch", RuntimeWarning)
            backend = "torch"
            resolved_target = None
        except Exception as error:
            warnings.warn(
                f"TileLang target {resolved_target!r} is unavailable ({error}); using torch",
                RuntimeWarning,
            )
            backend = "torch"
            resolved_target = None

    if backend == "ninetoothed":
        try:
            importlib.import_module("ninetoothed")
        except (ImportError, OSError) as error:
            warnings.warn(
                f"NineToothed is unavailable ({error}); using torch", RuntimeWarning
            )
            backend = "torch"
            resolved_target = None

    config = BackendConfig(backend, torch_device, resolved_target)
    _active_backend = config
    return config


def get_active_backend(*, default_to_torch: bool = False) -> BackendConfig:
    """Return the active configuration, optionally creating a CPU Torch default."""
    if _active_backend is None:
        if default_to_torch:
            return BackendConfig("torch", torch.device("cpu"), None)
        raise RuntimeError("No backend has been configured")
    return _active_backend


def synchronize(device: str | torch.device) -> None:
    """Wait for queued accelerator work before taking a timing sample."""
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
