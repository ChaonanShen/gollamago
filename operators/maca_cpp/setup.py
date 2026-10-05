import os
from pathlib import Path
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent
MACA_SOURCE = ROOT / "src" / "rms_norm.maca"
OPTIONAL_MACA_SOURCES = [ROOT / "src" / "rope" / "rope.maca"]


class MXMACABuildExtension(BuildExtension):
    """Compile native .maca device code with mxcc before linking the extension."""

    def build_extensions(self):
        maca_path = Path(os.environ.get("MACA_PATH", "/opt/maca"))
        mxcc = Path(
            os.environ.get("MXCC", maca_path / "mxgpu_llvm" / "bin" / "mxcc")
        )
        if not mxcc.is_file():
            raise RuntimeError(f"MXMACA compiler not found: {mxcc}")

        object_dir = Path(self.build_temp) / "maca"
        object_dir.mkdir(parents=True, exist_ok=True)
        maca_object = object_dir / "rms_norm.o"
        subprocess.run(
            [
                str(mxcc),
                "-x",
                "maca",
                "-O3",
                "-fPIC",
                "-std=c++17",
                "-offload-arch",
                os.environ.get("MACA_ARCH", "native"),
                f"--maca-path={maca_path}",
                "-c",
                str(MACA_SOURCE),
                "-o",
                str(maca_object),
            ],
            check=True,
        )
        objects = [str(maca_object)]
        for source in OPTIONAL_MACA_SOURCES:
            if source.is_file():
                optional_object = object_dir / f"{source.stem}.o"
                subprocess.run(
                    [
                        str(mxcc), "-x", "maca", "-O3", "-fPIC", "-std=c++17",
                        "-offload-arch", os.environ.get("MACA_ARCH", "native"),
                        f"--maca-path={maca_path}", "-c", str(source), "-o", str(optional_object),
                    ],
                    check=True,
                )
                objects.append(str(optional_object))
        for extension in self.extensions:
            extension.extra_objects = [*getattr(extension, "extra_objects", []), *objects]
            # Fresh .maca objects must trigger relinking even when bindings.cpp is unchanged.
            extension.depends = [*getattr(extension, "depends", []), *objects]
        super().build_extensions()


setup(
    name="maca-kernels",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="operators.maca_cpp.maca_kernels",
            sources=[
                str(ROOT / "src" / "bindings.cpp"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": MXMACABuildExtension},
)
