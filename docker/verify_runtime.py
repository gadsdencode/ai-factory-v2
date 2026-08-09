"""Verify the container's Python, dependency, and CUDA runtime."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import platform
import sqlite3
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

EXPECTED_TORCH_VERSION = "2.5.1"
EXPECTED_CUDA_VERSION = "12.4"
BYTES_PER_GIBIBYTE = 1024**3
FREE_VRAM_WARNING_RATIO = 0.8
PROFILE_MINIMUM_VRAM_GIB = {
    "8gb-safe": 7.5,
    "12gb-safe": 11.5,
}
RUNTIME_PACKAGES = {
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "datasets": "datasets",
    "einops": "einops",
    "huggingface-hub": "huggingface_hub",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "peft": "peft",
    "psutil": "psutil",
    "pyarrow": "pyarrow",
    "pydantic": "pydantic",
    "PyYAML": "yaml",
    "requests": "requests",
    "safetensors": "safetensors",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "tiktoken": "tiktoken",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "trl": "trl",
    "typer": "typer",
}
GPU_IMPORT_PACKAGES = {"bitsandbytes"}


def _build_parser() -> argparse.ArgumentParser:
    """Build the runtime verification argument parser."""
    parser = argparse.ArgumentParser(
        description="Verify AI Factory container dependencies and CUDA access."
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail when Docker has not exposed a CUDA-capable GPU.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_MINIMUM_VRAM_GIB),
        help=(
            "Validate CUDA device 0 against a hardware profile. Selecting a "
            "profile also requires GPU access."
        ),
    )
    return parser


def _public_version(version: str) -> str:
    """Return a distribution version without its local build suffix."""
    return version.split("+", maxsplit=1)[0]


def _to_gibibytes(byte_count: int) -> float:
    """Convert a byte count to gibibytes."""
    return byte_count / BYTES_PER_GIBIBYTE


def _profile_vram_error(profile: str | None, total_vram_gib: float) -> str | None:
    """Return an error when total VRAM is below a selected profile minimum."""
    if profile is None:
        return None

    minimum_vram_gib = PROFILE_MINIMUM_VRAM_GIB[profile]
    if total_vram_gib >= minimum_vram_gib:
        return None

    return (
        f"Profile '{profile}' requires at least {minimum_vram_gib:.1f} GiB "
        f"total VRAM on CUDA device 0; found {total_vram_gib:.1f} GiB."
    )


def _check_packages(require_gpu: bool) -> list[str]:
    """Check required packages and return any resulting errors."""
    errors: list[str] = []
    print("Dependencies:")
    for distribution_name, module_name in RUNTIME_PACKAGES.items():
        try:
            version = importlib.metadata.version(distribution_name)
            should_import = require_gpu or distribution_name not in GPU_IMPORT_PACKAGES
            if should_import:
                importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{distribution_name}: {exc}")
            print(f"  {distribution_name}: ERROR ({exc})")
        else:
            suffix = " (import deferred until GPU check)" if not should_import else ""
            print(f"  {distribution_name}: {version}{suffix}")
    return errors


def _check_torch(require_gpu: bool, profile: str | None = None) -> list[str]:
    """Validate the pinned PyTorch runtime and optional GPU requirement."""
    errors: list[str] = []
    try:
        import torch
    except Exception as exc:
        print("\nRuntime:")
        print(f"  Platform: {platform.platform()}")
        print(f"  Python: {platform.python_version()}")
        print(f"  PyTorch: ERROR ({exc})")
        return [f"PyTorch could not be imported: {exc}"]

    torch_version = _public_version(torch.__version__)
    cuda_version = torch.version.cuda or "unavailable"
    cuda_available = torch.cuda.is_available()

    print("\nRuntime:")
    print(f"  Platform: {platform.platform()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  PyTorch CUDA runtime: {cuda_version}")
    print(f"  CUDA available: {cuda_available}")

    if torch_version != EXPECTED_TORCH_VERSION:
        errors.append(
            f"Expected PyTorch {EXPECTED_TORCH_VERSION}, found {torch.__version__}."
        )
    if cuda_version != EXPECTED_CUDA_VERSION:
        errors.append(
            f"Expected CUDA runtime {EXPECTED_CUDA_VERSION}, found {cuda_version}."
        )

    if cuda_available:
        print(f"  GPU count: {torch.cuda.device_count()}")
        for device_index in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            free_vram_gib = _to_gibibytes(free_bytes)
            total_vram_gib = _to_gibibytes(total_bytes)
            print(f"  GPU {device_index}: {torch.cuda.get_device_name(device_index)}")
            print(
                f"    VRAM: {free_vram_gib:.1f} GiB free / "
                f"{total_vram_gib:.1f} GiB total"
            )

            if device_index == 0:
                profile_error = _profile_vram_error(profile, total_vram_gib)
                if profile_error is not None:
                    errors.append(profile_error)
                elif profile is not None:
                    print(f"    Profile: {profile} (compatible)")

            if free_vram_gib < total_vram_gib * FREE_VRAM_WARNING_RATIO:
                print(
                    "    WARNING: Less than 80% of VRAM is currently free. "
                    "Close GPU-heavy applications before training."
                )
    elif require_gpu:
        errors.append(
            "CUDA is unavailable. Confirm Docker Desktop uses the WSL2 backend "
            "and GPU support is enabled."
        )

    return errors


def _check_agent_storage() -> list[str]:
    """Verify agent directories and the persistent SQLite state mount."""
    errors: list[str] = []
    read_path = Path(os.environ.get("AGENT_ALLOWED_READ_PATH", "/data/allowed/read"))
    write_path = Path(os.environ.get("AGENT_ALLOWED_WRITE_PATH", "/data/allowed/write"))
    task_db_path = Path(os.environ.get("AGENT_TASK_DB_FILE", "tasks.db"))
    database_parent = task_db_path.parent
    if not task_db_path.is_absolute():
        database_parent = Path.cwd() / database_parent
    database_parent = database_parent.resolve()

    print("\nAgent storage:")
    print(f"  Read path: {read_path}")
    print(f"  Write path: {write_path}")
    print(f"  Task database: {task_db_path}")

    directory_checks = (
        ("read path", read_path, os.R_OK),
        ("write path", write_path, os.W_OK),
        ("task database directory", database_parent, os.W_OK),
    )
    for label, path, access_mode in directory_checks:
        if not path.is_dir():
            errors.append(f"Agent {label} does not exist: {path}.")
        elif not os.access(path, access_mode):
            errors.append(f"Agent {label} has insufficient permissions: {path}.")

    if not database_parent.is_dir() or not os.access(database_parent, os.W_OK):
        print("  SQLite write check: skipped")
        return errors

    probe_path: Path | None = None
    connection: sqlite3.Connection | None = None
    try:
        descriptor, filename = tempfile.mkstemp(
            dir=database_parent,
            prefix=".ai-factory-sqlite-check-",
            suffix=".db",
        )
        os.close(descriptor)
        probe_path = Path(filename)
        connection = sqlite3.connect(probe_path)
        connection.execute("CREATE TABLE runtime_check (id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO runtime_check DEFAULT VALUES")
        connection.commit()
        row = connection.execute("SELECT COUNT(*) FROM runtime_check").fetchone()
        if row != (1,):
            raise RuntimeError(f"Unexpected SQLite probe result: {row!r}")
    except Exception as exc:
        errors.append(f"SQLite write check failed in {database_parent}: {exc}")
        print(f"  SQLite write check: ERROR ({exc})")
    else:
        print(f"  SQLite {sqlite3.sqlite_version} write check: passed")
    finally:
        if connection is not None:
            connection.close()
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run dependency and GPU checks, returning a process exit code."""
    args = _build_parser().parse_args(argv)
    require_gpu = args.require_gpu or args.profile is not None
    errors = _check_packages(require_gpu)
    errors.extend(_check_torch(require_gpu, args.profile))
    errors.extend(_check_agent_storage())

    if errors:
        print("\nRuntime verification failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("\nRuntime verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
