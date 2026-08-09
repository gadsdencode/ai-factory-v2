"""Regression tests for the Docker Desktop configuration assets."""

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_compose() -> dict[str, Any]:
    """Load the Compose document from the repository root."""
    compose_path = REPOSITORY_ROOT / "compose.yaml"
    return cast(
        dict[str, Any],
        yaml.safe_load(compose_path.read_text(encoding="utf-8")),
    )


def test_gpu_service_reserves_all_nvidia_gpus() -> None:
    """The primary service must request NVIDIA GPU access from Compose."""
    compose = _load_compose()
    device = compose["services"]["ai-factory"]["deploy"]["resources"]["reservations"][
        "devices"
    ][0]

    assert device == {
        "driver": "nvidia",
        "count": "all",
        "capabilities": ["gpu"],
    }
    assert "deploy" not in compose["services"]["ai-factory-check"]


def test_compose_persists_models_outputs_datasets_and_agent_state() -> None:
    """The common service must retain model and agent runtime artifacts."""
    compose = _load_compose()
    common = compose["x-ai-factory-common"]
    volumes = common["volumes"]
    mounts = {(volume["source"], volume["target"]) for volume in volumes}

    assert mounts == {
        ("huggingface-cache", "/home/ai-factory/.cache/huggingface"),
        ("./src/data", "/workspace/src/data"),
        ("./training_output", "/workspace/src/training_output"),
        ("./agent_data/read", "/data/allowed/read"),
        ("./agent_data/write", "/data/allowed/write"),
        ("./agent_data/state", "/data/state"),
    }

    read_mount = next(
        volume for volume in volumes if volume["source"] == "./agent_data/read"
    )
    assert read_mount["read_only"] is True
    assert common["environment"]["AGENT_ALLOWED_READ_PATH"] == ("/data/allowed/read")
    assert common["environment"]["AGENT_ALLOWED_WRITE_PATH"] == ("/data/allowed/write")
    assert common["environment"]["AGENT_TASK_DB_FILE"] == ("/data/state/tasks.db")
    assert not common["environment"]["AGENT_TASK_DB_FILE"].startswith(
        common["environment"]["AGENT_ALLOWED_WRITE_PATH"]
    )
    assert common["environment"]["PYTORCH_CUDA_ALLOC_CONF"] == (
        "expandable_segments:True"
    )


@pytest.mark.parametrize(
    (
        "profile_name",
        "model_name",
        "minimum_vram_gib",
        "lora_rank",
        "worker_count",
    ),
    [
        ("8gb-safe", "Qwen/Qwen3.5-4B", 7.5, 16, 1),
        ("12gb-safe", "Qwen/Qwen3.5-9B", 11.5, 32, 2),
    ],
)
def test_hardware_profiles_use_conservative_docker_defaults(
    profile_name: str,
    model_name: str,
    minimum_vram_gib: float,
    lora_rank: int,
    worker_count: int,
) -> None:
    """Each hardware profile must preserve its documented safety contract."""
    profile_path = REPOSITORY_ROOT / "src" / f"config.{profile_name}.yaml"
    profile = cast(
        dict[str, Any],
        yaml.safe_load(profile_path.read_text(encoding="utf-8")),
    )

    assert profile["model"] == {
        "name": model_name,
        "max_length": 2048,
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
        "use_linear_attention_kernels": False,
    }
    assert profile["quantization"] == {
        "enabled": True,
        "quant_type": "nf4",
        "use_double_quant": True,
    }
    assert profile["lora"]["r"] == lora_rank
    assert profile["training"]["per_device_train_batch_size"] == 1
    assert profile["training"]["per_device_eval_batch_size"] == 1
    assert profile["training"]["gradient_accumulation_steps"] == 8
    assert profile["training"]["dataloader_num_workers"] == worker_count
    assert profile["training"]["output_dir"].endswith(profile_name)
    assert profile["dpo"]["per_device_train_batch_size"] == 1
    assert profile["dpo"]["per_device_eval_batch_size"] == 1
    assert profile["dpo"]["gradient_accumulation_steps"] == 4
    assert profile["dpo"]["lora_rank"] == lora_rank

    verifier = runpy.run_path(str(REPOSITORY_ROOT / "docker" / "verify_runtime.py"))
    profile_vram_error = cast(
        Callable[[str | None, float], str | None],
        verifier["_profile_vram_error"],
    )
    assert profile_vram_error(profile_name, minimum_vram_gib) is None
    assert profile_vram_error(profile_name, minimum_vram_gib - 0.1) is not None


def test_dockerfile_pins_the_canonical_torch_cuda_stack() -> None:
    """The image and constraints must preserve the repository version contract."""
    dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
    constraints = (REPOSITORY_ROOT / "docker/constraints.txt").read_text(
        encoding="utf-8"
    )

    assert "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:" in dockerfile
    assert "torch==2.5.1" in constraints
    assert "torchvision==0.20.1" in constraints
    assert "torchaudio==2.5.1" in constraints
    assert "/data/allowed/read" in dockerfile
    assert "/data/allowed/write" in dockerfile
    assert "/data/state" in dockerfile


def test_windows_host_check_covers_wsl_docker_gpu_and_container_preflight() -> None:
    """The Windows helper must verify each Docker Desktop host dependency."""
    host_check = (REPOSITORY_ROOT / "docker" / "Test-DockerDesktopHost.ps1").read_text(
        encoding="utf-8"
    )

    assert 'ValidateSet("8gb-safe", "12gb-safe")' in host_check
    assert "wsl.exe --version" in host_check
    assert "$minimumFreeDiskGiB = 60" in host_check
    assert "nvidia-smi" in host_check
    assert 'docker info --format "{{.OSType}}"' in host_check
    assert "--require-gpu" in host_check
    assert "--profile $Profile" in host_check


def test_sqlite_preflight_uses_isolated_state_mount(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The preflight must transact in state and remove its probe database."""
    read_path = tmp_path / "read"
    write_path = tmp_path / "write"
    state_path = tmp_path / "state"
    read_path.mkdir()
    write_path.mkdir()
    state_path.mkdir()
    monkeypatch.setenv("AGENT_ALLOWED_READ_PATH", str(read_path))
    monkeypatch.setenv("AGENT_ALLOWED_WRITE_PATH", str(write_path))
    monkeypatch.setenv("AGENT_TASK_DB_FILE", str(state_path / "tasks.db"))

    verifier = runpy.run_path(str(REPOSITORY_ROOT / "docker" / "verify_runtime.py"))
    check_storage = cast(Callable[[], list[str]], verifier["_check_agent_storage"])

    assert check_storage() == []
    assert list(state_path.iterdir()) == []
