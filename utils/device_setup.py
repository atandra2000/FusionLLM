# utils/device_setup.py
"""GPU setup helpers — tuned for **8× NVIDIA A100 SXM 80GB** on RunPod.

The canonical target is a single-node 8×A100 SXM 80GB RunPod instance
running FSDP2 (torch.distributed.fsdp.fully_shard).  The ``HardwareConfig``
dataclass carries the YAML ``hardware:`` block; ``setup_training_device``
verifies the actual GPU, enables A100-friendly backends (TF32, cuDNN
autotune), and inspects the NV topology for FSDP-friendly collectives.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class HardwareConfig:
    """Runtime hardware profile (from YAML ``hardware`` section)."""

    device: str = "cuda"
    profile: str = "a100_80gb_8x"
    min_vram_gb: float = 70.0
    n_gpus: int = 8
    enable_tf32: bool = True
    enable_bf16_reduced_precision: bool = True
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    num_workers: int = 8
    val_num_workers: int = 4
    prefetch_factor: int = 4
    empty_cache_every: int = 0
    async_checkpointing: bool = True
    async_wandb: bool = True
    async_mlflow: bool = True
    use_mmap_data: bool = True
    enable_nvlink_check: bool = True


def parse_hardware_config(yaml_hw: dict | None) -> HardwareConfig:
    if not yaml_hw:
        return HardwareConfig()
    return HardwareConfig(
        device=yaml_hw.get("device", "cuda"),
        profile=yaml_hw.get("profile", "a100_80gb_8x"),
        min_vram_gb=float(yaml_hw.get("min_vram_gb", 70.0)),
        n_gpus=int(yaml_hw.get("n_gpus", 8)),
        enable_tf32=bool(yaml_hw.get("enable_tf32", True)),
        enable_bf16_reduced_precision=bool(yaml_hw.get("enable_bf16_reduced_precision", True)),
        cudnn_benchmark=bool(yaml_hw.get("cudnn_benchmark", True)),
        cudnn_deterministic=bool(yaml_hw.get("cudnn_deterministic", False)),
        num_workers=int(yaml_hw.get("num_workers", 8)),
        val_num_workers=int(yaml_hw.get("val_num_workers", 4)),
        prefetch_factor=int(yaml_hw.get("prefetch_factor", 4)),
        empty_cache_every=int(yaml_hw.get("empty_cache_every", 0)),
        async_checkpointing=bool(yaml_hw.get("async_checkpointing", True)),
        async_wandb=bool(yaml_hw.get("async_wandb", True)),
        async_mlflow=bool(yaml_hw.get("async_mlflow", True)),
        use_mmap_data=bool(yaml_hw.get("use_mmap_data", True)),
        enable_nvlink_check=bool(yaml_hw.get("enable_nvlink_check", True)),
    )


def _visible_device_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _check_nvlink_topology(devices: list[int]) -> bool:
    """Best-effort: detect NVLink/peer access between the listed GPU ids.

    Returns True if at least one pairwise P2P path exists (NVLink or
    PCIe P2P).  This is a hint for FSDP2's all-gather scheduling, not
    a hard requirement.
    """
    if len(devices) < 2:
        return True
    try:
        a, b = devices[0], devices[1]
        return bool(torch.cuda.can_device_access_peer(a, b))
    except Exception:
        return False


def setup_training_device(hw: HardwareConfig) -> torch.device:
    """
    Select GPU, enable A100-friendly backends (TF32, cuDNN autotune),
    verify the 8×A100 SXM 80GB layout, and set the local device.

    Raises:
        RuntimeError: if CUDA is unavailable, fewer GPUs are visible than
                      ``hw.n_gpus``, or any visible GPU falls below
                      ``hw.min_vram_gb``.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this project (target: 8×A100 SXM 80GB "
            "on RunPod). No GPU was detected."
        )

    # Honour explicit device index from config / env
    if "CUDA_VISIBLE_DEVICES" not in os.environ and hw.device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = hw.device.split(":")[-1]

    visible = _visible_device_count()
    if visible < hw.n_gpus:
        raise RuntimeError(
            f"Profile expects ≥ {hw.n_gpus} GPUs but only {visible} are visible. "
            "On RunPod deploy a 8×A100 SXM 80GB instance (e.g. 8xA100 80GB Secure Cloud) "
            "or override `hardware.n_gpus` in the YAML."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if hw.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    torch.backends.cudnn.benchmark = hw.cudnn_benchmark
    torch.backends.cudnn.deterministic = hw.cudnn_deterministic

    # ── Per-rank VRAM check (rank 0 prints) ───────────────────────────────
    props = torch.cuda.get_device_properties(device)
    vram_gb = props.total_memory / 1024**3
    name = props.name
    if local_rank == 0:
        print(f"[hardware] profile={hw.profile}")
        print(f"[hardware] local_rank={local_rank} device={device} ({name}, {vram_gb:.1f} GB)")
        print(f"[hardware] visible_gpus={visible} (target {hw.n_gpus})")

    if vram_gb < hw.min_vram_gb:
        raise RuntimeError(
            f"GPU {name} (rank {local_rank}) has {vram_gb:.1f} GB VRAM; this "
            f"profile expects ≥ {hw.min_vram_gb:.0f} GB per GPU. "
            "Lower micro_batch_size in configs/pretrain.yaml or pick a smaller model."
        )

    if hw.enable_nvlink_check and local_rank == 0:
        ids = list(range(min(visible, hw.n_gpus)))
        nv_ok = _check_nvlink_topology(ids)
        if not nv_ok:
            print(
                "[hardware] warning: no peer access detected between GPU 0 and "
                "GPU 1 — FSDP2 all-gathers will fall back to host-staged copies."
            )

    return device


def log_gpu_memory(device: torch.device, prefix: str = "") -> tuple[float, float]:
    """Return (allocated_gb, reserved_gb) and optionally print."""
    if device.type != "cuda":
        return 0.0, 0.0
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    if prefix:
        print(f"{prefix} GPU mem: {alloc:.2f} GB alloc / {reserved:.2f} GB reserved")
    return alloc, reserved


def maybe_empty_cache(step: int, every: int) -> None:
    if every > 0 and step > 0 and step % every == 0:
        torch.cuda.empty_cache()
