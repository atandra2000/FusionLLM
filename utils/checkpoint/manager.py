# utils/checkpoint/manager.py
"""Checkpoint manager orchestrator.

This is the main CheckpointManager class that coordinates all
checkpoint operations using the sub-modules.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from utils.checkpoint.atomic import atomic_save_json, atomic_save_safetensors, atomic_save_torch
from utils.checkpoint.async_worker import AsyncCheckpointWorker
from utils.checkpoint.metadata import build_meta, load_best_val_loss, maybe_update_best
from utils.checkpoint import retention as _retention

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Save and load model checkpoints.

    Features
    --------
    • Atomic writes via temp-file + rename.
    • Sharded saves for multi-GPU (each rank saves its own partition).
    • Optional gzip compression for reduced disk I/O.
    • Saves best.safetensors / best_ema.safetensors whenever a new best
      validation loss is recorded.
    • Saves ema.safetensors alongside model weights when EMA state is provided.
    • keep_last_n() prunes old checkpoints to cap disk usage.
    • Persists best_val_loss across restarts via metadata.

    Usage
    -----
    ckpt = CheckpointManager("checkpoints/pretrain")
    ckpt.save(model, optimizer, step=1000,
              ema_state=ema.state_dict(),
              val_loss=2.34,
              extra_meta={"scheduler": sched.state_dict()})
    meta = ckpt.load(model, step=1000, device="cuda:0", optimizer=optimizer)
    latest = ckpt.latest_step()   # int or None

    Sharded (multi-GPU) usage:
    dist_ckpt = CheckpointManager("checkpoints/pretrain", sharded=True,
                                   world_size=8, rank=0)
    dist_ckpt.save_state_dict(model_state, ...)
    """

    def __init__(
        self,
        save_dir: str,
        async_mode: bool = True,
        sharded: bool = False,
        world_size: int = 1,
        rank: int = 0,
        compression: str = "none",
        checkpoint_backend: str = "safetensors",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sharded = sharded
        self.world_size = world_size
        self.rank = rank
        self.compression = compression
        self.checkpoint_backend = checkpoint_backend

        # best_val_loss is restored from the best_meta.json file on init
        self.best_val_loss: float = load_best_val_loss(self.save_dir)
        self._best_lock = threading.Lock()

        # Async checkpointing support
        self.async_mode = async_mode
        self._async_worker: AsyncCheckpointWorker | None = None
        self._async_thread: threading.Thread | None = None

        if self.async_mode:
            self._async_worker = AsyncCheckpointWorker()
            # Start the background worker thread
            self._async_thread = threading.Thread(
                target=self._async_worker._worker_loop,
                daemon=True,
                name="CheckpointAsyncWorker",
            )
            self._async_thread.start()

    def __del__(self):
        """Cleanup async worker on object deletion."""
        if self._async_worker is not None:
            self._async_worker.stop()

    def _stop_async_worker(self):
        """Stop the async worker thread (backward compat)."""
        if self._async_worker is not None:
            # Signal the worker to stop first
            if self._async_worker._shutdown is not None:
                self._async_worker._shutdown.set()
            if self._async_worker._queue is not None:
                try:
                    self._async_worker._queue.put(None, timeout=1.0)
                except:
                    pass
            if self._async_thread is not None and self._async_thread.is_alive():
                self._async_thread.join(timeout=2.0)
            self._async_thread = None
            self._async_worker.stop()

    # ──────────────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────────────

    def _step_dir(self, step: int) -> Path:
        """Directory for a given step (used in sharded mode)."""
        return self.save_dir / f"step_{step}"

    def _shard_dir(self, step: int) -> Path:
        """Per-rank directory within a step directory."""
        return self._step_dir(step) / f"rank_{self.rank}"

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        ema_state: dict | None = None,
        val_loss: float | None = None,
        extra_meta: dict | None = None,
        keep_last_n: int | None = None,
        extra_optimizers: dict[str, torch.optim.Optimizer] | None = None,
    ) -> None:
        """
        Atomically persist model weights, EMA weights, optimiser state, and metadata.

        Args:
            model:       unwrapped nn.Module (not DDP-wrapped)
            optimizer:   the primary optimiser to checkpoint (AdamW)
            step:        current training step (used as filename key)
            ema_state:   optional EMA shadow-weight state dict
            val_loss:    optional validation loss; triggers best.safetensors update
            extra_meta:  optional dict of JSON-serialisable metadata
            keep_last_n: if set, prune old checkpoints after saving
            extra_optimizers: optional name → optimiser (e.g. ``{"muon": muon_opt}``)
        """
        state = model.state_dict()
        
        # Deduplicate shared tensors (e.g., tied embeddings)
        seen_tensors = {}
        deduplicated_state = {}
        for key, tensor in state.items():
            tensor_ptr = tensor.data_ptr()
            if tensor_ptr in seen_tensors:
                continue
            seen_tensors[tensor_ptr] = key
            deduplicated_state[key] = tensor

        # ── Model weights ──────────────────────────────────────────────────
        weight_path = self.save_dir / f"model_step_{step}.safetensors"
        atomic_save_safetensors(deduplicated_state, weight_path, self.save_dir)

        # ── EMA weights ────────────────────────────────────────────────────
        if ema_state is not None:
            ema_path = self.save_dir / f"ema_step_{step}.safetensors"
            atomic_save_safetensors(ema_state, ema_path, self.save_dir)

        # ── Optimiser state ────────────────────────────────────────────────
        optim_path = self.save_dir / f"optim_step_{step}.pt"
        atomic_save_torch(optimizer.state_dict(), optim_path, self.save_dir)

        if extra_optimizers:
            for name, opt in extra_optimizers.items():
                path = self.save_dir / f"optim_{name}_step_{step}.pt"
                atomic_save_torch(opt.state_dict(), path, self.save_dir)

        # ── Metadata ───────────────────────────────────────────────────────
        meta = build_meta(step, self.best_val_loss, val_loss, extra_meta)
        meta_path = self.save_dir / f"meta_step_{step}.json"
        atomic_save_json(meta, meta_path, self.save_dir)

        self.best_val_loss = maybe_update_best(
            self.save_dir, state, ema_state, step, val_loss,
            self.best_val_loss, self._best_lock
        )

        logger.info("[checkpoint] saved step %d → %s", step, self.save_dir)

        # ── Prune old checkpoints ──────────────────────────────────────────
        if keep_last_n is not None:
            _retention.keep_last_n(self.save_dir, keep_last_n)

    def save_state_dict(
        self,
        state_dict: dict,
        optimizer: torch.optim.Optimizer,
        step: int,
        ema_state: dict | None = None,
        val_loss: float | None = None,
        extra_meta: dict | None = None,
        keep_last_n: int | None = None,
        extra_optimizer_states: dict[str, dict] | None = None,
    ) -> None:
        """
        Atomically persist an already-gathered state dict (used by FSDP path).

        Args:
            state_dict:  already-gathered model state dict
            optimizer:   the optimiser to checkpoint
            step:        current training step
            ema_state:   optional EMA shadow-weight state dict
            val_loss:    optional validation loss
            extra_meta:  optional JSON-serialisable metadata
            keep_last_n: if set, prune old checkpoints after saving
            extra_optimizer_states: optional name → pre-gathered optimizer state dict
        """
        weight_path = self.save_dir / f"model_step_{step}.safetensors"
        atomic_save_safetensors(state_dict, weight_path, self.save_dir)

        if ema_state is not None:
            ema_path = self.save_dir / f"ema_step_{step}.safetensors"
            atomic_save_safetensors(ema_state, ema_path, self.save_dir)

        optim_path = self.save_dir / f"optim_step_{step}.pt"
        atomic_save_torch(optimizer.state_dict(), optim_path, self.save_dir)

        if extra_optimizer_states:
            for name, state in extra_optimizer_states.items():
                path = self.save_dir / f"optim_{name}_step_{step}.pt"
                atomic_save_torch(state, path, self.save_dir)

        meta = build_meta(step, self.best_val_loss, val_loss, extra_meta)
        meta_path = self.save_dir / f"meta_step_{step}.json"
        atomic_save_json(meta, meta_path, self.save_dir)

        self.best_val_loss = maybe_update_best(
            self.save_dir, state_dict, ema_state, step, val_loss,
            self.best_val_loss, self._best_lock
        )

        logger.info("[checkpoint] saved state_dict step %d → %s", step, self.save_dir)

        if keep_last_n is not None:
            _retention.keep_last_n(self.save_dir, keep_last_n)

    def _save_sync_state(
        self,
        state: dict,
        optimizer: torch.optim.Optimizer,
        step: int,
        ema_state: dict | None = None,
        val_loss: float | None = None,
        extra_meta: dict | None = None,
        keep_last_n: int | None = None,
    ) -> None:
        """Fallback: write a regular (non-DCP) state dict synchronously."""
        weight_path = self.save_dir / f"model_step_{step}.safetensors"
        atomic_save_safetensors(state, weight_path, self.save_dir)
        if ema_state is not None:
            atomic_save_safetensors(ema_state, self.save_dir / f"ema_step_{step}.safetensors", self.save_dir)
        optim_path = self.save_dir / f"optim_step_{step}.pt"
        atomic_save_torch(optimizer.state_dict(), optim_path, self.save_dir)
        meta = build_meta(step, self.best_val_loss, val_loss, extra_meta)
        atomic_save_json(meta, self.save_dir / f"meta_step_{step}.json", self.save_dir)
        self.best_val_loss = maybe_update_best(
            self.save_dir, state, ema_state, step, val_loss,
            self.best_val_loss, self._best_lock
        )
        logger.info("[dcp fallback] saved step %d → %s", step, self.save_dir)
        if keep_last_n is not None:
            _retention.keep_last_n(self.save_dir, keep_last_n)

    # ──────────────────────────────────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────────────────────────────────

    def load(
        self,
        model: torch.nn.Module,
        step: int,
        device: str = "cuda",
        optimizer: torch.optim.Optimizer | None = None,
        extra_optimizers: dict[str, torch.optim.Optimizer] | None = None,
        strict: bool = True,
    ) -> dict:
        """
        Load model weights and optionally restore optimiser state.

        Returns metadata dict (includes "step", scheduler state, best_val_loss, etc.)
        """
        weight_path = self.save_dir / f"model_step_{step}.safetensors"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {weight_path}\nAvailable steps: {_retention.list_steps(self.save_dir)}"
            )

        weights = load_file(str(weight_path), device=device)
        missing, unexpected = model.load_state_dict(weights, strict=False)

        if missing:
            msg = (
                f"[checkpoint] {len(missing)} missing key(s): "
                f"{missing[:5]}{'…' if len(missing) > 5 else ''}"
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)

        if unexpected:
            msg = (
                f"[checkpoint] {len(unexpected)} unexpected key(s): "
                f"{unexpected[:5]}{'…' if len(unexpected) > 5 else ''}"
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)

        if optimizer is not None:
            optim_path = self.save_dir / f"optim_step_{step}.pt"
            if optim_path.exists():
                opt_state = torch.load(optim_path, map_location=device, weights_only=True)
                optimizer.load_state_dict(opt_state)
            else:
                logger.warning(
                    "[checkpoint] no optimiser state at %s — optimizer will start from scratch",
                    optim_path,
                )

        if extra_optimizers:
            for name, opt in extra_optimizers.items():
                path = self.save_dir / f"optim_{name}_step_{step}.pt"
                if path.exists():
                    opt_state = torch.load(path, map_location=device, weights_only=True)
                    opt.load_state_dict(opt_state)
                else:
                    logger.warning(
                        "[checkpoint] no %s optimiser state at %s",
                        name,
                        path,
                    )

        meta_path = self.save_dir / f"meta_step_{step}.json"
        meta: dict = {}
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            if "best_val_loss" in meta:
                self.best_val_loss = float(meta["best_val_loss"])
        else:
            logger.warning("[checkpoint] no metadata file at %s", meta_path)
            meta = {"step": step}

        logger.info("[checkpoint] loaded step %d from %s", step, self.save_dir)
        return meta

    def load_weights(
        self,
        step: int,
        device: str = "cuda",
    ) -> tuple:
        """
        Load the raw model weights and metadata without applying them.
        Used by the FSDP load path.

        Returns (weights_dict, meta_dict).
        """
        weight_path = self.save_dir / f"model_step_{step}.safetensors"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {weight_path}\nAvailable steps: {_retention.list_steps(self.save_dir)}"
            )

        weights = load_file(str(weight_path), device=device)

        meta_path = self.save_dir / f"meta_step_{step}.json"
        meta: dict = {}
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            if "best_val_loss" in meta:
                self.best_val_loss = float(meta["best_val_loss"])
        else:
            logger.warning("[checkpoint] no metadata file at %s", meta_path)
            meta = {"step": step}

        logger.info("[checkpoint] loaded weights step %d from %s", step, self.save_dir)
        return weights, meta

    # ──────────────────────────────────────────────────────────────────────
    # DCP-backed FSDP2 save / load
    # ──────────────────────────────────────────────────────────────────────

    def save_fsdp2_dcp(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        ema_state: dict | None = None,
        val_loss: float | None = None,
        extra_meta: dict | None = None,
        keep_last_n: int | None = None,
    ) -> None:
        """Gather FSDP2 state dicts on the calling thread, then queue to the async
        worker for a DCP-backed write.
        """
        self._execute_save_fsdp2_dcp(
            model=model,
            optimizer=optimizer,
            step=step,
            ema_state=ema_state,
            val_loss=val_loss,
            extra_meta=extra_meta,
            keep_last_n=keep_last_n,
        )

    def _execute_save_fsdp2_dcp(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        ema_state: dict | None = None,
        val_loss: float | None = None,
        extra_meta: dict | None = None,
        keep_last_n: int | None = None,
    ) -> None:
        """Internal DCP save — runs on main or async worker thread."""
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            get_optimizer_state_dict,
        )

        is_dist = self.world_size > 1 and dist.is_initialized()

        if not is_dist:
            logger.warning(
                "[dcp] distributed not initialized — falling back to safetensors save"
            )
            state = model.state_dict()
            self._save_sync_state(state, optimizer, step, ema_state, val_loss, extra_meta, keep_last_n)
            return

        step_dir = self._step_dir(step)
        if self.rank == 0:
            step_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        options = StateDictOptions(cpu_offload=True, full_state_dict=False)
        model_state = get_model_state_dict(model, options=options)
        optim_state = get_optimizer_state_dict(model, optimizer, options=options)

        dcp.save(
            {"model": model_state, "optimizer": optim_state},
            checkpoint_id=str(step_dir),
        )

        meta = build_meta(step, self.best_val_loss, val_loss, extra_meta)
        meta.update({"world_size": self.world_size, "checkpoint_backend": "dcp"})

        if self.rank == 0:
            atomic_save_json(meta, step_dir / "meta.json", self.save_dir)

        dist.barrier()

        with self._best_lock:
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self.rank == 0:
                    from utils.checkpoint.metadata import _update_best
                    _update_best(self.save_dir, model_state, ema_state, step, val_loss)

        if keep_last_n is not None and self.rank == 0:
            _retention.keep_last_n(self.save_dir, keep_last_n)

        if self.rank == 0:
            logger.info("[dcp] saved step %d → %s", step, step_dir)

    def load_fsdp2_dcp(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> dict:
        """Load FSDP2 model + optimizer state from a DCP checkpoint."""
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            get_optimizer_state_dict,
            set_model_state_dict,
            set_optimizer_state_dict,
        )

        step_dir = self._step_dir(step)
        meta_path = step_dir / "meta.json"

        is_dist = self.world_size > 1 and dist.is_initialized()

        if not is_dist:
            logger.warning("[dcp] distributed not initialized — falling back to safetensors load")
            return self.load(model, step, device="cpu", optimizer=optimizer)

        options = StateDictOptions(cpu_offload=True, full_state_dict=False)
        model_state = get_model_state_dict(model, options=options)
        optim_state = get_optimizer_state_dict(model, optimizer, options=options)

        dcp.load(
            {"model": model_state, "optimizer": optim_state},
            checkpoint_id=str(step_dir),
        )

        set_model_state_dict(model, model_state)
        set_optimizer_state_dict(model, optimizer, optim_state)

        meta: dict = {}
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            if "best_val_loss" in meta:
                self.best_val_loss = float(meta["best_val_loss"])
        else:
            logger.warning("[dcp] no metadata file at %s", meta_path)
            meta = {"step": step}

        if self.rank == 0:
            logger.info("[dcp] loaded step %d from %s", step, step_dir)
        return meta

    # ──────────────────────────────────────────────────────────────────────
    # Public wrappers for retention
    # ──────────────────────────────────────────────────────────────────────

    def latest_step(self) -> int | None:
        """Return the highest complete step number, or None."""
        return _retention.latest_step(self.save_dir)

    def list_checkpoints(self) -> list:
        """Return all complete checkpoint step numbers, sorted ascending."""
        return _retention.list_checkpoints(self.save_dir)

    def delete_checkpoint(self, step: int) -> None:
        """Remove all files for a given checkpoint step."""
        _retention.delete_checkpoint(self.save_dir, step)

    def keep_last_n(self, n: int) -> None:
        """Delete all but the `n` most recent complete checkpoints."""
        _retention.keep_last_n(self.save_dir, n)

    def _checkpoint_complete(self, step: int) -> bool:
        """Check if checkpoint for step is complete (backward compat)."""
        return _retention.checkpoint_complete(self.save_dir, step)

    def save_async(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        ema_state: dict | None = None,
        val_loss: float | None = None,
        extra_meta: dict | None = None,
        keep_last_n: int | None = None,
        callback=None,
        extra_optimizers: dict[str, torch.optim.Optimizer] | None = None,
    ) -> None:
        """Save checkpoint asynchronously (backward compat)."""
        def _save_job():
            try:
                self.save(
                    model, optimizer, step,
                    ema_state=ema_state,
                    val_loss=val_loss,
                    extra_meta=extra_meta,
                    keep_last_n=keep_last_n,
                    extra_optimizers=extra_optimizers,
                )
                if callback:
                    callback(None)
            except Exception as e:
                if callback:
                    callback(e)

        if self._async_worker:
            self._async_thread = threading.Thread(target=_save_job, daemon=True)
            self._async_thread.start()
        else:
            _save_job()
