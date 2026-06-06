# models/transformer.py
"""Transformer backbone with hybrid Mamba-2 / MLA schedule.

A layer is either:

* an **MLA layer** — Multi-Head Latent Attention + DeepSeekMoE FFN
  (low-rank KV cache, GQA on top, optional sliding window).
* a **Mamba-2 layer** — Mamba-2 SSM + a small FFN (no attention, so
  constant-in-time and constant-in-memory inference).

The schedule is a string per layer index: ``"mha"`` (default) or
``"ssm"``.  Mamba-2 layers are 3-5× cheaper than MLA at inference but
slightly worse on long-context recall.  The standard 6:1 ratio
(Nemotron-H) or 8:1 ratio (Jamba) is recommended.

Schedule string
---------------
``schedule="mha"*5 + "ssm"*1`` repeated for n_layers blocks the
5:1 ratio.  Shorthand:
* ``"mha"`` (all attention)
* ``"ssm:n"`` (every n-th layer is SSM, others are MHA)
* ``"6:1"`` (Nemotron-H pattern)
* ``"8:1"`` (Jamba pattern)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mla import MultiHeadLatentAttention
from .moe import DeepSeekMoE


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.RMSNorm):
        if module.weight is not None:
            module.weight.data.fill_(1.0)


class ParallelEmbedding(nn.Module):
    """Vocab-sharded embedding. All-reduce on forward; pure embedding lookup when world_size==1."""

    def __init__(self, num_embeddings: int, embedding_dim: int, world_size: int, rank: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.world_size = world_size
        self.rank = rank

        base = num_embeddings // world_size
        remainder = num_embeddings % world_size
        self.part_vocab_size = base + (remainder if rank == world_size - 1 else 0)
        self.vocab_start_idx = rank * base

        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, embedding_dim))
        # Phase 2.5: initialise the weight directly (the standard
        # ``_init_weights`` walks ``nn.Linear``/``nn.Embedding`` and
        # does NOT fire for raw ``nn.Parameter``).
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_mask = (input >= self.vocab_start_idx) & (
            input < self.vocab_start_idx + self.part_vocab_size
        )
        input_local = input - self.vocab_start_idx
        # Use bitwise NOT for boolean mask instead of 1 - mask (not supported in newer PyTorch)
        input_local = input_local * input_mask + (self.part_vocab_size - 1) * (~input_mask)

        output = nn.functional.embedding(input_local, self.weight)
        output = output * input_mask.unsqueeze(-1).to(output.dtype)

        if self.world_size > 1:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.all_reduce(output, group=dist.group.WORLD)
        return output


class DenseFFN(nn.Module):
    """A small dense FFN, used as the FFN for Mamba-2 (SSM) layers.

    Activation is configurable:
    * ``"swiglu"`` (default) — ``W2(SiLU(W1(x)) * W3(x))`` (3 weights).
    * ``"relu2"`` (legacy) — ``W2(relu(W1(x)) ** 2)`` (2 weights).
    All linears are ``bias=False`` (Qwen 3 / Nemotron-H / OLMo 2 pattern).
    """

    def __init__(self, dim: int, inter_dim: int, activation: str = "swiglu"):
        super().__init__()
        self.activation = activation
        if activation == "swiglu":
            self.w1 = nn.Linear(dim, inter_dim, bias=False)
            self.w2 = nn.Linear(inter_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, inter_dim, bias=False)
        elif activation == "relu2":
            self.w1 = nn.Linear(dim, inter_dim, bias=False)
            self.w2 = nn.Linear(inter_dim, dim, bias=False)
            self.w3 = None
        else:
            raise ValueError(f"Unknown activation: {activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.w2(torch.relu(self.w1(x)).square())


class TransformerBlock(nn.Module):
    """One block.  Slot is either MLA + MoE, or SSM + dense FFN.

    ``ssm_type`` (config, default ``"gdn"``) chooses between the gated
    delta net (Qwen3-Next) and the legacy Mamba-2 selective scan.
    Both expose the same ``forward(x) → y`` contract.
    """

    def __init__(
        self,
        config: dict,
        world_size: int,
        rank: int,
        layer_idx: int,
        use_checkpoint: bool,
        use_mamba: bool,
        checkpoint_policy: bool = None,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.checkpoint_policy = checkpoint_policy
        self.layer_idx = layer_idx
        self.use_mamba = use_mamba
        self._attn_supports_use_cache = True  # set False for GDN/Mamba
        # Explicit Module annotations (mypy can't infer through the
        # if/elif/else that picks the slot class).
        self.attn: nn.Module
        self.ffn: nn.Module

        # Phase 2.3: MoLE every Nth FFN slot.  ``mole_every_n=0`` (or
        # missing) disables MoLE.  MoLE and MoE are *never* co-resident
        # in the same block in Phase 2 scope; if MoLE is enabled on
        # this layer index, the dense FFN slot is replaced wholesale.
        mole_every_n = int(config.get("mole_every_n", 0))
        use_mole = mole_every_n > 0 and (layer_idx % mole_every_n == mole_every_n - 1)
        self.use_mole = use_mole

        if use_mamba:
            ssm_type = config.get("ssm_type", "gdn")
            self._attn_supports_use_cache = False
            if ssm_type == "gdn":
                from .gated_deltanet import GatedDeltaNet

                self.attn = GatedDeltaNet(
                    config, layer_idx=layer_idx, world_size=world_size, rank=rank
                )
            elif ssm_type == "mamba2":
                from .mamba import Mamba2Block

                self.attn = Mamba2Block(
                    config, layer_idx=layer_idx, world_size=world_size, rank=rank
                )
            else:
                raise ValueError(f"Unknown ssm_type: {ssm_type!r} (expected 'gdn' or 'mamba2')")
            # SSM blocks keep a dense FFN (or MoLE if enabled).
            if use_mole:
                from .mole import MoLE

                self.ffn = MoLE(
                    dim=config["dim"],
                    rank=config.get("mole_rank", 32),
                    n_experts=config.get("mole_n_experts", 8),
                    top_k=config.get("mole_top_k", 1),
                    every_n=mole_every_n,
                )
            else:
                self.ffn = DenseFFN(
                    config["dim"],
                    config.get("inter_dim", 4 * config["dim"]),
                    activation=config.get("ffn_activation", "swiglu"),
                )
        else:
            self.attn = MultiHeadLatentAttention(
                config,
                layer_idx=layer_idx,
                world_size=world_size,
                rank=rank,
            )
            if use_mole:
                from .mole import MoLE

                self.ffn = MoLE(
                    dim=config["dim"],
                    rank=config.get("mole_rank", 32),
                    n_experts=config.get("mole_n_experts", 8),
                    top_k=config.get("mole_top_k", 1),
                    every_n=mole_every_n,
                )
            else:
                self.ffn = DeepSeekMoE(config, world_size=world_size, rank=rank)
        self.norm1 = nn.RMSNorm(config["dim"], eps=1e-6)
        self.norm2 = nn.RMSNorm(config["dim"], eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use checkpoint_policy if provided, otherwise use use_checkpoint
        should_checkpoint = self.checkpoint_policy if self.checkpoint_policy is not None else self.use_checkpoint
        if should_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_kwargs = {"use_cache": False} if self._attn_supports_use_cache else {}
        x = x + self.attn(self.norm1(x), **attn_kwargs)
        x = x + self.ffn(self.norm2(x))
        return x

    def moe_layers(self) -> list[DeepSeekMoE]:
        return [self.ffn] if (isinstance(self.ffn, DeepSeekMoE) and not self.use_mole) else []


def parse_schedule(n_layers: int, schedule: str) -> list[bool]:
    """Returns a list of ``use_mamba`` flags, one per layer.

    Accepts:
    * ``"mha"`` — all attention
    * ``"ssm"`` — all Mamba-2 / GDN
    * ``"6:1"`` — every 7th layer is SSM (Nemotron-H pattern)
    * ``"8:1"`` — every 9th layer is SSM (Jamba pattern)
    * ``"ssm:N"`` — every Nth layer is SSM
    """
    if schedule in ("mha", "attention", "transformer"):
        return [False] * n_layers
    if schedule in ("ssm", "mamba", "mamba2"):
        return [True] * n_layers
    # `ssm:N` must be checked before the generic `a:b` branch since
    # `schedule.startswith("ssm:")` is unambiguous.
    if schedule.startswith("ssm:"):
        every = int(schedule.split(":")[1])
        return [(i + 1) % every == 0 for i in range(n_layers)]
    if ":" in schedule:
        a, b = schedule.split(":")
        period = int(a) + int(b)
        every = int(b)
        return [(i + 1) % period == 0 for i in range(n_layers)]
    raise ValueError(f"Unknown schedule: {schedule!r}")


class Transformer(nn.Module):
    """The full backbone. ``config`` is the ``model:`` block of the YAML."""

    def __init__(
        self,
        config: dict,
        world_size: int = 1,
        rank: int = 0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.config = config

        vocab_size = config["vocab_size"]
        max_seq_len = config["max_seq_len"]
        dim = config["dim"]
        n_layers = config["n_layers"]

        # ── Tied embedding (Llama 3 / Qwen 2.5 / Phi-4-mini pattern) ───
        # embed and head share the same parameter object.  Saves
        # 2 × vocab_size × dim parameters, matches the field.
        embed = ParallelEmbedding(vocab_size, dim, world_size, rank)
        self.embed = embed
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.tie_embeddings = config.get("tie_embeddings", True)
        if self.tie_embeddings:
            self.head.weight = embed.weight

        # ── Layer schedule ──────────────────────────────────────────────
        schedule = config.get("layer_schedule", "mha")
        use_mamba_flags = parse_schedule(n_layers, schedule)

        # Layer-type-aware activation checkpointing policy
        # Checkpoint ~50% of MLA layers + all MoE/GDN layers
        self.checkpoint_policy = self._get_checkpoint_policy(config, use_mamba_flags)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    world_size,
                    rank,
                    i,
                    use_checkpoint,
                    use_mamba=use_mamba_flags[i],
                    checkpoint_policy=self.checkpoint_policy[i],
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.RMSNorm(dim, eps=1e-6)

        self.register_buffer(
            "max_seq_len_buf", torch.tensor(max_seq_len, dtype=torch.long)
        )
        # Convenience attribute (int) for asserts; mirrors the buffer.
        self.max_seq_len: int = int(max_seq_len)

        # Phase 2.6: logit softcap.  Cap value is configurable via
        # ``logit_softcap`` (default 15.0).  Set to 0 to disable.
        self._logit_cap = float(config.get("logit_softcap", 15.0))

        # Phase 2.6: optional asymmetric rescale after the head.
        # Disabled by default; opt in via ``model.asymmetric_rescale: true``.
        self._asym_rescale_enabled = bool(config.get("asymmetric_rescale", False))
        if self._asym_rescale_enabled:
            self.asym_rescale = AsymmetricRescale(dim=vocab_size)
        else:
            self.asym_rescale = None

        # Initialise the head (skipped by _init_weights for the embed
        # which is the same parameter).
        self.apply(_init_weights)

        # Phase 2.5: μP re-initialisation.  Honours the existing
        # _init_weights by overriding only the params that need it.
        if config.get("muP", True):
            from .mup import muP_init

            muP_init(self, config)

    def _get_checkpoint_policy(self, config: dict, use_mamba_flags: list[bool]) -> list[bool]:
        """Get layer-type-aware activation checkpointing policy.
        
        Checkpointing strategy:
        - MLA layers: Checkpoint based on ratio (0.0 = none, 1.0 = all)
        - MoE layers: Always checkpoint (memory-heavy)
        - GDN/SSM layers: Always checkpoint (state-space layers have large state)
        - Dense FFN: Don't checkpoint (relatively small memory footprint)
        """
        n_layers = len(use_mamba_flags)
        checkpoint_mla_ratio = config.get("checkpoint_mla_ratio", 0.5)
        
        policy = []
        for i in range(n_layers):
            if use_mamba_flags[i]:
                # SSM/GDN layer - always checkpoint
                policy.append(True)
            else:
                # MLA layer - checkpoint based on ratio
                if checkpoint_mla_ratio <= 0.0:
                    policy.append(False)
                elif checkpoint_mla_ratio >= 1.0:
                    policy.append(True)
                else:
                    # Checkpoint based on ratio using alternating pattern
                    mla_count = sum(1 for j in range(i+1) if not use_mamba_flags[j])
                    # Use ratio to determine checkpoint frequency
                    policy.append(mla_count % max(1, int(1.0 / checkpoint_mla_ratio)) == 0)
        
        return policy

    def forward(
        self, tokens: torch.Tensor, start_pos: int = 0, use_cache: bool = False
    ) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        assert seqlen <= self.max_seq_len, f"sequence too long ({seqlen} > {self.max_seq_len})"
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        # Phase 2.6: logit softcap (always on unless explicitly disabled
        # by setting logit_softcap: 0 in config).
        if self._logit_cap > 0:
            logits = softcap_15(logits, cap=self._logit_cap)
        # Asymmetric rescale (gated by config flag, default off).
        if getattr(self, "_asym_rescale_enabled", False):
            logits = self.asym_rescale(logits)
        return logits

    def forward_with_hidden(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward that also returns the pre-head hidden state.  Used by MTP."""
        bsz, seqlen = tokens.shape
        assert seqlen <= self.max_seq_len
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        # Phase 2.6: logit softcap (consistent with `forward`).
        if getattr(self, "_logit_cap", 0.0) > 0:
            logits = softcap_15(logits, cap=self._logit_cap)
        if getattr(self, "_asym_rescale_enabled", False):
            logits = self.asym_rescale(logits)
        return logits, x

    def moe_layers(self) -> list[DeepSeekMoE]:
        out: list[DeepSeekMoE] = []
        for layer in self.layers:
            out.extend(layer.moe_layers())
        return out

    def compile_for_inference(
        self,
        mode: str = "max-autotune",
        dynamic: bool = True,
    ) -> "Transformer":
        """Compile the model for optimized inference.
        
        Args:
            mode: Compilation mode ('max-autotune' or 'reduce-overhead')
            dynamic: Enable dynamic shapes
            
        Returns:
            Self for chaining
        """
        if not hasattr(torch, 'compile'):
            print("Warning: torch.compile not available")
            return self
        
        # Compile the forward method
        self.forward = torch.compile(
            self.forward,
            mode=mode,
            dynamic=dynamic,
        )
        return self
    
    def get_compiled_submodules(self) -> dict[str, nn.Module]:
        """Get submodules suitable for compilation.
        
        Returns:
            Dictionary of module names to compile
        """
        compile_targets = {}
        skip_patterns = {'embed', 'head', 'norm'}
        
        for name, module in self.named_modules():
            # Only compile leaf modules with parameters
            if len(list(module.children())) > 0:
                continue
            if sum(p.numel() for p in module.parameters()) == 0:
                continue
            # Skip certain module types
            if any(pattern in name.lower() for pattern in skip_patterns):
                continue
            compile_targets[name] = module
        
        return compile_targets


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── Logit softcap + asymmetric rescale (Phase 2.6) ────────────────────────
def softcap_15(logits: torch.Tensor, cap: float = 15.0) -> torch.Tensor:
    """Apply the DeepSeek-V3 logit soft-cap: ``cap * tanh(logits / cap)``.

    Bounded above by ±cap; never NaN.  Used inside :class:`Transformer.forward`
    to cap the LM head's output before cross-entropy.  The default
    cap of 15.0 is the DeepSeek-V3 recipe; pass a different value
    for experimentation.
    """
    return cap * torch.tanh(logits / cap)


class AsymmetricRescale(nn.Module):
    """Per-(channel, token) learnable rescale: ``(x - μ) / (σ + ε) * s + b``.

    Zero-init ``s`` and ``b`` so the rescale is the identity at the
    start of training.  Used as a drop-in head transformer (Phase 2.6).
    """

    def __init__(self, dim: int = 1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Per-dim learnable scale and bias.  ``dim`` is the feature
        # dim along which mean / std are computed.
        self.scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True, unbiased=False)
        normed = (x - mu) / (sigma + self.eps)
        # 1 + scale, so scale=0 → identity rescale.
        return normed * (1.0 + self.scale) + self.bias
