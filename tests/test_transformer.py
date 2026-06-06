"""Unit tests for `models/transformer.py`.

Phase 0 scope (per `plan.md:0.3`):
  * schedule parser — accepted formats, edge cases.
  * `_init_weights` — std-init math, RMSNorm fill.
  * `TransformerBlock` — block construction for both MLA and Mamba slots.
  * `count_parameters` — total vs trainable accounting.

Forward tests that need a GPU or hit pre-existing torch 2.7 bool
subtract issues are marked `@pytest.mark.gpu` or `@pytest.mark.slow`.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.transformer import (
    DenseFFN,
    ParallelEmbedding,
    TransformerBlock,
    count_parameters,
    parse_schedule,
)


# ── parse_schedule ──────────────────────────────────────────────────────────
class TestParseSchedule:
    def test_mha_all_attention(self):
        assert parse_schedule(5, "mha") == [False] * 5

    def test_ssm_all_mamba(self):
        assert parse_schedule(5, "ssm") == [True] * 5

    def test_5_1_schedule(self):
        # period = 5 + 1 = 6; SSM at positions 5, 11, ...
        flags = parse_schedule(12, "5:1")
        assert flags == [False, False, False, False, False, True] * 2

    def test_6_1_schedule(self):
        # period = 7; SSM at position 6, 13, ...
        flags = parse_schedule(7, "6:1")
        assert flags == [False, False, False, False, False, False, True]

    def test_8_1_schedule(self):
        flags = parse_schedule(9, "8:1")
        assert flags == [False] * 8 + [True]

    def test_ssm_every_n(self):
        # "ssm:3" → every 3rd layer is SSM.  Fixed in Phase 2 by
        # reordering the parser branches to check `ssm:N` before
        # the generic `a:b` form.
        flags = parse_schedule(9, "ssm:3")
        # Layers 2, 5, 8 are SSM (every 3rd, 1-indexed).
        assert flags == [False, False, True, False, False, True, False, False, True]

    def test_ssm_every_n_4(self):
        flags = parse_schedule(8, "ssm:4")
        assert flags == [False, False, False, True, False, False, False, True]

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown schedule"):
            parse_schedule(4, "not-a-real-schedule")

    def test_short_n_layers_no_ssm(self):
        # 4 layers with 6:1 → no SSM, but the schedule is still accepted
        assert parse_schedule(4, "6:1") == [False] * 4


# ── DenseFFN ───────────────────────────────────────────────────────────────
class TestDenseFFN:
    def test_swiglu_has_three_linears(self):
        ffn = DenseFFN(dim=8, inter_dim=16, activation="swiglu")
        assert hasattr(ffn, "w1") and hasattr(ffn, "w2") and hasattr(ffn, "w3")
        assert ffn.w1.out_features == 16 and ffn.w1.in_features == 8

    def test_relu2_has_two_linears(self):
        ffn = DenseFFN(dim=8, inter_dim=16, activation="relu2")
        assert hasattr(ffn, "w1") and hasattr(ffn, "w2")
        assert ffn.w3 is None

    def test_swiglu_forward_shape(self):
        ffn = DenseFFN(dim=8, inter_dim=16, activation="swiglu")
        x = torch.randn(2, 4, 8)
        y = ffn(x)
        assert y.shape == x.shape

    def test_relu2_forward_shape(self):
        ffn = DenseFFN(dim=8, inter_dim=16, activation="relu2")
        x = torch.randn(2, 4, 8)
        y = ffn(x)
        assert y.shape == x.shape

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            DenseFFN(dim=8, inter_dim=16, activation="gelu")


# ── ParallelEmbedding ──────────────────────────────────────────────────────
class TestParallelEmbedding:
    # Phase 2.5: the pre-existing `ParallelEmbedding.__init__` bug
    # (left the weight uninitialised) is now fixed — the constructor
    # initialises the weight directly with N(0, 0.02).

    def test_weight_shape(self):
        emb = ParallelEmbedding(num_embeddings=16, embedding_dim=8, world_size=1, rank=0)
        assert emb.weight.shape == (16, 8)

    def test_multi_rank_vocab_shard(self):
        emb = ParallelEmbedding(num_embeddings=16, embedding_dim=8, world_size=2, rank=0)
        # Rank 0 owns [0, 7] (8 tokens)
        assert emb.part_vocab_size == 8
        assert emb.vocab_start_idx == 0

        emb1 = ParallelEmbedding(num_embeddings=16, embedding_dim=8, world_size=2, rank=1)
        assert emb1.part_vocab_size == 8
        assert emb1.vocab_start_idx == 8

    def test_weight_uses_parameter(self):
        # The weight is a registered Parameter (the init value is
        # not asserted here; see the note above).
        emb = ParallelEmbedding(num_embeddings=16, embedding_dim=8, world_size=1, rank=0)
        assert isinstance(emb.weight, torch.nn.Parameter)

    def test_weight_is_initialised(self):
        """The weight is no longer `torch.empty()` — it's init'd with
        a normal distribution.  We don't pin the exact std (μP might
        override later) but the weight must be finite and not all-zero.
        """
        emb = ParallelEmbedding(num_embeddings=16, embedding_dim=8, world_size=1, rank=0)
        assert torch.isfinite(emb.weight).all()
        assert emb.weight.abs().sum() > 0


# ── TransformerBlock construction ──────────────────────────────────────────
class TestTransformerBlock:
    def test_mla_block_constructs(self, tiny_cfg):
        # Layer index 0 in a 5:1 schedule is MLA (use_mamba=False)
        tiny_cfg["layer_schedule"] = "5:1"
        block = TransformerBlock(
            config=tiny_cfg,
            world_size=1,
            rank=0,
            layer_idx=0,
            use_checkpoint=False,
            use_mamba=False,
        )
        assert hasattr(block, "attn") and hasattr(block, "ffn")
        assert hasattr(block, "norm1") and hasattr(block, "norm2")

    def test_mamba_block_constructs(self, tiny_cfg):
        # Layer index 5 in a 5:1 schedule is SSM (use_mamba=True).
        # Phase 2.2: default ssm_type is "gdn", so the slot is a
        # GatedDeltaNet unless explicitly opted in to the legacy
        # Mamba-2 path.
        from models.gated_deltanet import GatedDeltaNet

        block = TransformerBlock(
            config=tiny_cfg,
            world_size=1,
            rank=0,
            layer_idx=5,
            use_checkpoint=False,
            use_mamba=True,
        )
        # Default SSM type is GDN (Phase 2.2).
        assert isinstance(block.attn, GatedDeltaNet)
        assert isinstance(block.ffn, DenseFFN)

    def test_mamba2_block_opt_in(self, tiny_cfg):
        # Opt-in to the legacy Mamba-2 path.
        from models.mamba import Mamba2Block

        tiny_cfg["ssm_type"] = "mamba2"
        block = TransformerBlock(
            config=tiny_cfg,
            world_size=1,
            rank=0,
            layer_idx=5,
            use_checkpoint=False,
            use_mamba=True,
        )
        assert isinstance(block.attn, Mamba2Block)

    def test_moe_layers_helper(self, tiny_cfg):
        block = TransformerBlock(
            config=tiny_cfg,
            world_size=1,
            rank=0,
            layer_idx=0,
            use_checkpoint=False,
            use_mamba=False,
        )
        # MLA block carries a DeepSeekMoE FFN
        moes = block.moe_layers()
        from models.moe import DeepSeekMoE

        assert len(moes) == 1 and isinstance(moes[0], DeepSeekMoE)

        # Mamba block does not
        mamba_block = TransformerBlock(
            config=tiny_cfg,
            world_size=1,
            rank=0,
            layer_idx=5,
            use_checkpoint=False,
            use_mamba=True,
        )
        assert mamba_block.moe_layers() == []


# ── count_parameters ────────────────────────────────────────────────────────
class TestCountParameters:
    def test_counts_all_params(self):
        lin = nn.Linear(8, 16)
        total, trainable = count_parameters(lin)
        assert total == 8 * 16 + 16
        assert trainable == total

    def test_frozen_params_excluded_from_trainable(self):
        lin = nn.Linear(8, 16)
        lin.weight.requires_grad_(False)
        total, trainable = count_parameters(lin)
        assert trainable == 16  # bias only
        assert total == 8 * 16 + 16
