# Graph Report - .  (2026-06-09)

## Corpus Check
- Corpus is ~43,201 words - fits in a single context window. You may not need a graph.

## Summary
- 948 nodes · 1735 edges · 64 communities (62 shown, 2 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 171 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Training Pipeline|Training Pipeline]]
- [[_COMMUNITY_Data Preparation|Data Preparation]]
- [[_COMMUNITY_NCCL Profiler|NCCL Profiler]]
- [[_COMMUNITY_Deduplication|Deduplication]]
- [[_COMMUNITY_Training Benchmarks|Training Benchmarks]]
- [[_COMMUNITY_Numerical Health|Numerical Health]]
- [[_COMMUNITY_Multi-Token Prediction|Multi-Token Prediction]]
- [[_COMMUNITY_Loss Functions|Loss Functions]]
- [[_COMMUNITY_Delta Rule|Delta Rule]]
- [[_COMMUNITY_Async Data Loader|Async Data Loader]]
- [[_COMMUNITY_Transformer Blocks|Transformer Blocks]]
- [[_COMMUNITY_DeepSeek MoE|DeepSeek MoE]]
- [[_COMMUNITY_Async Checkpoint|Async Checkpoint]]
- [[_COMMUNITY_Checkpoint Loading|Checkpoint Loading]]
- [[_COMMUNITY_MLA Attention|MLA Attention]]
- [[_COMMUNITY_Checkpoint Manager|Checkpoint Manager]]
- [[_COMMUNITY_NorMuon Optimizer|NorMuon Optimizer]]
- [[_COMMUNITY_Distributed Comm|Distributed Comm]]
- [[_COMMUNITY_Compile Benchmarks|Compile Benchmarks]]
- [[_COMMUNITY_Expert Dispatch|Expert Dispatch]]
- [[_COMMUNITY_Atomic Checkpoint|Atomic Checkpoint]]
- [[_COMMUNITY_FSDP Checkpoint|FSDP Checkpoint]]
- [[_COMMUNITY_Mamba Blocks|Mamba Blocks]]
- [[_COMMUNITY_Logging|Logging]]
- [[_COMMUNITY_Checkpoint Retention|Checkpoint Retention]]
- [[_COMMUNITY_Curriculum|Curriculum]]
- [[_COMMUNITY_CE Softcap|CE Softcap]]
- [[_COMMUNITY_Scheduler & Setup|Scheduler & Setup]]
- [[_COMMUNITY_Async Sharding|Async Sharding]]
- [[_COMMUNITY_Curriculum Shards|Curriculum Shards]]
- [[_COMMUNITY_Scheduling|Scheduling]]
- [[_COMMUNITY_Compilation|Compilation]]
- [[_COMMUNITY_Hardware Config|Hardware Config]]
- [[_COMMUNITY_MoE Benchmarks|MoE Benchmarks]]
- [[_COMMUNITY_Activation Monitor|Activation Monitor]]
- [[_COMMUNITY_Linear ReLU²|Linear ReLU²]]
- [[_COMMUNITY_Expert Forward|Expert Forward]]
- [[_COMMUNITY_Routing Gate|Routing Gate]]
- [[_COMMUNITY_RoPE|RoPE]]
- [[_COMMUNITY_Script Configs 1|Script Configs 1]]
- [[_COMMUNITY_Script Configs 2|Script Configs 2]]
- [[_COMMUNITY_Health Monitor|Health Monitor]]
- [[_COMMUNITY_Distributed Wrap|Distributed Wrap]]
- [[_COMMUNITY_Tensor Validation|Tensor Validation]]
- [[_COMMUNITY_MoE Routing|MoE Routing]]
- [[_COMMUNITY_Shard Index|Shard Index]]
- [[_COMMUNITY_Flash Attention|Flash Attention]]
- [[_COMMUNITY_Parallel Embedding|Parallel Embedding]]
- [[_COMMUNITY_Config Bundle|Config Bundle]]
- [[_COMMUNITY_Best Model Tracker|Best Model Tracker]]
- [[_COMMUNITY_MoLE Model|MoLE Model]]
- [[_COMMUNITY_Runs Logger|Runs Logger]]
- [[_COMMUNITY_muP|muP]]
- [[_COMMUNITY_Health Checks|Health Checks]]
- [[_COMMUNITY_Shard Sampler|Shard Sampler]]
- [[_COMMUNITY_Smoke Script|Smoke Script]]
- [[_COMMUNITY_Cautious Optimizer|Cautious Optimizer]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Grouped GEMM|Grouped GEMM]]

## God Nodes (most connected - your core abstractions)
1. `ConfigBundle` - 39 edges
2. `DeepSeekMoE` - 28 edges
3. `CheckpointManager` - 26 edges
4. `Pretrainer` - 25 edges
5. `MultiHeadLatentAttention` - 24 edges
6. `NumericalHealthMonitor` - 24 edges
7. `Muon` - 24 edges
8. `AsyncShardLoader` - 23 edges
9. `Transformer` - 23 edges
10. `CautiousAdamW` - 18 edges

## Surprising Connections (you probably didn't know these)
- `Tensor` --uses--> `DeepSeekMoE`  [INFERRED]
  benchmarks/benchmark_moe_vectorized.py → models/moe/moe.py
- `Module` --uses--> `Transformer`  [INFERRED]
  benchmarks/benchmark_compile.py → models/transformer.py
- `Tensor` --uses--> `Transformer`  [INFERRED]
  benchmarks/benchmark_compile.py → models/transformer.py
- `dtype` --uses--> `DeepSeekMoE`  [INFERRED]
  benchmarks/benchmark_moe.py → models/moe/moe.py
- `DeepSeekMoE` --uses--> `DeepSeekMoE`  [INFERRED]
  benchmarks/benchmark_moe_vectorized.py → models/moe/moe.py

## Import Cycles
- None detected.

## Communities (64 total, 2 thin omitted)

### Community 0 - "Training Pipeline"
Cohesion: 0.07
Nodes (42): Dataset, make_synthetic_loader(), Any, Tensor, Phase 0 eval stub — perplexity on a token loader.  Why a stub? ----------- Phase, Yield ``(tokens, targets)`` batches of random tokens.      Deterministic: the sa, Compute mean cross-entropy and perplexity over ``loader``.      Args:         mo, run_perplexity() (+34 more)

### Community 1 - "Data Preparation"
Cohesion: 0.08
Nodes (43): __getattr__(), Data package — see submodules for the actual surface.  * ``data.dedup``        —, PEP 562 lazy attribute access — keeps top-level imports cheap., collect(), _collect_cosmopedia(), _collect_finemath(), _collect_fineweb2(), _collect_fineweb_edu() (+35 more)

### Community 2 - "NCCL Profiler"
Cohesion: 0.11
Nodes (22): check_nccl_available(), get_nccl_version(), NCCLProfileConfig, NCCLProfiler, NCCLProfileResult, profile_communication(), Module, Tensor (+14 more)

### Community 3 - "Deduplication"
Cohesion: 0.11
Nodes (23): deduplicate_docs(), exact_prefix_dedup(), _jaccard(), LSHIndex, md5_fallback(), md5_fallback_key(), MinHashHasher, _normalize() (+15 more)

### Community 4 - "Training Benchmarks"
Cohesion: 0.09
Nodes (21): benchmark_memory_usage(), benchmark_training_step(), dtype, Module, Optimizer, Benchmark a full training step (forward + backward + optimizer step).          A, Benchmark memory usage during training., estimate_model_memory() (+13 more)

### Community 5 - "Numerical Health"
Cohesion: 0.15
Nodes (24): AdamW, CautiousAdamW, GradScaler, NumericalHealthMonitor, NumericalHealthMonitor, Get current statistics., Reset all statistics., Monitors numerical health during training. (+16 more)

### Community 6 - "Multi-Token Prediction"
Cohesion: 0.11
Nodes (16): Linear, mtp_loss_weight_schedule(), MTPBlock, MTPModule, MultiTokenPrediction, device, Module, Tensor (+8 more)

### Community 7 - "Loss Functions"
Cohesion: 0.12
Nodes (18): compute_loss(), FusionLLMLoss, LossConfig, MoELoadBalancingLoss, MTPLoss, Tensor, Multi-Token Prediction auxiliary loss., Compute MTP auxiliary loss.                  Args:             mtp_logits: List (+10 more)

### Community 8 - "Delta Rule"
Cohesion: 0.10
Nodes (21): benchmark_delta_rule(), benchmark_delta_rule_vs_sequential(), main(), dtype, Benchmark GatedDeltaNet delta-rule implementation.          Args:         seqlen, Compare chunked vs sequential delta-rule implementations.          Note: Sequent, _autotune_configs(), chunked_delta_rule() (+13 more)

### Community 9 - "Async Data Loader"
Cohesion: 0.15
Nodes (10): AsyncShardLoader, ndarray, Tensor, Two-stage async loader over the sharded mmap corpus.      Args:         manifest, Start the async worker thread (no-op in sync mode)., Signal the async worker to exit and wait for it., Synchronous iteration: read one micro-batch, yield, repeat., Async iteration: pull pre-paged micro-batches from the queue. (+2 more)

### Community 10 - "Transformer Blocks"
Cohesion: 0.13
Nodes (10): GatedDeltaNet, One GDN block.  Drop-in for the attention slot in a layer.      Config keys (all, AsymmetricRescale, DenseFFN, DeepSeekMoE, One block.  Slot is either MLA + MoE, or SSM + dense FFN.      ``ssm_type`` (con, Get layer-type-aware activation checkpointing policy.                  Checkpoin, Per-(channel, token) learnable rescale: ``(x - μ) / (σ + ε) * s + b``.      Zero (+2 more)

### Community 11 - "DeepSeek MoE"
Cohesion: 0.14
Nodes (11): Tensor, DeepSeekMoE, Single expert forward pass (SwiGLU or ReLU²)., Compute all shared expert outputs and sum them., Args:             x: (T, dim) — flattened token representations         Returns:, DeepSeekMoE with shared experts and aux-loss-free load balancing.      Expert pa, Build (T*topk, E) one-hot assignment matrix weighted by routing scores., Router z-loss from the gate's cached pre-sigmoid logits. (+3 more)

### Community 12 - "Async Checkpoint"
Cohesion: 0.12
Nodes (10): AsyncCheckpointWorker, Check if the worker thread is running., Background thread for async checkpoint operations., Start background thread., Stop background thread and wait for pending operations., Background thread that processes async checkpoint requests., Submit an operation to the async worker.                  Args:             oper, Directory for a given step (used in sharded mode). (+2 more)

### Community 13 - "Checkpoint Loading"
Cohesion: 0.17
Nodes (17): CheckpointManager, find_latest_checkpoint(), load_checkpoint(), ConfigBundle, Module, Muon, Optimizer, Load a checkpoint.      Args:         step: Checkpoint step to load         cfg: (+9 more)

### Community 14 - "MLA Attention"
Cohesion: 0.18
Nodes (9): MultiHeadLatentAttention, device, dtype, Tensor, Compose (sliding window + causal) mask, optionally combine with         an exter, Per-layer rotary embedding table with YaRN scaling and grow-on-demand.      Args, RotaryEmbedding, bench_mla() (+1 more)

### Community 15 - "Checkpoint Manager"
Cohesion: 0.12
Nodes (9): CheckpointManager, Stop the async worker thread (backward compat)., Save and load model checkpoints.      Features     --------     • Atomic writes, Load the raw model weights and metadata without applying them.         Used by t, Return the highest complete step number, or None., Return all complete checkpoint step numbers, sorted ascending., Remove all files for a given checkpoint step., Check if checkpoint for step is complete (backward compat). (+1 more)

### Community 16 - "NorMuon Optimizer"
Cohesion: 0.14
Nodes (14): NorMuon, Optimizer, NorMuon, Module, NorMuon — orthogonalized Adam with per-row RMS for matrix params.      Args:, Get summary of optimizer configuration., Validate NorMuon configuration and return warnings.          Args:         lr: L, Validate parameter groups for NorMuon.          Args:         param_groups: List (+6 more)

### Community 17 - "Distributed Comm"
Cohesion: 0.12
Nodes (12): all_gather(), all_to_all_single(), CommProfile, NCCLProfiler, Tensor, Print profiling summary., All-gather a tensor from all ranks and return a list., All-to-all single tensor operation for expert parallelism.          Args: (+4 more)

### Community 18 - "Compile Benchmarks"
Cohesion: 0.16
Nodes (14): benchmark_compile_performance(), benchmark_forward(), create_test_config(), main(), Module, Tensor, Benchmark torch.compile performance.          Args:         dim: Model dimension, Run compile benchmarks. (+6 more)

### Community 19 - "Expert Dispatch"
Cohesion: 0.21
Nodes (12): Tensor, all_to_all_dispatch(), All-to-all expert dispatch (DeepSeek-V3 style) - falls back to scatter-gather., Scatter-gather dispatch: iterate active experts, compute, scatter-add., Try the Triton grouped-GEMM fast-path.  Returns True on success., scatter_gather_dispatch(), try_grouped_gemm(), Tensor (+4 more)

### Community 20 - "Atomic Checkpoint"
Cohesion: 0.18
Nodes (13): atomic_save_json(), atomic_save_safetensors(), atomic_save_torch(), _json_default(), JSON serialiser for types that json.dump cannot handle natively., Write a state dict as safetensors atomically via temp+rename.          Args:, Pickle an object via torch.save atomically via temp+rename.          Args:, Write a JSON file atomically via temp+rename.          Args:         obj: Dictio (+5 more)

### Community 21 - "FSDP Checkpoint"
Cohesion: 0.20
Nodes (9): Atomically persist model weights, EMA weights, optimiser state, and metadata., Load model weights and optionally restore optimiser state.          Returns meta, Gather FSDP2 state dicts on the calling thread, then queue to the async, Internal DCP save — runs on main or async worker thread., Load FSDP2 model + optimizer state from a DCP checkpoint., Delete all but the `n` most recent complete checkpoints., Save checkpoint asynchronously (backward compat)., Module (+1 more)

### Community 22 - "Mamba Blocks"
Cohesion: 0.19
Nodes (10): Mamba2Block, Tensor, Reference implementation of the Mamba-2 SSD scan.          Computes, for each he, One Mamba-2 block.  Drop-in for the attention slot in a layer., x: (bsz, seqlen, d_model)  →  (bsz, seqlen, d_model)., count_parameters(), _init_weights(), parse_schedule() (+2 more)

### Community 23 - "Logging"
Cohesion: 0.15
Nodes (6): Any, Tensor, Log per-expert load histograms (sparse step logging)., Upload a file artefact to W&B (rank-0 only)., Logs training and validation metrics to **W&B**, with stdout as the tertiary sin, TrainerLogger

### Community 24 - "Checkpoint Retention"
Cohesion: 0.25
Nodes (13): checkpoint_complete(), delete_checkpoint(), keep_last_n(), latest_step(), list_checkpoints(), list_steps(), Delete all but the `n` most recent complete checkpoints.          best.safetenso, Return all step numbers that have checkpoint files or directories.          Args (+5 more)

### Community 25 - "Curriculum"
Cohesion: 0.20
Nodes (10): Curriculum, Curriculum, Curriculum manifest and 2-stage sampler.  The pre-training corpus is split into, Two-stage curriculum over the sharded corpus.      Args:         manifest_path:, Hot-swap to stage 2 if ``step >= switch_step`` and not already done.          Re, advance_curriculum(), init_curriculum(), ConfigBundle (+2 more)

### Community 26 - "CE Softcap"
Cohesion: 0.20
Nodes (12): ce_softcap(), _ce_softcap_fwd_kernel(), fused_ce_softcap(), constexpr, Tensor, Fused CE + softcap via Triton (fast path).          Falls back to the pure-PyTor, Fused CE + softcap (pure-PyTorch fallback)., Apply the tanh-based logit softcap in-place.      ``out = softcap_value * tanh(l (+4 more)

### Community 27 - "Scheduler & Setup"
Cohesion: 0.24
Nodes (7): Warmup-Stable-Decay scheduler.      Args:         optimizer: wrapped optimizer(s, WSDScheduler, Initialize the distributed process group.      Returns:         (world_size, ran, setup_distributed(), get_logger(), init_logging(), Module-level initialiser (called by the trainer once).

### Community 28 - "Async Sharding"
Cohesion: 0.23
Nodes (9): load_manifest(), open_shard(), Path, Two-stage async sharded loader.  Powers the data path of the pre-training loop., Read a ``shards/manifest.jsonl`` and return the list of rows., CPU worker that fills the pinned-memory buffer.          One micro-batch = one s, Memory-map a shard's data section as a numpy int32 array.      Usage:         wi, read_shard_header() (+1 more)

### Community 29 - "Curriculum Shards"
Cohesion: 0.18
Nodes (6): CurriculumStage, Path, Sample one shard with replacement, weighted by stage weight., Build the Vose alias table over the per-shard weights., One stage of the curriculum.      Holds a list of ``ShardMeta`` (all of them) pl, Random

### Community 30 - "Scheduling"
Cohesion: 0.18
Nodes (8): BatchSizeSchedule, _interpolate(), Return the scheduled sequence length at *step*., Return the scheduled value at *step* for an ``initial → final`` ramp.      Args:, Schedule micro-batch size from ``initial`` to ``final``.      The schedule ramps, Return the scheduled micro-batch size at *step*., Schedule max sequence length from ``initial`` to ``final``.      This matches mo, SeqLenSchedule

### Community 31 - "Compilation"
Cohesion: 0.27
Nodes (12): compile_model(), get_default_config(), profile_compilation(), Any, Module, Compile the entire model.          Args:         model: Model to compile, Verify that compilation works correctly.          Args:         model: Model to, Get default compilation configuration.          Returns:         Dictionary with (+4 more)

### Community 32 - "Hardware Config"
Cohesion: 0.22
Nodes (11): _check_nvlink_topology(), HardwareConfig, log_gpu_memory(), parse_hardware_config(), device, Return (allocated_gb, reserved_gb) and optionally print., Runtime hardware profile (from YAML ``hardware`` section)., Best-effort: detect NVLink/peer access between the listed GPU ids.      Returns (+3 more)

### Community 33 - "MoE Benchmarks"
Cohesion: 0.29
Nodes (11): benchmark_moe_forward(), benchmark_moe_scaling(), benchmark_moe_vs_dense(), create_moe_config(), main(), DeepSeekMoE, Tensor, Benchmark MoE vs dense FFN.          Args:         dim: Model dimension (+3 more)

### Community 34 - "Activation Monitor"
Cohesion: 0.20
Nodes (7): ActivationMonitor, Module, Hook-based activation monitor for nn.Module., Register forward hooks on all layers., Create a hook function for a named module., Remove all registered hooks., Clear stored activations.

### Community 35 - "Linear ReLU²"
Cohesion: 0.22
Nodes (9): fused_linear_relu2(), linear_relu2(), _linear_relu2_fwd_kernel(), constexpr, Tensor, Fused Linear + ReLU² via Triton.          Falls back to the pure-PyTorch version, Fused Linear + ReLU² (pure-PyTorch fallback)., Linear + ReLU²  (pure-PyTorch fallback).      ``out = relu(x @ W.T + bias) ** 2` (+1 more)

### Community 36 - "Expert Forward"
Cohesion: 0.20
Nodes (6): Tensor, Expert, expert_forward_single(), Single expert FFN.      Activation is configurable per-instance:     * ``"swiglu, Single expert forward pass using raw weight tensors (SwiGLU or ReLU²)., Refresh precomputed weight stacks after optimizer step.

### Community 37 - "Routing Gate"
Cohesion: 0.24
Nodes (6): Tensor, AuxLossFreeGate, compute_routing_segments(), Args:             x: (T, dim) — flattened token representations          Returns, Shared sort/segment/capacity logic for scatter-gather routing.      Returns:, Auxiliary-Loss-Free Load Balancing Gate (DeepSeek-V3).      Routing decision

### Community 38 - "RoPE"
Cohesion: 0.24
Nodes (8): apply_rope(), device, Tensor, Ensure the table covers at least ``end_pos`` positions on ``device``.          T, Apply RoPE to ``x`` (convenience wrapper).          ``x`` is expected to have sh, Apply non-uniform YaRN frequency scaling.      The standard YaRN formula: each d, Apply rotary embeddings to a query/key tensor.      Args:         x:          ``, _yarn_freq_scaling()

### Community 39 - "Script Configs 1"
Cohesion: 0.18
Nodes (10): CUDA_VISIBLE_DEVICES, NCCL_ASYNC_ERROR_HANDLING, NCCL_BUFFSIZE, NCCL_DEBUG, NCCL_IB_DISABLE, NCCL_P2P_LEVEL, NCCL_SOCKET_IFNAME, PYTORCH_CUDA_ALLOC_CONF (+2 more)

### Community 40 - "Script Configs 2"
Cohesion: 0.18
Nodes (10): CUDA_VISIBLE_DEVICES, NCCL_ASYNC_ERROR_HANDLING, NCCL_BUFFSIZE, NCCL_DEBUG, NCCL_IB_DISABLE, NCCL_P2P_LEVEL, NCCL_SOCKET_IFNAME, PYTORCH_CUDA_ALLOC_CONF (+2 more)

### Community 41 - "Health Monitor"
Cohesion: 0.24
Nodes (9): create_health_monitor(), HealthConfig, init_health_monitor(), Configuration for numerical health checks., Create a health monitor with optional alert callback.          Args:         con, Initialize the numerical health monitor from ConfigBundle.      Args:         cf, Register a callback to save checkpoint on spike detection.      Args:         he, Register a callback to be called on anomaly detection. (+1 more)

### Community 42 - "Distributed Wrap"
Cohesion: 0.24
Nodes (9): all_to_all(), configure_reshard(), is_main_process(), dtype, Module, All-to-all list of tensors operation.          Args:         output_list: Pre-al, Apply FSDP2 (``fully_shard``) to ``model``.      The wrapping policy is *per-Tra, Configure ``reshard_after_forward`` per FSDP unit.      Keeps parameter shards r (+1 more)

### Community 43 - "Tensor Validation"
Cohesion: 0.20
Nodes (10): Module, Tensor, Check a scalar for NaN/Inf and raise RuntimeError if found.      Used for loss v, Check a tensor for NaN/Inf and raise RuntimeError if found.      Used for gradie, Check all gradients in a model for NaN/Inf.      Called before the optimizer ste, Check a loss tensor for NaN/Inf before backward.      Accepts a torch.Tensor (un, validate_gradients(), validate_loss() (+2 more)

### Community 44 - "MoE Routing"
Cohesion: 0.38
Nodes (8): benchmark_moe_routing(), benchmark_moe_vs_dense(), benchmark_routing_overhead(), main(), dtype, Compare MoE vs dense FFN computation., Benchmark routing overhead only (without expert computation)., Benchmark MoE routing computation.          Args:         dim: Model dimension

### Community 45 - "Shard Index"
Cohesion: 0.27
Nodes (5): A view over a list of shards with rank-aware offsets.      Iterating with ``__it, Return the indices that this rank will iterate this epoch., Replace the shard list (used by curriculum hot-swap)., ShardIndex, ShardMeta

### Community 46 - "Flash Attention"
Cohesion: 0.27
Nodes (8): flash_attention(), has_flash_attn(), long_short_window_mask(), device, Tensor, Return True if the ``flash_attn`` package is installed and CUDA is available., Dispatch to FA3 or pytorch SDPA.      Args:         query:  ``(bsz, n_heads, seq, Build attention masks for a long-short window schedule.      In a (period - 1):1

### Community 47 - "Parallel Embedding"
Cohesion: 0.24
Nodes (6): ParallelEmbedding, Tensor, Forward that also returns the pre-head hidden state.  Used by MTP., Apply the DeepSeek-V3 logit soft-cap: ``cap * tanh(logits / cap)``.      Bounded, Vocab-sharded embedding. All-reduce on forward; pure embedding lookup when world, softcap_15()

### Community 48 - "Config Bundle"
Cohesion: 0.27
Nodes (7): ConfigBundle, Composite configuration accepted by :class:`Pretrainer`., Optimizer, Simple warmup + cosine decay scheduler., WarmupCosineDecayScheduler, ConfigBundle, Tensor

### Community 49 - "Best Model Tracker"
Cohesion: 0.31
Nodes (8): load_best_val_loss(), maybe_update_best(), Copy current weights to best.safetensors (and best_ema.safetensors).          Ar, Restore best_val_loss from best_meta.json if it exists.          Args:         s, Update best checkpoint if val_loss improved (thread-safe).          Args:, _update_best(), Lock, Path

### Community 50 - "MoLE Model"
Cohesion: 0.25
Nodes (4): MoLE, Tensor, One MoLE bank: ``n_experts`` low-rank experts with a per-layer router.      Args, x: (bsz, seqlen, dim)  →  (bsz, seqlen, dim).

### Community 51 - "Runs Logger"
Cohesion: 0.25
Nodes (6): init_runs_csv(), Tensor, Get current activations., Initialize the runs CSV logger., Append-only CSV logger that writes eval metrics to ``runs.csv``.      Created on, RunsCsvLogger

### Community 52 - "muP"
Cohesion: 0.33
Nodes (6): _is_gate_like(), muP_init(), muP_rescale_lr(), Module, Apply μP initialisation in place.      Args:         model:  the module to initi, Rescale the base learning rate for a parameter shape.      μP says: ``lr ∝ 1 / p

### Community 53 - "Health Checks"
Cohesion: 0.29
Nodes (3): Update gradient statistics and check for anomalies.                  Args:, Check activations for NaN or Inf.                  Args:             activations, Update loss statistics and check for spikes.                  Args:

### Community 54 - "Shard Sampler"
Cohesion: 0.40
Nodes (3): Sample one shard from the active stage., Return a *list* of shards in the active stage's in-scope sources.          Used, ShardMeta

### Community 55 - "Smoke Script"
Cohesion: 0.40
Nodes (4): CUDA_VISIBLE_DEVICES, PYTORCH_CUDA_ALLOC_CONF, TOKENIZERS_PARALLELISM, run_smoke.sh script

### Community 56 - "Cautious Optimizer"
Cohesion: 0.50
Nodes (3): _cautious_mask(), Tensor, Compute sign mask for cautious weight decay.

### Community 59 - "Grouped GEMM"
Cohesion: 0.67
Nodes (3): constexpr, _grouped_gemm_kernel(), Compute grouped GEMM: ``c[e] = a[offsets[e]:offsets[e+1]] @ b[e]``.          Gri

## Knowledge Gaps
- **51 isolated node(s):** `$schema`, `plugin`, `@opencode-ai/plugin`, `dtype`, `Optimizer` (+46 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CheckpointManager` connect `Checkpoint Manager` to `Async Checkpoint`, `Checkpoint Loading`, `Atomic Checkpoint`, `FSDP Checkpoint`, `Scheduler & Setup`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Why does `DeepSeekMoE` connect `DeepSeek MoE` to `MoE Benchmarks`, `Expert Forward`, `Routing Gate`, `MoE Routing`, `Expert Dispatch`, `Mamba Blocks`?**
  _High betweenness centrality (0.136) - this node is a cross-community bridge._
- **Why does `Random` connect `Curriculum Shards` to `Curriculum`, `Async Sharding`, `Shard Index`, `Data Preparation`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Are the 29 inferred relationships involving `ConfigBundle` (e.g. with `CautiousAdamW` and `CheckpointManager`) actually correct?**
  _`ConfigBundle` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `DeepSeekMoE` (e.g. with `dtype` and `DeepSeekMoE`) actually correct?**
  _`DeepSeekMoE` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `Pretrainer` (e.g. with `ConfigBundle` and `Namespace`) actually correct?**
  _`Pretrainer` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `MultiHeadLatentAttention` (e.g. with `RotaryEmbedding` and `AsymmetricRescale`) actually correct?**
  _`MultiHeadLatentAttention` has 9 INFERRED edges - model-reasoned connections that need verification._