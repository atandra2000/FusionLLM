# FusionLLM V2 Stabilization and Architecture Refactor

You are acting as a Principal Research Engineer responsible for preparing this repository for large-scale pretraining.

Your objective is NOT to maximize novelty.

Your objective is to maximize:

1. Convergence quality
2. Training stability
3. Throughput
4. Long-run reliability
5. Distributed scalability
6. Research value

while minimizing:

* architectural complexity
* optimization risk
* routing instability
* implementation fragility

Assume this model will be trained by a small team with finite compute resources.

Every change must have strong empirical justification.

Novel ideas are welcome only if they improve the probability of successful training.

---

# IMPORTANT

Before modifying any code:

1. Inspect the entire repository.
2. Read all architecture documentation.
3. Trace all training paths.
4. Identify current bottlenecks.
5. Produce a migration plan.

Do not begin coding until the architecture review is complete.

---

# PRIMARY OBJECTIVE

Transform the current repository into FusionLLM V2.

Preserve:

* MLA
* Sparse MoE
* Multi Token Prediction
* FSDP2
* μP scaling
* Muon/NorMuon optimizer strategy

Improve:

* throughput
* memory efficiency
* routing stability
* convergence
* distributed scaling

Do NOT introduce unnecessary architectural branches.

---

# ARCHITECTURE DECISIONS (MANDATORY)

These decisions are fixed unless code inspection reveals a critical issue.

---

## 1. KEEP MLA

Do not redesign MLA.

Current MLA is already one of the strongest parts of the repository.

Instead:

* profile it
* optimize it
* fuse operations
* improve kernels

Focus on:

* FlashAttention integration
* projection fusion
* memory efficiency
* KV compression efficiency

Avoid introducing Multi-Resolution MLA at this stage.

Create benchmark evidence before proposing MLA redesigns.

---

## 2. REPLACE CURRENT GATED DELTANET

Current sequential recurrence must be removed.

Any implementation containing:

for t in range(seq_len)

inside the critical forward path is unacceptable.

Replace current state-space implementation with a chunked parallel architecture inspired by:

* Mamba-2
* Kimi Delta Attention
* chunked scan systems

Requirements:

* parallel scan
* Triton-friendly implementation
* flash-friendly implementation
* no Python recurrence in training path
* scalable to 4k, 8k and 16k context

The replacement should occupy roughly:

10–20% of network depth

and should complement MLA rather than compete with it.

---

## 3. SIMPLIFY MoE

Remove proposals involving:

* hierarchical routers
* domain routers
* multi-level routing

The repository should use:

Top-4 routing
+
shared experts
+
residual dense path
+
load balancing

Target architecture:

Attention
↓
Dense FFN
↓
Sparse Experts
↓
Residual Merge

Reason:

This improves stability while preserving sparse capacity.

---

## 4. REMOVE HARDCODED EXPERT SPECIALIZATION

Do not implement:

* reasoning experts
* code experts
* retrieval experts
* compression experts

Use generic experts.

Instead create tooling that measures:

* token distribution
* specialization emergence
* routing entropy
* expert similarity

Allow specialization to emerge naturally.

---

## 5. REVISE MTP

Replace current MTP strategy with:

t+1
t+2
t+4

Do not implement t+8 prediction.

Requirements:

* adaptive loss weighting
* stable target alignment
* efficient implementation

Provide justification and benchmarks.

---

# TARGET BLOCK STRUCTURE

Preferred repeating pattern:

MLA
MLA
MLA
SSM
MLA
MoE

repeat

SSM should represent:

10–20% of total layers

not the majority.

MLA remains the dominant modeling mechanism.

---

# TRAINING STABILITY REQUIREMENTS

Create:

docs/TRAINING_STABILITY_PLAN.md

Implement the following.

---

## EMA

Add Exponential Moving Average weights.

Target:

0.999–0.9999 decay

Support:

* checkpoint save
* checkpoint restore
* evaluation using EMA

---

## DeepNorm Evaluation

Audit residual scaling.

Determine whether DeepNorm-style scaling improves stability.

Implement if beneficial.

Document reasoning.

---

## Router Warmup

Add early-training stabilization.

Examples:

* router temperature scheduling
* shared expert emphasis
* load-balancing warmup

Goal:

prevent early expert collapse.

---

## Expert Collapse Detection

Automatically track:

* expert utilization
* routing entropy
* expert load variance
* token concentration

Trigger warnings when collapse is detected.

---

## Numerical Safety Layer

Implement:

* NaN detection
* Inf detection
* gradient explosion detection
* activation explosion detection

before optimizer step.

Training should fail loudly rather than silently diverge.

---

# PERFORMANCE REFACTOR

Create:

docs/PERFORMANCE_REFACTOR.md

Priority order matters.

---

## Priority 1

Eliminate sequential recurrence.

This is the highest-impact optimization.

Estimate expected speedup.

Provide benchmarks.

---

## Priority 2

Optimize MoE routing.

Reduce:

* argsort overhead
* scatter overhead
* gather overhead
* communication overhead

Keep routing logic simple.

Optimize implementation rather than introducing new routing algorithms.

---

## Priority 3

Optimize MLA kernels.

Benchmark:

* einsum
* batched GEMM
* fused projection variants

Use measurements rather than assumptions.

---

## Priority 4

FlashAttention Integration

Ensure MLA path uses the most efficient attention backend available.

Document fallback paths.

---

# MEMORY OPTIMIZATION

Create:

docs/MEMORY_OPTIMIZATION.md

Implement:

---

## Static Buffer Reuse

Preallocate and reuse:

* routing buffers
* assignment buffers
* communication buffers

Reduce allocator pressure.

Reduce fragmentation.

---

## Activation Checkpointing Policy

Checkpoint:

* all MoE layers
* all SSM layers
* approximately 50% of MLA layers

Provide rationale.

Do not checkpoint everything blindly.

---

## Long Context Planning

Document memory behavior for:

4096
8192
16384

context lengths.

Include formulas and expected scaling.

---

# DISTRIBUTED SYSTEMS

Create:

docs/DISTRIBUTED_SCALING.md

Focus on practical scalability.

---

## Maintain FSDP2

Do not introduce:

* pipeline parallelism
* tensor parallelism
* context parallelism

unless code inspection proves necessity.

---

## Add Expert Parallelism

Design clean support for:

FSDP2 + Expert Parallel

This should be the primary scaling strategy.

---

## Recovery Validation

Implement and test:

* checkpoint restart
* optimizer restore
* EMA restore
* dataloader restore
* router restore

Training should survive interruptions.

---

# MONITORING

Create:

docs/MONITORING_GUIDE.md

Track:

loss
learning rate
gradient norm
activation norm
router entropy
attention entropy
expert utilization
expert collapse indicators
tokens/sec
GPU utilization
communication overhead

Provide dashboards if applicable.

---

# DELIVERABLES

Produce:

docs/AUDIT_REPORT.md

docs/FUSIONLLM_V2_ARCHITECTURE.md

docs/TRAINING_STABILITY_PLAN.md

docs/PERFORMANCE_REFACTOR.md

docs/MEMORY_OPTIMIZATION.md

docs/DISTRIBUTED_SCALING.md

docs/MONITORING_GUIDE.md

docs/MIGRATION_PLAN.md

---

# IMPLEMENTATION PHASES

Phase 1:
Performance and bottleneck elimination

Phase 2:
Training stability and convergence improvements

Phase 3:
Distributed scaling improvements

Phase 4:
Experimental research features

No experimental architecture work should begin until phases 1–3 are complete.

---

# SUCCESS CRITERIA

A successful outcome is NOT the most novel architecture.

A successful outcome is:

* higher throughput
* better convergence
* stable expert utilization
* lower memory usage
* cleaner distributed scaling
* successful multi-day training runs

When forced to choose between novelty and robustness, choose robustness.
