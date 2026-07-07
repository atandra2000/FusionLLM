#!/usr/bin/env python3
"""Training benchmark utility (A100 80GB optimized config)."""

import argparse
import time
import torch
from models.fusionllm import FusionLLM


def get_config() -> dict:
    """Get optimized config for A100 80GB."""
    return {
        "vocab_size": 64000,
        "max_seq_len": 4096,
        "dim": 768,
        "n_layers": 24,
        "n_heads": 12,
        "n_kv_groups": 8,
        "q_lora_rank": 192,
        "kv_lora_rank": 96,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 32,
        "v_head_dim": 64,
        "qk_norm": True,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "n_activated_experts": 2,
        "moe_inter_dim": 2048,
        "inter_dim": 2048,
        "gdn_d_state": 32,
        "gdn_d_conv": 4,
        "gdn_headdim": 32,
        "gdn_d_inner": 1024,
        "gdn_chunk_size": 64,
        "mtp_depth": 0,
        "muP": True,
        "logit_softcap": 15.0,
        "tie_embeddings": True,
        "dtype": "bf16",
        "wandb_enabled": False,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "use_compile": True,
        "compile_mode": "reduce-overhead",
    }


def create_mock_data_iter(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    """Create mock data iterator for benchmarking."""
    while True:
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = tokens.clone()
        yield tokens, targets


def benchmark(steps: int):
    """Run benchmark with optimized config."""
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type != "cuda":
        print("WARNING: CUDA not available, running on CPU (will be much slower)")
    
    print(f"=== Benchmark (A100 80GB Optimized) ===")
    print(f"Device: {device}")
    print(f"Steps: {steps}")
    print(f"Micro batch: {config['micro_batch_size']}")
    print(f"GA steps: {config['gradient_accumulation_steps']}")
    print(f"torch.compile: {config.get('use_compile', False)}")
    print()
    
    print("Building model...")
    model = FusionLLM(config).to(device)
    
    if config.get("use_compile") and device.type == "cuda":
        print("Compiling model (this may take 1-2 minutes)...")
        model = torch.compile(model, mode=config.get("compile_mode", "reduce-overhead"), fullgraph=True, dynamic=False)
        print("Compilation complete")
    
    data_iter = create_mock_data_iter(
        config["micro_batch_size"],
        config["max_seq_len"],
        config["vocab_size"],
        device,
    )
    
    print("Warmup...")
    model.train()
    for _ in range(10):
        tokens, targets = next(data_iter)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast(enabled=False):
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, config["vocab_size"]), targets.view(-1))
            loss.backward()
        model.zero_grad(set_to_none=True)
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print(f"Running {steps} steps...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    
    total_tokens = 0
    for step in range(steps):
        tokens, targets = next(data_iter)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast(enabled=False):
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, config["vocab_size"]), targets.view(-1))
            loss.backward()
        model.zero_grad(set_to_none=True)
        total_tokens += config["micro_batch_size"] * config["max_seq_len"]
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{steps}")
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tokens_per_sec = total_tokens / elapsed
    print(f"\n=== Results ===")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Elapsed: {elapsed:.2f} sec")
    print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")
    if tokens_per_sec > 0:
        estimated_days = 8.31e9 / tokens_per_sec / 86400
        print(f"Estimated training time for 8.31B tokens: {estimated_days:.2f} days")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training benchmark")
    parser.add_argument("--steps", type=int, default=100, help="Number of benchmark steps (default: 100)")
    args = parser.parse_args()
    
    benchmark(args.steps)
