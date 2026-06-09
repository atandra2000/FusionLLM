---
type: community
cohesion: 0.07
members: 61
---

# Training Pipeline

**Cohesion:** 0.07 - loosely connected
**Members:** 61 nodes

## Members
- [[.__getitem__()]] - code - training/dataset.py
- [[.__init__()_21]] - code - training/dataset.py
- [[.__len__()_3]] - code - training/dataset.py
- [[._amp_context()]] - code - training/trainer.py
- [[._log()]] - code - training/trainer.py
- [[._maybe_eval()]] - code - training/trainer.py
- [[._update_schedules()]] - code - training/trainer.py
- [[.load_checkpoint()]] - code - training/trainer.py
- [[.save_checkpoint()]] - code - training/trainer.py
- [[.train()]] - code - training/trainer.py
- [[.train_step()]] - code - training/trainer.py
- [[Any]] - code - eval/eval_core.py
- [[Build ConfigBundle from YAML configuration and CLI arguments.]] - rationale - training/pretrain.py
- [[Checkpoint configuration.]] - rationale - training/configs.py
- [[CheckpointConfig]] - code - training/configs.py
- [[Compute mean cross-entropy and perplexity over ``loader``.      Args         mo]] - rationale - eval/eval_core.py
- [[ConfigBundle_4]] - code - training/pretrain.py
- [[Data loading configuration.]] - rationale - training/configs.py
- [[DataConfig]] - code - training/configs.py
- [[Dataset]] - code
- [[EvalConfig_1]] - code - training/validation.py
- [[EvalConfig]] - code - training/configs.py
- [[Evaluation configuration.]] - rationale - training/configs.py
- [[Evaluation package for the fusionllm project.  Phase 0 ships a single entry poin]] - rationale - eval/__init__.py
- [[FSDP2-aware pre-training loop (v2 — accepts class`ConfigBundle`).]] - rationale - training/trainer.py
- [[Logging configuration.]] - rationale - training/configs.py
- [[LoggingConfig]] - code - training/configs.py
- [[Main entrypoint for training.]] - rationale - training/pretrain.py
- [[Module_2]] - code - eval/run_lm_eval.py
- [[Module_11]] - code - training/validation.py
- [[Namespace_1]] - code - training/pretrain.py
- [[OptimConfig]] - code - training/configs.py
- [[Optimizer configuration.]] - rationale - training/configs.py
- [[Optional lm-eval-harness wrapper for eval during training (Phase 6.2).  Graceful]] - rationale - eval/run_lm_eval.py
- [[Packed pre-training dataset.]] - rationale - training/dataset.py
- [[Phase 0 eval stub — perplexity on a token loader.  Why a stub ----------- Phase]] - rationale - eval/eval_core.py
- [[PretrainDataset]] - code - training/dataset.py
- [[Pretrainer]] - code - training/trainer.py
- [[Run a set of lm-eval-harness tasks on model.      Args         model a model]] - rationale - eval/run_lm_eval.py
- [[Run evaluation if enabled and at the right step.      Args         step Curren]] - rationale - training/validation.py
- [[ScheduleConfig]] - code - training/configs.py
- [[Tear down the distributed process group (no-op if not initialised).]] - rationale - utils/distributed.py
- [[Tensor_4]] - code - eval/eval_core.py
- [[Training schedule configuration.]] - rationale - training/configs.py
- [[Yield ``(tokens, targets)`` batches of random tokens.      Deterministic the sa]] - rationale - eval/eval_core.py
- [[__init__.py_2]] - code - eval/__init__.py
- [[__init__.py_8]] - code - training/__init__.py
- [[build_config_from_yaml()]] - code - training/pretrain.py
- [[cleanup_distributed()]] - code - utils/distributed.py
- [[configs.py]] - code - training/configs.py
- [[dataset.py]] - code - training/dataset.py
- [[device_5]] - code - training/validation.py
- [[eval_core.py]] - code - eval/eval_core.py
- [[main()_7]] - code - training/pretrain.py
- [[make_synthetic_loader()]] - code - eval/eval_core.py
- [[maybe_eval()]] - code - training/validation.py
- [[pretrain.py]] - code - training/pretrain.py
- [[run_lm_eval()]] - code - eval/run_lm_eval.py
- [[run_lm_eval.py]] - code - eval/run_lm_eval.py
- [[run_perplexity()]] - code - eval/eval_core.py
- [[validation.py]] - code - training/validation.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Training_Pipeline
SORT file.name ASC
```

## Connections to other communities
- 12 edges to [[_COMMUNITY_Config Bundle]]
- 12 edges to [[_COMMUNITY_Scheduler & Setup]]
- 6 edges to [[_COMMUNITY_Checkpoint Loading]]
- 5 edges to [[_COMMUNITY_Distributed Wrap]]
- 2 edges to [[_COMMUNITY_Async Data Loader]]
- 2 edges to [[_COMMUNITY_Numerical Health]]
- 2 edges to [[_COMMUNITY_Curriculum]]
- 2 edges to [[_COMMUNITY_Scheduling]]
- 1 edge to [[_COMMUNITY_Multi-Token Prediction]]
- 1 edge to [[_COMMUNITY_Compile Benchmarks]]
- 1 edge to [[_COMMUNITY_NorMuon Optimizer]]
- 1 edge to [[_COMMUNITY_Cautious Optimizer]]

## Top bridge nodes
- [[Pretrainer]] - degree 25, connects to 6 communities
- [[__init__.py_8]] - degree 21, connects to 6 communities
- [[configs.py]] - degree 15, connects to 5 communities
- [[pretrain.py]] - degree 15, connects to 3 communities
- [[.train()]] - degree 11, connects to 3 communities