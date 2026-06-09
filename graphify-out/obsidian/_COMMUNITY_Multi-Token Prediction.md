---
type: community
cohesion: 0.11
members: 28
---

# Multi-Token Prediction

**Cohesion:** 0.11 - loosely connected
**Members:** 28 nodes

## Members
- [[.__init__()_12]] - code - models/mtp.py
- [[.__init__()_13]] - code - models/mtp.py
- [[.__init__()_14]] - code - models/mtp.py
- [[._get_causal_mask()]] - code - models/mtp.py
- [[._reinject_shared_heads()]] - code - models/mtp.py
- [[.compute_mtp_loss()]] - code - models/mtp.py
- [[.forward()_7]] - code - models/mtp.py
- [[.forward()_8]] - code - models/mtp.py
- [[.forward()_9]] - code - models/mtp.py
- [[.load_state_dict()]] - code - models/mtp.py
- [[.set_output_head()]] - code - models/mtp.py
- [[Compute the weighted MTP loss across all depths.          Returns 0 if ``mtp_pai]] - rationale - models/mtp.py
- [[Cross-entropy with a soft-cap (DeepSeek-V3 §3.3.1).      Mathematically equivale]] - rationale - models/mtp.py
- [[Linear]] - code - models/mtp.py
- [[MTP head for prediction depth d (1-indexed).]] - rationale - models/mtp.py
- [[MTPBlock]] - code - models/mtp.py
- [[MTPModule]] - code - models/mtp.py
- [[Module_3]] - code - models/mtp.py
- [[MultiTokenPrediction]] - code - models/mtp.py
- [[One MTP block a tiny pre-norm transformer over the fused input.      Phase 2.4]] - rationale - models/mtp.py
- [[Per-depth loss weight schedule.      The default is ``0.3, 0.2, 0.1`` for dept]] - rationale - models/mtp.py
- [[Run the main model and all MTP heads.          Returns ``(main_logits, mtp_pairs]] - rationale - models/mtp.py
- [[Tensor_17]] - code - models/mtp.py
- [[Wraps the main class`Transformer` with ``mtp_depth`` MTP heads.      Returns a]] - rationale - models/mtp.py
- [[device_2]] - code - models/mtp.py
- [[mtp.py]] - code - models/mtp.py
- [[mtp_loss_weight_schedule()]] - code - models/mtp.py
- [[softcap_ce()]] - code - models/mtp.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Multi-Token_Prediction
SORT file.name ASC
```

## Connections to other communities
- 3 edges to [[_COMMUNITY_Scheduler & Setup]]
- 2 edges to [[_COMMUNITY_Mamba Blocks]]
- 2 edges to [[_COMMUNITY_Config Bundle]]
- 1 edge to [[_COMMUNITY_Training Pipeline]]

## Top bridge nodes
- [[MultiTokenPrediction]] - degree 13, connects to 4 communities
- [[mtp.py]] - degree 7, connects to 2 communities