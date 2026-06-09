---
type: community
cohesion: 0.25
members: 14
---

# Checkpoint Retention

**Cohesion:** 0.25 - loosely connected
**Members:** 14 nodes

## Members
- [[Delete all but the `n` most recent complete checkpoints.          best.safetenso]] - rationale - utils/checkpoint/retention.py
- [[Path_7]] - code - utils/checkpoint/retention.py
- [[Remove all files for a given checkpoint step.          Args         save_dir C]] - rationale - utils/checkpoint/retention.py
- [[Return all complete checkpoint step numbers, sorted ascending.          Args]] - rationale - utils/checkpoint/retention.py
- [[Return all step numbers that have checkpoint files or directories.          Args]] - rationale - utils/checkpoint/retention.py
- [[Return the highest complete step number, or None.          Args         save_di]] - rationale - utils/checkpoint/retention.py
- [[True iff all required files exist for this step.          Args         save_dir]] - rationale - utils/checkpoint/retention.py
- [[checkpoint_complete()]] - code - utils/checkpoint/retention.py
- [[delete_checkpoint()]] - code - utils/checkpoint/retention.py
- [[keep_last_n()]] - code - utils/checkpoint/retention.py
- [[latest_step()]] - code - utils/checkpoint/retention.py
- [[list_checkpoints()]] - code - utils/checkpoint/retention.py
- [[list_steps()]] - code - utils/checkpoint/retention.py
- [[retention.py]] - code - utils/checkpoint/retention.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Checkpoint_Retention
SORT file.name ASC
```

## Connections to other communities
- 1 edge to [[_COMMUNITY_Checkpoint Manager]]

## Top bridge nodes
- [[retention.py]] - degree 7, connects to 1 community