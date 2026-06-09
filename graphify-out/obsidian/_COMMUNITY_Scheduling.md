---
type: community
cohesion: 0.18
members: 13
---

# Scheduling

**Cohesion:** 0.18 - loosely connected
**Members:** 13 nodes

## Members
- [[.__init__()_31]] - code - training/schedules.py
- [[.__init__()_32]] - code - training/schedules.py
- [[.get_batch_size()]] - code - training/schedules.py
- [[.get_seq_len()]] - code - training/schedules.py
- [[BatchSizeSchedule]] - code - training/schedules.py
- [[Return the scheduled micro-batch size at step.]] - rationale - training/schedules.py
- [[Return the scheduled sequence length at step.]] - rationale - training/schedules.py
- [[Return the scheduled value at step for an ``initial → final`` ramp.      Args]] - rationale - training/schedules.py
- [[Schedule max sequence length from ``initial`` to ``final``.      This matches mo]] - rationale - training/schedules.py
- [[Schedule micro-batch size from ``initial`` to ``final``.      The schedule ramps]] - rationale - training/schedules.py
- [[SeqLenSchedule]] - code - training/schedules.py
- [[_interpolate()]] - code - training/schedules.py
- [[schedules.py]] - code - training/schedules.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Scheduling
SORT file.name ASC
```

## Connections to other communities
- 5 edges to [[_COMMUNITY_Scheduler & Setup]]
- 4 edges to [[_COMMUNITY_Config Bundle]]
- 2 edges to [[_COMMUNITY_Training Pipeline]]

## Top bridge nodes
- [[BatchSizeSchedule]] - degree 9, connects to 3 communities
- [[SeqLenSchedule]] - degree 9, connects to 3 communities
- [[schedules.py]] - degree 4, connects to 1 community