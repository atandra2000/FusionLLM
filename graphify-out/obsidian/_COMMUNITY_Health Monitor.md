---
type: community
cohesion: 0.24
members: 11
---

# Health Monitor

**Cohesion:** 0.24 - loosely connected
**Members:** 11 nodes

## Members
- [[.register_alert_callback()]] - code - training/numerical_health.py
- [[Configuration for numerical health checks.]] - rationale - training/numerical_health.py
- [[Create a health monitor with optional alert callback.          Args         con]] - rationale - training/numerical_health.py
- [[HealthConfig]] - code - training/numerical_health.py
- [[Initialize the numerical health monitor from ConfigBundle.      Args         cf]] - rationale - training/numerical_health.py
- [[Register a callback to be called on anomaly detection.]] - rationale - training/numerical_health.py
- [[Register a callback to save checkpoint on spike detection.      Args         he]] - rationale - training/numerical_health.py
- [[create_health_monitor()]] - code - training/numerical_health.py
- [[init_health_monitor()]] - code - training/numerical_health.py
- [[numerical_health.py]] - code - training/numerical_health.py
- [[register_spike_callback()]] - code - training/numerical_health.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Health_Monitor
SORT file.name ASC
```

## Connections to other communities
- 7 edges to [[_COMMUNITY_Numerical Health]]
- 5 edges to [[_COMMUNITY_Scheduler & Setup]]
- 3 edges to [[_COMMUNITY_Runs Logger]]
- 3 edges to [[_COMMUNITY_Tensor Validation]]
- 2 edges to [[_COMMUNITY_Activation Monitor]]

## Top bridge nodes
- [[numerical_health.py]] - degree 13, connects to 5 communities
- [[HealthConfig]] - degree 7, connects to 3 communities
- [[init_health_monitor()]] - degree 6, connects to 2 communities
- [[register_spike_callback()]] - degree 6, connects to 2 communities
- [[create_health_monitor()]] - degree 5, connects to 1 community