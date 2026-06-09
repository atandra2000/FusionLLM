---
type: community
cohesion: 0.22
members: 13
---

# Hardware Config

**Cohesion:** 0.22 - loosely connected
**Members:** 13 nodes

## Members
- [[Best-effort detect NVLinkpeer access between the listed GPU ids.      Returns]] - rationale - utils/device_setup.py
- [[HardwareConfig]] - code - utils/device_setup.py
- [[Return (allocated_gb, reserved_gb) and optionally print.]] - rationale - utils/device_setup.py
- [[Runtime hardware profile (from YAML ``hardware`` section).]] - rationale - utils/device_setup.py
- [[Select GPU, enable A100-friendly backends (TF32, cuDNN autotune),     verify the]] - rationale - utils/device_setup.py
- [[_check_nvlink_topology()]] - code - utils/device_setup.py
- [[_visible_device_count()]] - code - utils/device_setup.py
- [[device_6]] - code - utils/device_setup.py
- [[device_setup.py]] - code - utils/device_setup.py
- [[log_gpu_memory()]] - code - utils/device_setup.py
- [[maybe_empty_cache()]] - code - utils/device_setup.py
- [[parse_hardware_config()]] - code - utils/device_setup.py
- [[setup_training_device()]] - code - utils/device_setup.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Hardware_Config
SORT file.name ASC
```

## Connections to other communities
- 2 edges to [[_COMMUNITY_Scheduler & Setup]]

## Top bridge nodes
- [[device_setup.py]] - degree 8, connects to 1 community
- [[setup_training_device()]] - degree 7, connects to 1 community