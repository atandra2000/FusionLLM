---
source_file: "training/numerical_health.py"
type: "code"
community: "Health Monitor"
location: "L1"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Health_Monitor
---

# numerical_health.py

## Connections
- [[ActivationMonitor]] - `contains` [EXTRACTED]
- [[HealthConfig]] - `contains` [EXTRACTED]
- [[NumericalHealthMonitor]] - `contains` [EXTRACTED]
- [[RunsCsvLogger]] - `imports` [EXTRACTED]
- [[create_health_monitor()]] - `contains` [EXTRACTED]
- [[init_health_monitor()]] - `contains` [EXTRACTED]
- [[init_runs_csv()]] - `contains` [EXTRACTED]
- [[register_spike_callback()]] - `contains` [EXTRACTED]
- [[tensor_checks.py]] - `imports_from` [EXTRACTED]
- [[train_step.py]] - `imports_from` [EXTRACTED]
- [[trainer.py]] - `imports_from` [EXTRACTED]
- [[validate_gradients()]] - `imports` [EXTRACTED]
- [[validate_scalar()]] - `imports` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Health_Monitor