# DeepSeek Project Agents and Skills

## Available Agents

### Core Agents
- **general**: Default general-purpose agent for everyday tasks
- **reviewer**: Code correctness reviewer - validates implementations, checks numerical stability, ensures best practices
- **architect**: Systems architect - designs and optimizes software architectures
- **research**: Research specialist - analyzes latest research and technologies
- **plan**: Strategic planning agent - coordinates complex development tasks
- **debugger**: Debugging specialist - identifies and resolves bugs
- **optimizer**: Performance optimization specialist - analyzes and improves system performance
- **tester**: Testing specialist - designs and evaluates test strategies
- **documenter**: Documentation specialist - creates and maintains technical documentation
- **training**: Training specialist - optimizes training configurations
- **cuda**: GPU optimization specialist - optimizes CUDA kernels and memory efficiency

### DeepSeek-Specialized Agents
- **deepseek-reviewer**: Reviews deep learning code for numerical stability, gradient issues, and best practices
- **deepseek-architect**: Analyzes model architecture decisions (MoE, attention, Mamba-MLA)
- **deepseek-training-analyst**: Evaluates training configurations and optimization strategies
- **deepseek-moe-specialist**: Reviews Mixture-of-Experts implementations

## Available Skills

### Project Skills
- **deepseek-analyzer**: Analyzes DeepSeek model implementations (NUMERICAL STABILITY, ARCHITECTURE CORRECTNESS)

### Configuration Skills
- **customize-opencode**: For configuring opencode itself (opencode.json, skills, agents)

## Usage
- `/agent <name>` - Use a specific agent
- `/skill <name>` - Load a specific skill
- `/help` - Show available commands