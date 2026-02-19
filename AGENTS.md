# AGENTS.md

## Project Structure

JAX-based diffusion policy RL algorithms.

- **Agents**: `flowrl/agent/online/` (online) and `flowrl/agent/offline/` (offline)
- **Configs**: `flowrl/config/` (dataclasses), `examples/**/config/` (YAML hyperparams)
- **Entry points**: `examples/` — organized by scenario (online/offline) and benchmark (DMControl, HumanoidBench, MuJoCo)

## General Instructions
- Keep the code clean, readable and consistent with existing ones.
- You can sacrifice grammar in your response for conciseness.

## Adding a New Algorithm

1. Read the official implementation; identify all hyperparameters.
2. Implement in `flowrl/agent/`. Match existing code style — reference SDAC (online) or DAC (offline).
3. Add config dataclasses and YAML files consistent with the official implementation.

## Proofreading Algorithms

- Flag anything abnormal vs. standard RL practices.
- If an official implementation is provided, diff against it for logic and hyperparameter mismatches.

## Pre-Release Checklist

- **README**: Identify undocumented algorithms. Prompt me to add experiment links — only include algorithms with results. Do not overwrite existing README content.
- **Dependencies**: If changes are needed, ask me before updating.
