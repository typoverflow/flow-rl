# Flow RL

[![PyPI version](https://img.shields.io/pypi/v/flowrl.svg)](https://pypi.org/project/flowrl) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.11%2B-green.svg)](https://www.python.org/) [![Python 3.8+](https://static.pepy.tech/badge/flowrl)](https://pepy.tech/projects/flowrl)

Flow RL is a high-performance reinforcement learning library, combining modern deep RL algorithms with flow and diffusion models for advanced policy parameterization, planning ability or dynamics modeling. It features:
- **State-of-the-Art Algorithms and Efficiency**
  We provide JAX implementations of SOTA algorithms, such FQL, BDPO, DAC and etc;
- **Flexible Flow Architectures**
  We provide built-in support various types of flow and diffusion models, such as CNFs and DDPM;
- **Comprehensive Evaluations**
  We test the algorithms on commonly adopted benchmark and provide the results.

## 🚀 Installation & Usage
Currently FlowRL is hosted on PyPI and therefore can be installed via `pip install flowrl`. However, we recommend to clone and install the library using the following commands:
```bash
git clone https://github.com/typoverflow/flow-rl.git
cd flow-rl
pip install -e .
```

The entry files are presented in `examples/`. Please refer to the scripts in `scripts/` for how to execute the algorithms.

## 📊 Supported Algorithms
Offline RL:
|Algorithm|Location|WandB Report|
|:---:|:---:|:---:|
|IQL|`flowrl/agent/iql.py`| |
|IVR|`flowrl/agent/ivr.py`| |
|BDPO|`flowrl/agent/bdpo.py`| |

## 📝 Citing Flow RL
If you use Flow RL in your research, please cite:
```bibtex
@software{flow_rl,
  author       = {Chen-Xiao Gao and Mingjun Cao},
  title        = {Flow RL: Flow-based Reinforcement Learning Algorithms},
  year         = 2025,
  version      = {v0.0.1},
  url          = {https://github.com/typoverflow/flow-rl}
}
```

## 💎 Acknowledgements
Inspired by foundational work from
- [Jax-CORL](https://github.com/nissymori/JAX-CORL)
- [DAC](https://github.com/Fang-Lin93/DAC)
