# FlowRL: A Taxonomy and Modular Framework for Reinforcement Learning with Diffusion Policies

[![PyPI version](https://img.shields.io/pypi/v/flowrl.svg)](https://pypi.org/project/flowrl) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-green.svg)](https://www.python.org/) [![Python 3.8+](https://static.pepy.tech/badge/flowrl)](https://pepy.tech/projects/flowrl) [![arXiv](https://img.shields.io/badge/arXiv-2603.27450-b31b1b.svg)](https://arxiv.org/abs/2603.27450)

Flow RL is a high-performance reinforcement learning library, combining modern deep RL algorithms with flow and diffusion models for advanced policy parameterization, planning ability or dynamics modeling. It features:
- **State-of-the-Art Algorithms and Efficiency**: We provide JAX implementations of SOTA algorithms, such FQL, BDPO, DAC and etc;
- **Flexible Flow Architectures**: We provide built-in support various types of flow and diffusion models, such as CNFs and DDPM;
- **Comprehensive Evaluations**: We test the algorithms on commonly adopted benchmark and provide the results.
Please check our arXiv paper for more details about the module design and benchmark results:
<p align="center">
  <a href="https://arxiv.org/abs/2603.27450">
    <img src="https://img.shields.io/badge/arXiv-2603.27450-b31b1b.svg" alt="arXiv:2603.27450"/>
  </a>
</p>

## 🚀 Installation & Usage
Currently FlowRL is hosted on PyPI and therefore can be installed via `pip install flowrl`. However, we recommend to clone and install the library using the following commands:
```bash
git clone https://github.com/typoverflow/flow-rl.git
cd flow-rl
pip install -e .
```

Alternatively, you can use our Docker image:
```bash
docker pull typoverflow/flow-rl
docker run --gpus all -it typoverflow/flow-rl bash
```

The entry files are presented in `examples/`. Please refer to the scripts in `scripts/` for how to execute the algorithms.

## 📊 Supported Algorithms

Offline RL:
| Algorithm | Location |
|:---:|:---:|
| IQL | `flowrl/agent/offline/iql.py` |
| IVR | `flowrl/agent/offline/ivr.py` |
| DQL | `flowrl/agent/offline/dql.py` |
| DTQL | `flowrl/agent/offline/dtql.py` |
| DAC | `flowrl/agent/offline/dac.py` |
| FQL | `flowrl/agent/offline/fql/fql.py` |
| BDPO | `flowrl/agent/offline/bdpo/bdpo.py` |

Online RL (On-Policy):
| Algorithm | Location |
|:---:|:---:|
| PPO | `flowrl/agent/online/ppo.py` |
| DPPO | `flowrl/agent/online/dppo.py` |
| FPOPP | `flowrl/agent/online/fpopp.py` |
| GenPO | `flowrl/agent/online/genpo.py` |

Online RL (Off-Policy):
| Algorithm | Location |
|:---:|:---:|
| SAC | `flowrl/agent/online/sac.py` |
| TD3 | `flowrl/agent/online/td3.py` |
| TD7 | `flowrl/agent/online/td7/td7.py` |
| QSM | `flowrl/agent/online/qsm.py` |
| QVPO | `flowrl/agent/online/qvpo.py` |
| DACER | `flowrl/agent/online/dacer.py` |
| SDAC | `flowrl/agent/online/sdac.py` |
| DPMD | `flowrl/agent/online/dpmd.py` |
| IDEM | `flowrl/agent/online/idem.py` |

## 📝 Citing Flow RL
If you use Flow RL in your research, please cite:
```bibtex
@article{gao2026flowrl,
  title={FlowRL: A Taxonomy and Modular Framework for Reinforcement Learning with Diffusion Policies},
  author={Gao, Chenxiao and Chen, Edward and Chen, Tianyi and Dai, Bo},
  journal={arXiv preprint arXiv:2603.27450},
  year={2026}
}
```

## 💎 Acknowledgements
Inspired by foundational work from
- [Jax-CORL](https://github.com/nissymori/JAX-CORL)
- [DAC](https://github.com/Fang-Lin93/DAC)
