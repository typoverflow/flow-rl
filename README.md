# Flow RL

[![PyPI version](https://img.shields.io/pypi/v/flowrl.svg)](https://pypi.org/project/flowrl) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-green.svg)](https://www.python.org/) [![Python 3.8+](https://static.pepy.tech/badge/flowrl)](https://pepy.tech/projects/flowrl)

Flow RL is a high-performance reinforcement learning library, combining modern deep RL algorithms with flow and diffusion models for advanced policy parameterization, planning ability or dynamics modeling. It features:
- **State-of-the-Art Algorithms and Efficiency**: We provide JAX implementations of SOTA algorithms, such FQL, BDPO, DAC and etc;
- **Flexible Flow Architectures**: We provide built-in support various types of flow and diffusion models, such as CNFs and DDPM;
- **Comprehensive Evaluations**: We test the algorithms on commonly adopted benchmark and provide the results.

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
|Algorithm|Location|WandB Report|
|:---:|:---:|:---:|
|IQL|`flowrl/agent/iql.py`|[[Performance]](https://wandb.ai/lamda-rl/flow-rl?nw=urvdu9rz7b&panelDisplayName=eval%2Fmean&panelSectionName=eval) [[Full Log]](https://wandb.ai/lamda-rl/flow-rl?nw=urvdu9rz7b)|
|IVR|`flowrl/agent/ivr.py`|[[Performance]](https://wandb.ai/lamda-rl/flow-rl/panel/nz7r4sj4n?nw=oslzekjlr1q) [[Full Log]](https://wandb.ai/lamda-rl/flow-rl?nw=oslzekjlr1q)|
|FQL|`flowrl/agent/fql/fql.py`|[[Performance]](https://wandb.ai/lamda-rl/flow-rl?nw=u9y84ki7rdi&panelDisplayName=eval%2Fmean&panelSectionName=eval) [[Full Log]](https://wandb.ai/lamda-rl/flow-rl?nw=u9y84ki7rdi)|
|DAC|`flowrl/agent/dac.py`|[[Performance]](https://wandb.ai/lamda-rl/flow-rl/panel/nz7r4sj4n?nw=uqr7jg46c5) [[Full Log]](https://wandb.ai/lamda-rl/flow-rl?nw=uqr7jg46c5)|
|BDPO|`flowrl/agent/bdpo/bdpo.py`|[[Performance]](https://wandb.ai/lamda-rl/flow-rl/panel/nz7r4sj4n?nw=2q8v54gusia) [[Full Log]](https://wandb.ai/lamda-rl/flow-rl?nw=2q8v54gusia)|

Online RL
<table>
  <thead>
    <tr>
      <th style="border-right: 1px solid #ddd; padding: 8px;">Algorithm</th>
      <th style="border-right: 1px solid #ddd; padding: 8px;">Location</th>
      <th style="padding: 8px;">WandB Report</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 1px solid #ddd; padding: 8px;">SAC</td>
      <td style="border-right: 1px solid #ddd; padding: 8px;"><code>flowrl/agent/online/sac.py</code></td>
      <td style="padding: 8px;" rowspan="7">
        <!-- Add your content here, for example: -->
        <a href="https://wandb.ai/gaochenxiao/flow-rl-online?nw=nvcqz1emwuc">Gym-MuJoCo Results</a><br>
        <!-- More content can be placed here if needed -->
      </td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #ddd; padding: 8px;">TD3</td>
      <td style="border-right: 1px solid #ddd; padding: 8px;"><code>flowrl/agent/online/td3.py</code></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #ddd; padding: 8px;">TD7</td>
      <td style="border-right: 1px solid #ddd; padding: 8px;"><code>flowrl/agent/online/td7.py</code></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #ddd; padding: 8px;">SDAC</td>
      <td style="border-right: 1px solid #ddd; padding: 8px;"><code>flowrl/agent/online/sdac.py</code></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #ddd; padding: 8px;">QSMAgent</td>
      <td style="border-right: 1px solid #ddd; padding: 8px;"><code>flowrl/agent/online/qsm.py</code></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #ddd; padding: 8px;">DACERAgent</td>
      <td style="border-right: 1px solid #ddd; padding: 8px;"><code>flowrl/agent/online/dacer.py</code></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #ddd; padding: 8px;">QVPOAgent</td>
      <td style="border-right: 1px solid #ddd; padding: 8px;"><code>flowrl/agent/online/qvpo.py</code></td>
    </tr>
  </tbody>
</table>

## 📝 Citing Flow RL
If you use Flow RL in your research, please cite:
```bibtex
@software{flow_rl,
  author       = {Chen-Xiao Gao and Mingjun Cao and Edward Chen},
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
