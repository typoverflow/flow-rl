import os
import numpy as np
import scipy
from matplotlib import pyplot as plt

from tqdm import tqdm

from flowrl.agent.base import BaseAgent
from flowrl.dataset.toy2d import inf_train_gen

SAMPLE_GRAPH_SIZE = 2000

def energy_sample(task: str, ss: float, sample_per_state: int = 1000):
    """Sample data with energy-based weighting."""
    data, e = inf_train_gen(task, batch_size=1000*sample_per_state)
    ori_e = e
    e = e * ss
    
    index = np.random.choice(
        1000*sample_per_state, 
        p=scipy.special.softmax(e).squeeze(), 
        size=sample_per_state, 
        replace=False
    )
    data = data[index]
    ori_e = ori_e[index]
    return data, ori_e

def plot_data(out_dir: str, task: str):
    """Plot the dataset distribution with different temperature (beta) values."""
    plt.figure(figsize=(12, 3.0))
    ss = [0, 3, 10, 20]
    axes = []
    
    for i, s in enumerate(ss):
        plt.subplot(1, len(ss), i+1)
        data, e = energy_sample(task, s, sample_per_state=SAMPLE_GRAPH_SIZE)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(-4.5, 4.5)
        plt.ylim(-4.5, 4.5)
        
        if i == 0:
            mappable = plt.scatter(
                data[:, 0], data[:, 1], s=1, c=e, 
                cmap="winter", vmin=0, vmax=1, rasterized=True
            )
            plt.yticks(ticks=[-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4])
        else:
            plt.scatter(
                data[:, 0], data[:, 1], s=1, c=e, 
                cmap="winter", vmin=0, vmax=1, rasterized=True
            )
            plt.yticks(ticks=[-4, -2, 0, 2, 4], labels=[None, None, None, None, None])
        
        axes.append(plt.gca())
        plt.xticks(ticks=[-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4])
        plt.title(r'$\beta={}$'.format(s))
    
    plt.tight_layout()
    plt.gcf().colorbar(mappable, ax=axes, fraction=0.1, pad=0.02, aspect=12)
    saveto = os.path.join(out_dir, "data.png")
    plt.savefig(saveto, dpi=300)
    plt.close()
    print(f"Saved data plot to {saveto}")

def plot_sample(out_dir: str, agent: BaseAgent):
    """Plot samples from the agent's policy."""
    plt.figure(figsize=(3, 3))
    
    # Sample actions
    obs = np.zeros((SAMPLE_GRAPH_SIZE, 1))
    actions, _ = agent.sample_actions(
        obs=obs,
        deterministic=False,
        num_samples=1,
    )
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)
    plt.scatter(actions[:, 0], actions[:, 1], s=1, rasterized=True)
    plt.yticks(ticks=[-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4])
    plt.xticks(ticks=[-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4])
    plt.tight_layout()
    saveto = os.path.join(out_dir, "sample.png")
    plt.savefig(saveto, dpi=300)
    plt.close()
    tqdm.write(f"Saved sample plot to {saveto}")

