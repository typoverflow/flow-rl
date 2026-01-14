import os

import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

from flowrl.agent.base import BaseAgent
from flowrl.agent.offline import *
from flowrl.agent.online import *
from flowrl.dataset.toy2d import Toy2dDataset, inf_train_gen

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
                cmap="viridis", vmin=0, vmax=1, rasterized=True
            )
            plt.yticks(ticks=[-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4])
        else:
            plt.scatter(
                data[:, 0], data[:, 1], s=1, c=e,
                cmap="viridis", vmin=0, vmax=1, rasterized=True
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


def plot_energy(out_dir, task: str, agent: BaseAgent):
    plt.rc('font', family='Times New Roman', size=16)

    x_range = np.linspace(-4.5, 4.5, 90)
    y_range = np.linspace(-4.5, 4.5, 90)
    a, b = np.meshgrid(x_range, y_range, indexing='ij')
    id_matrix = np.stack([b, a], axis=-1).reshape(-1, 2)
    zero = np.zeros((90*90, 1))

    # Get vmin and vmax for color scale
    data, e = energy_sample(task, 0, sample_per_state=SAMPLE_GRAPH_SIZE)
    vmin = e.min()
    vmax = e.max()

    def default_plot():
        pass

    def bdpo_plot():
        tt = [0, 1, 3, 5, 10, 20, 30, 40, 50]
        plt.figure(figsize=(30, 3.0))
        axes = []
        for i, t in enumerate(tt):
            plt.subplot(1, len(tt), i+1)
            if t == 0:
                critic = agent.q0_target
                c = critic(zero, id_matrix).mean(axis=0).reshape(90, 90)
            else:
                critic = agent.vt_target
                t_input = np.ones((90*90, 1)) * t
                c = critic(zero, id_matrix, t_input).mean(axis=0).reshape(90, 90)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlim(0, 89)
            plt.ylim(0, 89)
            if i == 0:
                mappable = plt.imshow(
                    c, origin='lower', vmin=vmin, vmax=vmax,
                    cmap="viridis", rasterized=True
                )
                plt.yticks(ticks=[5, 25, 45, 65, 85], labels=[-4, -2, 0, 2, 4])
            else:
                plt.imshow(
                    c, origin='lower', vmin=vmin, vmax=vmax,
                    cmap="viridis", rasterized=True
                )
                plt.yticks(ticks=[5, 25, 45, 65, 85], labels=[None, None, None, None, None])

            axes.append(plt.gca())
            plt.xticks(ticks=[5, 25, 45, 65, 85], labels=[-4, -2, 0, 2, 4])
            plt.title(f't={t}')
        plt.tight_layout()
        cbar = plt.gcf().colorbar(mappable, ax=axes, fraction=0.1, pad=0.02, aspect=12)
        plt.gcf().axes[-1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        saveto = os.path.join(out_dir, "qt_space.png")
        plt.savefig(saveto, dpi=300)
        plt.close()
        tqdm.write(f"Saved value plot to {saveto}")

    def dac_plot():
        plt.figure(figsize=(5, 3.0))
        axes = []
        c = agent.critic(zero, id_matrix).mean(axis=0).reshape(90, 90)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlim(0, 89)
        plt.ylim(0, 89)
        mappable = plt.imshow(c, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis", rasterized=True)
        plt.yticks(ticks=[5, 25, 45, 65, 85], labels=[-4, -2, 0, 2, 4])

        axes.append(plt.gca())
        plt.xticks(ticks=[5, 25, 45, 65, 85], labels=[-4, -2, 0, 2, 4])
        plt.title('Q Value')
        plt.gcf().colorbar(mappable, ax=axes, fraction=0.1, pad=0.02, aspect=12)
        plt.gcf().axes[-1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        saveto = os.path.join(out_dir, "qt_space.png")
        plt.savefig(saveto, dpi=300)
        plt.close()
        tqdm.write(f"Saved value plot to {saveto}")

    def aca_plot():
        tt = [0, 1, 3, 5, 10, 20]
        plt.figure(figsize=(20, 3.0))
        axes = []
        for i, t in enumerate(tt):
            plt.subplot(1, len(tt), i+1)
            if t == 0:
                model = agent.critic_target
                c = model(zero, id_matrix).mean(axis=0).reshape(90, 90)
            else:
                model = agent.value_target
                t_input = np.ones((90*90, 1)) * t
                c = model(zero, id_matrix, t_input).mean(axis=0).reshape(90, 90)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlim(0, 89)
            plt.ylim(0, 89)
            if i == 0:
                mappable = plt.imshow(
                    c, origin="lower", vmin=vmin, vmax=vmax,
                    cmap="viridis", rasterized=True
                )
                plt.yticks(ticks=[5, 25, 45, 65, 85], labels=[-4, -2, 0, 2, 4])
            else:
                plt.imshow(
                    c, origin="lower", vmin=vmin, vmax=vmax,
                    cmap="viridis", rasterized=True
                )
                plt.yticks(ticks=[5, 25, 45, 65, 85], labels=[None, None, None, None, None])

            axes.append(plt.gca())
            plt.xticks(ticks=[5, 25, 45, 65, 85], labels=[-4, -2, 0, 2, 4])
            plt.title(f't={t}')
        plt.tight_layout()
        cbar = plt.gcf().colorbar(mappable, ax=axes, fraction=0.1, pad=0.02, aspect=12)
        plt.gcf().axes[-1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        saveto = os.path.join(out_dir, "qt_space.png")
        plt.savefig(saveto, dpi=300)
        plt.close()
        tqdm.write(f"Saved value plot to {saveto}")

    if isinstance(agent, BDPOAgent):
        bdpo_plot()
    elif isinstance(agent, DACAgent):
        dac_plot()
    elif isinstance(agent, ACAAgent):
        aca_plot()
    else:
        default_plot()



def compute_metrics(out_dir: str, task: str, agent: BaseAgent):
    def default_metrics():
        return {}

    def bdpo_metrics():
        def test_value(task, from_dataset):
            datanum = 10000

            if from_dataset:
                # Get values from dataset
                dataset = Toy2dDataset(task=task, data_size=datanum, scan=False)
                batch = dataset.sample(batch_size=datanum)
                values = batch.reward.flatten()
            else:
                # Get values from actor samples
                save_is_pretraining = agent._is_pretraining
                agent._is_pretraining = True  # use behavior target to sample unadjusted actions
                samples, _ = agent.sample_actions(
                    obs=np.zeros((datanum, 1)),
                    deterministic=False,
                    num_samples=1,
                )
                agent._is_pretraining = save_is_pretraining
                # Compute value using the appropriate value function
                values = agent.q0_target(
                    np.zeros((datanum, 1)),
                    samples,
                ).mean(0)
            eta = agent.cfg.critic.eta
            T = agent.cfg.critic.steps

            # Compute target as logsumexp
            target = jax.scipy.special.logsumexp(
                values / (eta),
                b=(1 / datanum)
            ) * (eta)

            # Compute value at T using random samples
            N = 1000
            random_actions = jax.random.normal(jax.random.PRNGKey(0), shape=(N, 2))
            value_at_T = agent.vt_target(
                np.zeros((N, 1)),
                random_actions,
                np.ones((N, 1)) * T,
            ).mean()
            abs_error = np.abs(target - value_at_T)
            rel_error = abs_error / (np.abs(target) + 1e-8)

            prefix = "dataset" if from_dataset else "actor"
            return {
                f"{prefix}_abs_error": float(abs_error),
                f"{prefix}_rel_error": float(rel_error),
                f"{prefix}_value_at_T": float(value_at_T),
                f"{prefix}_target": float(target)
            }
        return {
            **test_value(task, from_dataset=True),
            **test_value(task, from_dataset=False),
        }

    if isinstance(agent, BDPOAgent):
        return bdpo_metrics()
    else:
        return default_metrics()
