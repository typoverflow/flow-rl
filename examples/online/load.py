import os

import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
from omegaconf import OmegaConf
from tqdm import tqdm, trange

import wandb
from flowrl.agent.online import *
from flowrl.config.online.mujoco import Config
from flowrl.dataset.buffer.state import ReplayBuffer
from flowrl.env.online.dmc_env import DMControlEnv
from flowrl.types import *
from flowrl.utils.logger import CompositeLogger
from flowrl.utils.misc import set_seed_everywhere

jax.config.update("jax_default_matmul_precision", "float32")

SUPPORTED_AGENTS: Dict[str, BaseAgent] = {
    "sac": SACAgent,
    "td3": TD3Agent,
    "td7": TD7Agent,
    "sdac": SDACAgent,
    "dpmd": DPMDAgent,
    "qsm": QSMAgent,
    "ctrlsr_td3": CtrlSRTD3Agent,
    "diffsr_td3": DiffSRTD3Agent,
    "diffsr_qsm": DiffSRQSMAgent,
}

class OffPolicyTrainer():
    def __init__(self, cfg: Config):
        self.cfg = cfg

        set_seed_everywhere(cfg.seed)
        self.logger = CompositeLogger(
            log_dir="/".join([cfg.log.dir, cfg.algo.name, cfg.log.tag, cfg.task]),
            name="seed"+str(cfg.seed),
            logger_config={
                "TensorboardLogger": {"activate": True},
                "WandbLogger": {
                    "activate": True,
                    "config": OmegaConf.to_container(cfg),
                    "settings": wandb.Settings(_disable_stats=True),
                    "project": cfg.log.project,
                    "entity": cfg.log.entity
                } if ("project" in cfg.log and "entity" in cfg.log) else {"activate": False},
            }
        )
        self.ckpt_save_dir = os.path.join(self.logger.log_dir, "ckpt")
        OmegaConf.save(cfg, os.path.join(self.logger.log_dir, "config.yaml"))
        print("="*35+" Config "+"="*35)
        print(OmegaConf.to_yaml(cfg))
        print("="*80)
        print(f"\nSave results to: {self.logger.log_dir}\n")

        # create env
        self.frame_skip = cfg.frame_skip
        self.train_env = DMControlEnv(cfg.task, cfg.seed, False, cfg.frame_skip, cfg.frame_stack)
        self.eval_env = [DMControlEnv(cfg.task, cfg.seed + i*100, False, cfg.frame_skip, cfg.frame_stack) for i in range(cfg.eval.num_episodes)]

        # create buffer
        self.use_lap_buffer = cfg.lap_alpha > 0
        self.buffer = ReplayBuffer(
            obs_dim=self.train_env.observation_space.shape[-1],
            action_dim=self.train_env.action_space.shape[-1],
            max_size=cfg.buffer_size,
            norm_obs=True,
            norm_reward=cfg.norm_reward,
            lap_alpha=cfg.lap_alpha,
        )

        # create agent
        self.agent = SUPPORTED_AGENTS[cfg.algo.name](
            obs_dim=self.train_env.observation_space.shape[-1],
            act_dim=self.train_env.action_space.shape[-1],
            cfg=cfg.algo,
            seed=cfg.seed,
        )

        self.global_step = 0
        self.global_episode = 0

    @property
    def global_frame(self) -> int:
        return self.frame_skip * self.global_step

    def train(self):
        cfg = self.cfg

        load_dir = os.path.join(
            "/localscratch/cgao304/save/diffsr_qsm/save-qens10",
            self.cfg.task,
        )
        for seed in os.listdir(load_dir):
            load_dir = os.path.join(load_dir, seed, "ckpt")
            break

        # load_dir = f"/nethome/cgao304/workspace/flow-rl/logs/diffsr_qsm/save-qens10/{self.cfg.task}/ckpt"
        iters = os.listdir(load_dir)
        iters = [int(i) for i in iters if i.isdigit()]
        iters = sorted(iters)
        ld_scores = []
        scores = []
        for iter in tqdm(iters):
            ckpt_dir = os.path.join(load_dir, str(iter))
            print(f"Loading checkpoint from {ckpt_dir}")
            self.agent.load(ckpt_dir)

            obs_mean = np.load(os.path.join(ckpt_dir, "obs_rms_mean.npy"))
            obs_mean_square = jnp.load(os.path.join(ckpt_dir, "obs_rms_mean_square.npy"))
            scaler = jnp.load(os.path.join(ckpt_dir, "scaler.npy"))
            print(f"Scaler: {scaler}")
            self.buffer.obs_rms.mean = obs_mean
            self.buffer.obs_rms.mean_square = obs_mean_square
            self.agent.scaler = scaler

            ld_metrics = self.eval_and_save(deterministic=True)
            metrics = self.eval_and_save(deterministic=False)
            ld_scores.append(ld_metrics['mean'])
            scores.append(metrics['mean'])
            print(f"Iter {iter}: LD = {ld_metrics['mean']}, QSM = {metrics['mean']}")

        import matplotlib.pyplot as plt
        plt.plot(iters, ld_scores, label="LD")
        plt.plot(iters, scores, label="QSM")
        plt.legend()
        plt.title(f"{self.cfg.task}")
        plt.savefig("ld_scores.png")
        plt.close()

    def eval_and_save(self, deterministic: bool = True):
        # initialize arrays to store results
        returns = np.zeros(self.cfg.eval.num_episodes)
        lengths = np.zeros(self.cfg.eval.num_episodes)

        # reset all environments
        obs = np.stack([env.reset()[0] for env in self.eval_env], axis=0)
        returns = np.zeros(self.cfg.eval.num_episodes)
        lengths = np.zeros(self.cfg.eval.num_episodes)
        dones = np.zeros(self.cfg.eval.num_episodes, dtype=bool)

        # run episodes in parallel
        while not np.all(dones):
            # get actions for all environments
            actions, _ = self.agent.sample_actions(
                self.buffer.normalize_obs(obs),
                deterministic=deterministic,
                num_samples=self.cfg.eval.num_samples,
            )
            actions = np.asarray(actions)

            # step all environments
            obs, rewards, terminated, truncated, infos = zip(*[env.step(action) for env, action in zip(self.eval_env, actions)])
            obs = np.stack(obs, axis=0)
            rewards = np.stack(rewards, axis=0)
            terminated = np.stack(terminated, axis=0)
            truncated = np.stack(truncated, axis=0)
            new_dones = terminated | truncated

            returns += rewards * (1-dones)
            lengths += 1 * (1-dones)
            dones = dones | new_dones

        eval_metrics = {
            "mean": np.mean(returns),
            "median": np.median(returns),
            "std": np.std(returns),
            "min": np.min(returns),
            "max": np.max(returns),
            "length": np.mean(lengths),
        }
        return eval_metrics

class OnPolicyTrainer():
    pass


@hydra.main(config_path="./config/dmc", config_name="config", version_base=None)
def main(cfg: Config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    try:
        algo_name = cfg.algo.name
    except omegaconf.errors.MissingMandatoryValue:
        err_string = "Algorithm is not specified. Please specify the algorithm via `algo=<algo_name>` in command."
        err_string += "\nAvailable algorithms are:\n  "
        err_string += "\n  ".join(SUPPORTED_AGENTS.keys())
        print(err_string)
        exit(1)

    trainer = OffPolicyTrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
