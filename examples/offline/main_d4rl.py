import os
from functools import partial
from typing import Type

import gym
import hydra
import numpy as np
import omegaconf
import wandb
from gym.wrappers.transform_observation import TransformObservation
from omegaconf import OmegaConf
from tqdm import trange

from flowrl.agent.offline import *
from flowrl.config.offline.d4rl_config import Config
from flowrl.dataset.d4rl import D4RLDataset
from flowrl.env.offline.d4rl import make_env
from flowrl.types import *
from flowrl.utils.logger import CompositeLogger
from flowrl.utils.misc import set_seed_everywhere

SUPPORTED_AGENTS: Dict[str, Type[BaseAgent]] = {
    "iql": IQLAgent,
    "bdpo": BDPOAgent,
    "categorical_bdpo": CategoricalBDPOAgent,
    "ivr": IVRAgent,
    "fql": FQLAgent,
    "dac": DACAgent,
    "dql": DQLAgent,
    "dtql": DTQLAgent,
}

class Trainer():
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

        self.dataset = D4RLDataset(
            task=cfg.task,
            clip_eps=cfg.data.clip_eps,
            scan=cfg.data.scan,
            norm_obs=cfg.data.norm_obs,
            norm_reward=cfg.data.norm_reward,
        )
        self.obs_mean, self.obs_std = self.dataset.get_obs_stats()
        def make_env_fn(task, seed):
            env = make_env(task, seed)
            env = TransformObservation(env, lambda obs: (obs-self.obs_mean)/self.obs_std)
            return env
        self.env = make_env_fn(cfg.task, cfg.seed)
        self.eval_envs = gym.vector.SyncVectorEnv([
            partial(make_env_fn, cfg.task, cfg.seed+i) for i in range(cfg.eval.num_episodes)
        ])

        self.agent = SUPPORTED_AGENTS[cfg.algo.name](
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            cfg=cfg.algo,
            seed=cfg.seed,
        )

    def train(self):
        try:
            if self.cfg.load is not None:
                self.agent.load(self.cfg.load)
            else:
                if self.cfg.pretrain_steps > 0:
                    for i in trange(self.cfg.pretrain_steps+1, desc="pretraining"):
                        if i % self.cfg.eval.interval == 0:
                            self.eval_and_save(i, prefix="pretrain_eval")
                        batch = self.dataset.sample(batch_size=self.cfg.data.batch_size)
                        update_info = self.agent.pretrain_step(batch, step=i)
                        if i % self.cfg.log.interval == 0:
                            self.logger.log_scalars("pretrain", update_info, step=i)
            if self.cfg.pretrain_only:
                print("Pretrain completed! Aborting. ")
                exit(0)

            # actual training
            self.agent.prepare_training()
            for i in trange(self.cfg.train_steps+1, desc="training"):
                if i % self.cfg.eval.interval == 0:
                    self.eval_and_save(i)
                if i % self.cfg.eval.stats_interval == 0:
                    batch = self.dataset.sample(batch_size=self.cfg.data.batch_size)
                    stats = self.agent.compute_statistics(batch)
                    if stats is not None:
                        self.logger.log_scalars("", stats, step=i)
                batch = self.dataset.sample(batch_size=self.cfg.data.batch_size)
                update_info = self.agent.train_step(batch, step=i)
                if i % self.cfg.log.interval == 0:
                    self.logger.log_scalars("", update_info, step=i)
        except KeyboardInterrupt:
            print("Stopped by keyboard interruption. ")
        except RuntimeError as e:
            print("Stopped by exception: ", str(e))

    def eval_and_save(self, step: int, prefix: str = "eval"):
        # Initialize arrays to store results
        returns = np.zeros(self.cfg.eval.num_episodes)
        lengths = np.zeros(self.cfg.eval.num_episodes)

        # Reset all environments
        observations = self.eval_envs.reset()
        dones = np.zeros(self.cfg.eval.num_episodes, dtype=bool)

        # Run episodes in parallel
        while not np.all(dones):
            # Get actions for all environments
            actions, _ = self.agent.sample_actions(
                observations,
                deterministic=True,
                num_samples=self.cfg.eval.num_samples,
            )

            # Step all environments
            observations, _, new_dones, infos = self.eval_envs.step(actions)

            # Update done flags and collect episode returns/lengths
            for i, (done, info) in enumerate(zip(new_dones, infos)):
                if done and not dones[i]:  # Only process newly done episodes
                    returns[i] = info['episode']['return']
                    lengths[i] = info['episode']['length']
                    dones[i] = True

        eval_metrics = {
            "mean": np.mean(returns),
            "median": np.median(returns),
            "std": np.std(returns),
            "min": np.min(returns),
            "max": np.max(returns),
            "length": np.mean(lengths),
        }
        self.logger.log_scalars(prefix, eval_metrics, step=step)
        if self.cfg.log.save_ckpt:
            self.agent.save(os.path.join(self.ckpt_save_dir, f"{step}"))


@hydra.main(config_path="./config/d4rl", config_name="config", version_base=None)
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

    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
