import os
import hydra
import numpy as np
from tqdm import trange
from omegaconf import OmegaConf

from flowrl.dataset.d4rl import D4RLDataset
from flowrl.env.offline.d4rl import make_env
from flowrl.utils.logger import TensorboardLogger
from gymnasium.wrappers.transform_observation import TransformObservation
from flowrl.agent import SUPPORTED_AGENTS
from flowrl.config.d4rl import Config


class Trainer():
    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.logger = TensorboardLogger(
            "/".join([cfg.log.dir, cfg.algo.name, cfg.log.tag, cfg.task]),
            "_".join(["seed"+str(cfg.seed), cfg.log.tag]),
            activate=True
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
            norm_obs=cfg.norm_obs,
            norm_reward=cfg.data.norm_reward,
        )
        self.obs_mean, self.obs_std = self.dataset.get_obs_stats()
        self.env = TransformObservation(make_env(cfg.task), lambda obs: (obs-self.obs_mean)/self.obs_std)

        self.agent = SUPPORTED_AGENTS[cfg.algo.name](
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            cfg=cfg.algo
        )

    def train(self):
        try:
            # pretraining
            if self.cfg.mode == "pretrain":
                for i in trange(self.cfg.pretrain_steps, desc="pretraining"):
                    if i % self.cfg.eval.interval == 0:
                        self.eval_and_save(i, use_behavior=True)
                    batch = self.dataset.sample(batch_size=self.cfg.data.batch_size)
                    update_info = self.agent.pretrain_step(batch, step=i)
                    if i % self.cfg.log.interval == 0:
                        self.logger.log_scalars("pretrain", update_info, step=i)
                self.eval_and_save(i+1, use_behavior=True)
                return
            else:
                if self.cfg.load is not None:
                    self.agent.load(self.cfg.load)

            # actual training
            self.agent.prepare_training()
            for i in trange(self.cfg.train_steps+1, desc="training"):
                if i % self.cfg.eval.interval == 0:
                    self.eval_and_save(i, use_behavior=False)
                if i % self.cfg.eval.stats_interval == 0:
                    batch = self.dataset.sample(batch_size=self.cfg.data.batch_size)
                    stats = self.agent.compute_statistics(batch)
                    if stats is not None:
                        self.logger.log_scalars("", stats, step=i)
                batch = self.dataset.sample(batch_size=self.cfg.data.batch_size)
                update_info = self.agent.train_step(batch, step=i)
                if i % self.cfg.log.interval == 0:
                    self.logger.log_scalars("train", update_info, step=i)
        except (KeyboardInterrupt, RuntimeError) as e:
            print("Stopped by exception: ", e)
    
    def eval_and_save(self, step: int, use_behavior: bool, ):
        returns, lengths, info = [], [], {}
        for _ in range(self.cfg.eval.num_episodes):
            (observation, _), done = self.env.reset(), False
            while not done:
                action, _ = self.agent.sample_actions(
                    observation.reshape(1, self.agent.obs_dim),
                    use_behavior=use_behavior,
                    temperature=self.cfg.eval.temperature,
                    num_samples=self.cfg.eval.num_samples,
                )
                action = action[0]
                observation, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

            # episodic statistics from wrappers/EpisodeMonitor
            returns.append(info['episode']['return'])
            lengths.append(info['episode']['length'])

        eval_metrics = {
            'mean': np.mean(returns),
            'median': np.median(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'max': np.max(returns),
            'length': np.mean(lengths)
        }
        self.logger.log_scalars("pretrain_eval", eval_metrics, step=step)
        if self.cfg.log.save_ckpt:
            self.agent.save(os.path.join(self.ckpt_save_dir, f"{step}.ckpt"))


@hydra.main(config_path="./config/d4rl", config_name="config", version_base=None)
def main(cfg: Config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
