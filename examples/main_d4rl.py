import os
import random
import hydra
import numpy as np
from tqdm import trange
from omegaconf import OmegaConf, DictConfig

from flowrl.dataset.d4rl import D4RLDataset
from flowrl.env.offline.d4rl import make_env
from flowrl.utils.logger import TensorboardLogger
from flowrl.agent import *


SUPPORTED_AGENTS = {
    "dummy": DummyAgent, 
    "bdpo_discrete": BDPO_Discrete, 
    "DAC": DAC, 
}


class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg

        self.logger = TensorboardLogger(
            "/".join([cfg.log_dir, cfg.algo.cls, cfg.name, cfg.task]),
            "_".join(["seed"+str(cfg.seed), cfg.name]),
            activate=not cfg.debug
        )
        OmegaConf.save(cfg, os.path.join(self.logger.log_dir, "config.yaml"))

        self.env = make_env(cfg.task)
        self.dataset = D4RLDataset(
            task=cfg.task, 
            **cfg.data
        )
        self.obs_mean, self.obs_std = self.dataset.get_obs_stats()
        # TODO: @cmj: implement obs normalization wrapper for env

        self.agent = SUPPORTED_AGENTS[cfg.algo.cls](
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            **cfg.algo
        )
        
        self.global_step = 0
        

    def train(self):
        cfg = self.cfg

        pass
    
    def eval(self): 
        pass


@hydra.main(config_path="./config/d4rl", config_name="config")
def main(cfg: DictConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    trainer = Trainer(cfg)
    trainer.train()

