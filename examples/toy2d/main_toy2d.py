import os
from typing import Type

import hydra
import omegaconf
from omegaconf import OmegaConf
from tqdm import trange

import wandb
from examples.toy2d.utils import compute_metrics, plot_data, plot_energy, plot_sample
from flowrl.agent.offline import *
from flowrl.agent.online import *
from flowrl.config.toy2d.config import Config
from flowrl.dataset.toy2d import Toy2dDataset
from flowrl.types import *
from flowrl.utils.logger import CompositeLogger
from flowrl.utils.misc import set_seed_everywhere

SUPPORTED_AGENTS: Dict[str, Type[BaseAgent]] = {
    "bdpo": BDPOAgent,
    "dac": DACAgent,
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
        self.plot_save_dir = os.path.join(self.logger.log_dir, "images")
        os.makedirs(self.plot_save_dir, exist_ok=True)

        OmegaConf.save(cfg, os.path.join(self.logger.log_dir, "config.yaml"))
        print("="*35+" Config "+"="*35)
        print(OmegaConf.to_yaml(cfg))
        print("="*80)
        print(f"\nSave results to: {self.logger.log_dir}\n")

        self.dataset = Toy2dDataset(
            task=cfg.task,
            scan=cfg.data.scan,
        )

        self.agent = SUPPORTED_AGENTS[cfg.algo.name](
            obs_dim=1, # dummy observation
            act_dim=2, # 2D action space
            cfg=cfg.algo,
            seed=cfg.seed,
        )

        # Plot dataset once at the beginning if enabled
        plot_data(self.plot_save_dir, self.cfg.task)

    def plot_and_eval(self, step: int, stage: str):
        """Plot visualizations and compute metrics."""
        metrics = {}

        savedir = os.path.join(self.plot_save_dir, stage, f"{step}")
        os.makedirs(savedir, exist_ok=True)

        plot_sample(savedir, self.agent)
        plot_energy(savedir, self.cfg.task, self.agent)
        metrics = compute_metrics(savedir, self.cfg.task, self.agent)
        if metrics:
            self.logger.log_scalars(f"{stage}_eval", metrics, step=step)

    def train(self):
        try:
            if self.cfg.load is not None:
                self.agent.load(self.cfg.load)
            else:
                if self.cfg.pretrain_steps > 0:
                    for i in trange(self.cfg.pretrain_steps+1, desc="pretraining"):
                        if i % self.cfg.log.save_interval == 0:
                            self.save(i, stage="pretrain")
                        if i % self.cfg.eval.interval == 0 and i > 0:
                            self.plot_and_eval(i, stage="pretrain")
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
                if i % self.cfg.log.save_interval == 0:
                    self.save(i, stage="train")
                if i % self.cfg.eval.interval == 0 and i > 0:
                    self.plot_and_eval(i, stage="train")
                if i % self.cfg.log.stats_interval == 0:
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

    def save(self, step: int, stage: str):
        if self.cfg.log.save_ckpt:
            self.agent.save(os.path.join(self.ckpt_save_dir, stage, f"{step}"))


@hydra.main(config_path="./config", config_name="config", version_base=None)
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
