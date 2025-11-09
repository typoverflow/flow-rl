import argparse
import os

from omegaconf import OmegaConf

from flowrl.agent.offline import *
from flowrl.config.offline.toy2d import Config
from flowrl.utils.plot_toy2d import plot_data, plot_sample

SUPPORTED_AGENTS: dict[str, type[BaseAgent]] = {
    "bdpo": BDPOAgent,
    "dac": DACAgent,
}


def main():
    parser = argparse.ArgumentParser(description="Plot toy2d results")
    parser.add_argument("--log_dir", type=str, required=True, 
                       help="Path to the log directory containing checkpoints and config")
    parser.add_argument("--step", type=int, default=None,
                       help="Training step to load (default: use the step specified in config)")
    parser.add_argument("--pretrain", action="store_true",
                       help="Plot pretrain results (behavior policy)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (default: use the seed from config)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots (default: log_dir/images)")
    
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.join(args.log_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    cfg: Config = OmegaConf.load(config_path)
    
    # Override seed if specified
    if args.seed is not None:
        cfg.seed = args.seed
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    # Create output directory
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(args.log_dir, "images", "plot_script")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Output directory: {out_dir}")
    print(f"Task: {cfg.task}")
    print(f"Algorithm: {cfg.algo.name}")
    print(f"Seed: {cfg.seed}")
    
    # Plot dataset
    plot_data(out_dir, cfg.task)
    
    # Create agent
    algo_name = cfg.algo.name
    if algo_name not in SUPPORTED_AGENTS:
        raise ValueError(f"Unsupported algorithm: {algo_name}. "
                        f"Available: {list(SUPPORTED_AGENTS.keys())}")
    
    agent = SUPPORTED_AGENTS[algo_name](
        obs_dim=1,
        act_dim=2,
        cfg=cfg.algo,
        seed=cfg.seed,
    )
    
    if args.pretrain:
        # Load pretrain checkpoint
        step = args.step if args.step is not None else cfg.pretrain_steps
        ckpt_path = os.path.join(args.log_dir, "ckpt", "pretrain", str(step))
        print(f"Loading pretrain checkpoint from {ckpt_path}")
        agent.load(ckpt_path)
        
        # Plot samples from behavior policy
        plot_sample(out_dir, agent)
        
    else:
        # Load training checkpoint
        step = args.step if args.step is not None else cfg.train_steps
        
        # First check if we need to load pretrain checkpoint
        if cfg.load is not None:
            print(f"Loading pretrain checkpoint from {cfg.load}")
            agent.load(cfg.load)
        elif cfg.pretrain_steps > 0:
            pretrain_ckpt = os.path.join(args.log_dir, "ckpt", "pretrain", str(cfg.pretrain_steps))
            if os.path.exists(pretrain_ckpt):
                print(f"Loading pretrain checkpoint from {pretrain_ckpt}")
                agent.load(pretrain_ckpt)
        
        # Prepare training (copy behavior to actor, etc.)
        agent.prepare_training()
        
        # Load training checkpoint
        train_ckpt = os.path.join(args.log_dir, "ckpt", "train", str(step))
        print(f"Loading training checkpoint from {train_ckpt}")
        if not os.path.exists(train_ckpt):
            raise FileNotFoundError(f"Training checkpoint not found at {train_ckpt}")
        agent.load(train_ckpt)

        # Plot samples from actor
        plot_sample(out_dir, agent)
        
        # Use agent's custom plot_toy2d method if available
        agent_metrics = agent.plot_toy2d(out_dir, cfg.task)
        if agent_metrics is not None:
            print("Agent metrics:")
            for key, value in agent_metrics.items():
                print(f"  {key}: {value:.4f}")
    
    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
