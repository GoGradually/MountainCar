import argparse

from train import TrainingConfig, run_training
from viz import save_reward_plot


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN for MountainCar-v0")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--episodes", type=positive_int, default=None)
    parser.add_argument("--trials", type=positive_int, default=None)
    parser.add_argument("--sync-interval", type=positive_int, default=None)
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    if args.profile == "quick":
        config = TrainingConfig(
            episodes=100,
            trials=1,
            sync_interval=10,
            render_mode=None,
            log_progress=False,
        )
    else:
        config = TrainingConfig()

    episodes = args.episodes if args.episodes is not None else config.episodes
    trials = args.trials if args.trials is not None else config.trials
    sync_interval = (
        args.sync_interval if args.sync_interval is not None else config.sync_interval
    )

    return TrainingConfig(
        episodes=episodes,
        trials=trials,
        sync_interval=sync_interval,
        env_id=config.env_id,
        render_mode=config.render_mode,
        epsilon=config.epsilon,
        gamma=config.gamma,
        lr=config.lr,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        action_space=config.action_space,
        reward_shaping_scale=config.reward_shaping_scale,
        reward_shaping_offset=config.reward_shaping_offset,
        device=config.device,
        seed=args.seed,
        log_device=config.log_device,
        log_progress=config.log_progress,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    result = run_training(config)

    if args.plot_path is not None:
        plot_path = args.plot_path
    elif args.profile == "quick":
        plot_path = "artifacts/quick_reward.png"
    else:
        plot_path = "artifacts/full_reward.png"

    save_reward_plot(result.reward_means, plot_path, f"MountainCar ({args.profile})")

    last_reward = result.reward_means[-1] if result.reward_means else 0.0
    print(f"Elapsed: {result.elapsed_sec:.4f}s")
    print(f"Last mean reward: {last_reward:.4f}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
