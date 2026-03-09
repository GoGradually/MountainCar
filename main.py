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
    parser.add_argument("--timesteps", type=positive_int, default=None)
    parser.add_argument("--trials", type=positive_int, default=None)
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    if args.profile == "quick":
        config = TrainingConfig(
            total_timesteps=20_000,
            trials=1,
            render_mode=None,
            log_progress=False,
        )
    else:
        config = TrainingConfig()

    timesteps = args.timesteps if args.timesteps is not None else config.total_timesteps
    trials = args.trials if args.trials is not None else config.trials

    return TrainingConfig(
        total_timesteps=timesteps,
        trials=trials,
        env_id=config.env_id,
        render_mode=config.render_mode,
        agent_config=config.agent_config,
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

    save_reward_plot(
        result.reward_means,
        plot_path,
        f"MountainCar DQN ({args.profile})",
    )

    last_reward = result.reward_means[-1] if result.reward_means else 0.0
    print(f"Elapsed: {result.elapsed_sec:.4f}s")
    print(f"Last mean reward: {last_reward:.4f}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
