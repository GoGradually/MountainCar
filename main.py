import argparse
from pathlib import Path

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


def resolve_plot_path(args: argparse.Namespace) -> str:
    if args.plot_path is not None:
        return args.plot_path
    if args.profile == "quick":
        return "artifacts/quick_reward.png"
    return "artifacts/full_reward.png"


def derive_eval_log_path(plot_path: str) -> str:
    plot_file = Path(plot_path)
    return str(plot_file.with_name(f"{plot_file.stem}_eval.csv"))


def build_config(args: argparse.Namespace, eval_log_path: str | None = None) -> TrainingConfig:
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
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        eval_window=config.eval_window,
        eval_seed=config.eval_seed,
        eval_log_path=eval_log_path,
        log_device=config.log_device,
        log_progress=config.log_progress,
    )


def main() -> None:
    args = parse_args()
    plot_path = resolve_plot_path(args)
    eval_log_path = derive_eval_log_path(plot_path)
    config = build_config(args, eval_log_path=eval_log_path)
    result = run_training(config)

    save_reward_plot(
        result.eval_timesteps,
        result.eval_reward_raw,
        result.eval_reward_smoothed,
        plot_path,
        f"MountainCar DQN ({args.profile})",
        "Timesteps",
        "Mean Evaluation Reward",
    )

    last_reward = result.eval_reward_smoothed[-1] if result.eval_reward_smoothed else 0.0
    print(f"Elapsed: {result.elapsed_sec:.4f}s")
    print(f"Last smoothed eval reward: {last_reward:.4f}")
    print(f"Eval log saved to: {eval_log_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
