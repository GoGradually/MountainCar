import argparse

from main import build_config, derive_eval_log_path, resolve_plot_path


def test_resolve_plot_path_uses_profile_default():
    args = argparse.Namespace(
        profile="quick",
        timesteps=None,
        trials=None,
        plot_path=None,
        seed=None,
    )

    assert resolve_plot_path(args) == "artifacts/quick_reward.png"


def test_resolve_plot_path_uses_custom_path():
    args = argparse.Namespace(
        profile="full",
        timesteps=None,
        trials=None,
        plot_path="custom/output.png",
        seed=None,
    )

    assert resolve_plot_path(args) == "custom/output.png"


def test_derive_eval_log_path_uses_plot_stem():
    assert derive_eval_log_path("artifacts/custom_reward.png") == "artifacts/custom_reward_eval.csv"


def test_build_config_sets_eval_log_path():
    args = argparse.Namespace(
        profile="full",
        timesteps=None,
        trials=None,
        plot_path=None,
        seed=7,
    )

    config = build_config(args, eval_log_path="artifacts/full_reward_eval.csv")

    assert config.eval_log_path == "artifacts/full_reward_eval.csv"
    assert config.seed == 7
