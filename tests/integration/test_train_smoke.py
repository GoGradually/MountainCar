import csv

import numpy as np
import pytest

import train as train_module
from agent import AgentConfig
from train import TrainingConfig, run_training


class FixedLengthEnv:
    def __init__(self, episode_lengths: list[int]):
        self.episode_lengths = episode_lengths
        self.reset_count = 0
        self.steps_in_episode = 0
        self.current_episode_length = episode_lengths[0]
        self.reset_seeds: list[int | None] = []

    def reset(self, seed=None):
        assert self.reset_count < len(self.episode_lengths)
        self.current_episode_length = self.episode_lengths[self.reset_count]
        self.reset_count += 1
        self.steps_in_episode = 0
        self.reset_seeds.append(seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.steps_in_episode += 1
        terminated = self.steps_in_episode >= self.current_episode_length
        return np.zeros(2, dtype=np.float32), 1.0, terminated, False, {}

    def close(self):
        return None


class EnvFactory:
    def __init__(
        self,
        training_episode_lengths: list[int],
        eval_episode_lengths_per_call: list[list[int]],
    ):
        self.training_env = FixedLengthEnv(training_episode_lengths)
        self.eval_episode_lengths_per_call = eval_episode_lengths_per_call
        self.call_count = 0
        self.eval_envs: list[FixedLengthEnv] = []

    def __call__(self, *args, **kwargs):
        if self.call_count == 0:
            env = self.training_env
        else:
            eval_index = self.call_count - 1
            assert eval_index < len(self.eval_episode_lengths_per_call)
            env = FixedLengthEnv(self.eval_episode_lengths_per_call[eval_index])
            self.eval_envs.append(env)
        self.call_count += 1
        return env


@pytest.mark.integration
def test_run_training_stops_at_exact_timesteps_and_discards_partial_episode(monkeypatch, tmp_path):
    env_factory = EnvFactory([5, 5, 5], [[3] * 20, [3] * 20])
    monkeypatch.setattr(train_module.gym, "make", env_factory)
    eval_log_path = tmp_path / "single_trial_eval.csv"

    config = TrainingConfig(
        total_timesteps=12,
        trials=1,
        render_mode=None,
        seed=123,
        eval_freq=5,
        n_eval_episodes=20,
        eval_window=2,
        eval_seed=99,
        eval_log_path=str(eval_log_path),
        agent_config=AgentConfig(
            buffer_size=64,
            batch_size=4,
            train_start=8,
            train_freq=2,
            gradient_steps=1,
            target_sync_every=20,
        ),
        log_device=False,
        log_progress=False,
    )

    result = run_training(config)

    assert len(result.reward_histories) == 1
    assert result.reward_histories[0] == [5.0, 5.0]
    assert result.reward_means == [5.0, 5.0]
    assert result.timesteps_per_trial == [12]
    assert result.eval_timesteps == [5, 10]
    assert result.eval_reward_raw == [3.0, 3.0]
    assert result.eval_reward_smoothed == [3.0, 3.0]
    assert len(env_factory.eval_envs) == 2
    assert all(len(env.reset_seeds) == 20 for env in env_factory.eval_envs)
    assert result.elapsed_sec >= 0.0

    with eval_log_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))

    assert rows == [
        ["trial", "timestep", "eval_reward"],
        ["1", "5", "3.0"],
        ["1", "10", "3.0"],
    ]


@pytest.mark.integration
def test_run_training_averages_available_trials_only(monkeypatch, tmp_path):
    env_factory = EnvFactory(
        [5, 5, 4, 4, 4],
        [[2] * 20, [2] * 20, [2] * 20, [4] * 20, [4] * 20, [4] * 20],
    )
    monkeypatch.setattr(train_module.gym, "make", env_factory)
    eval_log_path = tmp_path / "multi_trial_eval.csv"

    config = TrainingConfig(
        total_timesteps=9,
        trials=2,
        render_mode=None,
        seed=123,
        eval_freq=3,
        n_eval_episodes=20,
        eval_window=2,
        eval_seed=99,
        eval_log_path=str(eval_log_path),
        agent_config=AgentConfig(
            buffer_size=64,
            batch_size=4,
            train_start=8,
            train_freq=2,
            gradient_steps=1,
            target_sync_every=20,
        ),
        log_device=False,
        log_progress=False,
    )

    result = run_training(config)

    assert result.reward_histories == [[5.0], [4.0, 4.0]]
    assert result.reward_means == [4.5, 4.0]
    assert result.timesteps_per_trial == [9, 9]
    assert result.eval_timesteps == [3, 6, 9]
    assert result.eval_reward_raw == [3.0, 3.0, 3.0]
    assert result.eval_reward_smoothed == [3.0, 3.0, 3.0]

    with eval_log_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))

    assert rows == [
        ["trial", "timestep", "eval_reward"],
        ["1", "3", "2.0"],
        ["1", "6", "2.0"],
        ["1", "9", "2.0"],
        ["2", "3", "4.0"],
        ["2", "6", "4.0"],
        ["2", "9", "4.0"],
    ]


@pytest.mark.integration
def test_run_training_uses_reproducible_random_eval_seeds(monkeypatch):
    first_factory = EnvFactory([3, 3], [[2] * 3])
    monkeypatch.setattr(train_module.gym, "make", first_factory)

    config = TrainingConfig(
        total_timesteps=3,
        trials=1,
        render_mode=None,
        seed=123,
        eval_freq=3,
        n_eval_episodes=3,
        eval_seed=77,
        agent_config=AgentConfig(
            buffer_size=64,
            batch_size=4,
            train_start=100,
            train_freq=2,
            gradient_steps=1,
            target_sync_every=20,
            eps_start=1.0,
            eps_final=1.0,
        ),
        log_device=False,
        log_progress=False,
    )

    first_result = run_training(config)
    first_eval_seeds = [env.reset_seeds[:] for env in first_factory.eval_envs]

    second_factory = EnvFactory([3, 3], [[2] * 3])
    monkeypatch.setattr(train_module.gym, "make", second_factory)
    second_result = run_training(config)
    second_eval_seeds = [env.reset_seeds[:] for env in second_factory.eval_envs]

    assert first_result.eval_reward_raw == second_result.eval_reward_raw
    assert first_eval_seeds == second_eval_seeds
    assert len(first_eval_seeds[0]) == 3
    assert len(set(first_eval_seeds[0])) == 3
