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

    def reset(self, seed=None):
        assert self.reset_count < len(self.episode_lengths)
        self.current_episode_length = self.episode_lengths[self.reset_count]
        self.reset_count += 1
        self.steps_in_episode = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.steps_in_episode += 1
        terminated = self.steps_in_episode >= self.current_episode_length
        return np.zeros(2, dtype=np.float32), 1.0, terminated, False, {}

    def close(self):
        return None


@pytest.mark.integration
def test_run_training_stops_at_exact_timesteps_and_discards_partial_episode(monkeypatch):
    env = FixedLengthEnv([5, 5, 5])
    monkeypatch.setattr(train_module.gym, "make", lambda *args, **kwargs: env)

    config = TrainingConfig(
        total_timesteps=12,
        trials=1,
        render_mode=None,
        seed=123,
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
    assert result.elapsed_sec >= 0.0


@pytest.mark.integration
def test_run_training_averages_available_trials_only(monkeypatch):
    env = FixedLengthEnv([5, 5, 4, 4, 4])
    monkeypatch.setattr(train_module.gym, "make", lambda *args, **kwargs: env)

    config = TrainingConfig(
        total_timesteps=9,
        trials=2,
        render_mode=None,
        seed=123,
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
