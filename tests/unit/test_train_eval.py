import numpy as np

import train as train_module
from train import TrainingConfig, _aggregate_eval_histories, _evaluate_agent, _smooth_rewards


def test_aggregate_eval_histories_means_trials_at_matching_timesteps():
    eval_timesteps, eval_rewards = _aggregate_eval_histories(
        [
            [(2_000, -180.0), (4_000, -140.0)],
            [(2_000, -160.0), (4_000, -120.0)],
            [(2_000, -200.0)],
        ]
    )

    assert eval_timesteps == [2_000, 4_000]
    assert eval_rewards == [-180.0, -130.0]


def test_smooth_rewards_uses_trailing_window():
    rewards = [-200.0, -180.0, -160.0, -140.0, -120.0, -100.0]

    smoothed = _smooth_rewards(rewards, window=3)

    assert smoothed == [
        -200.0,
        -190.0,
        -180.0,
        -160.0,
        -140.0,
        -120.0,
    ]


def test_smooth_rewards_handles_window_larger_than_history():
    rewards = [-200.0, -150.0]

    smoothed = _smooth_rewards(rewards, window=5)

    assert smoothed == [-200.0, -175.0]


def test_evaluate_agent_uses_greedy_actions(monkeypatch):
    class EvalEnv:
        def __init__(self):
            self.steps = 0

        def reset(self, seed=None):
            self.steps = 0
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action):
            self.steps += 1
            return np.zeros(2, dtype=np.float32), 1.0, self.steps >= 2, False, {}

        def close(self):
            return None

    class Agent:
        def __init__(self):
            self.greedy_calls = 0

        def get_greedy_action(self, state):
            self.greedy_calls += 1
            return 1

        def get_action(self, state):
            raise AssertionError("evaluation should not use exploratory actions")

    monkeypatch.setattr(train_module.gym, "make", lambda *args, **kwargs: EvalEnv())

    config = TrainingConfig(n_eval_episodes=3)
    agent = Agent()

    mean_reward = _evaluate_agent(
        config=config,
        agent=agent,
        eval_rng=np.random.default_rng(0),
    )

    assert mean_reward == 2.0
    assert agent.greedy_calls == 6
