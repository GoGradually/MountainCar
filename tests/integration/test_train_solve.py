import pytest

from agent import AgentConfig
from train import TrainingConfig, run_training


@pytest.mark.integration
@pytest.mark.slow
def test_run_training_reaches_mountain_car_threshold():
    config = TrainingConfig(
        total_timesteps=120_000,
        trials=3,
        render_mode=None,
        seed=123,
        eval_freq=2_000,
        n_eval_episodes=20,
        eval_window=5,
        agent_config=AgentConfig(),
        log_device=False,
        log_progress=False,
    )

    result = run_training(config)
    final_window = float(result.eval_reward_smoothed[-1])

    assert final_window >= -110.0
