import pytest

from train import TrainingConfig, run_training


@pytest.mark.integration
def test_run_training_smoke():
    config = TrainingConfig(
        episodes=3,
        trials=1,
        sync_interval=1,
        render_mode=None,
        seed=123,
        buffer_size=64,
        batch_size=4,
        log_device=False,
        log_progress=False,
    )

    result = run_training(config)

    assert len(result.reward_histories) == 1
    assert len(result.reward_histories[0]) == 3
    assert len(result.reward_means) == 3
    assert result.elapsed_sec >= 0.0
