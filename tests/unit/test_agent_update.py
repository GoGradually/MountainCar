import numpy as np
import torch
from unittest.mock import patch

from agent import AgentConfig, DQNAgent


def _params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [param.detach().clone() for param in model.parameters()]


def _changed(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    return any(
        not torch.allclose(prev, cur)
        for prev, cur in zip(before, after, strict=True)
    )


def _transition(i: int, action_space: int):
    state = np.array([i * 0.01, i * 0.02], dtype=np.float32)
    action = i % action_space
    reward = 1.0
    next_state = state + 0.05
    done = False
    return state, action, reward, next_state, done


def test_update_does_not_train_before_warmup():
    config = AgentConfig(
        buffer_size=64,
        batch_size=4,
        train_start=8,
        train_freq=1,
        gradient_steps=1,
    )
    agent = DQNAgent(total_timesteps=1_000, config=config, device="cpu", log_device=False)
    initial_params = _params(agent.qnet)

    for i in range(config.train_start - 1):
        agent.update(*_transition(i, agent.action_space))

    updated_params = _params(agent.qnet)
    assert not _changed(initial_params, updated_params)


def test_next_state_values_use_target_network_max():
    config = AgentConfig(action_space=2)
    agent = DQNAgent(total_timesteps=1_000, config=config, device="cpu", log_device=False)

    class FixedNet(torch.nn.Module):
        def __init__(self, values: list[float]):
            super().__init__()
            self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0] if x.dim() > 1 else 1
            return self.values.unsqueeze(0).repeat(batch, 1)

    agent.qnet = FixedNet([5.0, 1.0]).to(agent.device)
    agent.target_qnet = FixedNet([2.0, 9.0]).to(agent.device)

    next_state = torch.zeros((3, 2), dtype=torch.float32, device=agent.device)
    next_q = agent._next_state_values(next_state)

    assert torch.allclose(next_q, torch.tensor([9.0, 9.0, 9.0], device=agent.device))


def test_update_respects_train_frequency_after_warmup():
    config = AgentConfig(
        buffer_size=64,
        batch_size=4,
        train_start=4,
        train_freq=3,
        gradient_steps=1,
        target_sync_every=9999,
    )
    agent = DQNAgent(total_timesteps=1_000, config=config, device="cpu", log_device=False)
    initial_params = _params(agent.qnet)

    for i in range(4):
        agent.update(*_transition(i, agent.action_space))

    updated_params = _params(agent.qnet)
    assert not _changed(initial_params, updated_params)


def test_update_changes_qnet_parameters_on_train_step():
    np.random.seed(0)
    torch.manual_seed(0)
    config = AgentConfig(
        buffer_size=128,
        batch_size=4,
        train_start=4,
        train_freq=2,
        gradient_steps=2,
        target_sync_every=9999,
    )
    agent = DQNAgent(total_timesteps=1_000, config=config, device="cpu", log_device=False)
    initial_params = _params(agent.qnet)

    for i in range(6):
        agent.update(*_transition(i, agent.action_space))

    updated_params = _params(agent.qnet)
    assert _changed(initial_params, updated_params)


def test_update_applies_gradient_clipping_on_train_step():
    config = AgentConfig(
        buffer_size=64,
        batch_size=4,
        train_start=4,
        train_freq=1,
        gradient_steps=1,
        target_sync_every=9999,
        max_grad_norm=3.5,
    )
    agent = DQNAgent(total_timesteps=1_000, config=config, device="cpu", log_device=False)

    with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
        for i in range(4):
            agent.update(*_transition(i, agent.action_space))

    assert mock_clip.call_count == 1
    params, max_norm = mock_clip.call_args.args
    expected_params = list(agent.qnet.parameters())
    clipped_params = list(params)
    assert len(clipped_params) == len(expected_params)
    assert all(
        clipped is expected
        for clipped, expected in zip(clipped_params, expected_params, strict=True)
    )
    assert max_norm == 3.5


def test_update_skips_gradient_clipping_before_warmup_and_off_frequency():
    config = AgentConfig(
        buffer_size=64,
        batch_size=4,
        train_start=4,
        train_freq=3,
        gradient_steps=1,
        target_sync_every=9999,
    )
    agent = DQNAgent(total_timesteps=1_000, config=config, device="cpu", log_device=False)

    with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
        for i in range(4):
            agent.update(*_transition(i, agent.action_space))

    assert mock_clip.call_count == 0


def test_target_network_syncs_on_schedule():
    config = AgentConfig(
        buffer_size=128,
        batch_size=4,
        train_start=4,
        train_freq=1,
        gradient_steps=1,
        target_sync_every=5,
    )
    agent = DQNAgent(total_timesteps=1_000, config=config, device="cpu", log_device=False)
    with torch.no_grad():
        for param in agent.qnet.parameters():
            param.add_(0.5)

    out_of_sync = any(
        not torch.allclose(source, target)
        for source, target in zip(
            agent.qnet.parameters(), agent.target_qnet.parameters(), strict=True
        )
    )
    assert out_of_sync

    for i in range(5):
        agent.update(*_transition(i, agent.action_space))

    for source, target in zip(
        agent.qnet.parameters(), agent.target_qnet.parameters(), strict=True
    ):
        assert torch.allclose(source, target)
