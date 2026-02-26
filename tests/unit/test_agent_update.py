import numpy as np
import torch

from agent import DQNAgent


def test_update_changes_qnet_parameters():
    agent = DQNAgent(batch_size=4, buffer_size=64, device="cpu", log_device=False)
    initial_params = [param.detach().clone() for param in agent.qnet.parameters()]

    for i in range(10):
        state = np.array([i * 0.01, i * 0.02], dtype=np.float32)
        action = i % agent.action_space
        reward = 1.0
        next_state = state + 0.05
        done = False
        agent.update(state, action, reward, next_state, done)

    updated_params = [param.detach().clone() for param in agent.qnet.parameters()]
    changed = any(
        not torch.allclose(before, after)
        for before, after in zip(initial_params, updated_params, strict=True)
    )

    assert changed


def test_sync_qnet_copies_weights_to_target_network():
    agent = DQNAgent(device="cpu", log_device=False)
    with torch.no_grad():
        for param in agent.qnet.parameters():
            param.add_(0.5)

    agent.sync_qnet()

    for source, target in zip(
        agent.qnet.parameters(), agent.target_qnet.parameters(), strict=True
    ):
        assert torch.allclose(source, target)
