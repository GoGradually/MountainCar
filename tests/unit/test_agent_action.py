import numpy as np
import pytest
import torch

from agent import AgentConfig, DQNAgent


def test_epsilon_linearly_decays_and_clamps():
    config = AgentConfig(
        n_timesteps=100,
        exploration_fraction=0.5,
        eps_start=1.0,
        eps_final=0.1,
    )
    agent = DQNAgent(config=config, device="cpu", log_device=False)

    assert agent.exploration_steps == 50
    assert agent.epsilon() == pytest.approx(1.0)

    agent.global_step = 25
    assert agent.epsilon() == pytest.approx(0.55)

    agent.global_step = 50
    assert agent.epsilon() == pytest.approx(0.1)

    agent.global_step = 80
    assert agent.epsilon() == pytest.approx(0.1)


def test_get_action_greedy_after_exploration_window():
    config = AgentConfig(
        n_timesteps=10,
        exploration_fraction=0.5,
        eps_start=1.0,
        eps_final=0.0,
    )
    agent = DQNAgent(config=config, device="cpu", log_device=False)
    agent.global_step = agent.exploration_steps

    with torch.no_grad():
        for param in agent.qnet.parameters():
            param.zero_()
        agent.qnet.l3.bias.copy_(torch.tensor([0.0, 1.0, 2.0]))

    action = agent.get_action(np.array([0.0, 0.0], dtype=np.float32))

    assert action == 2


def test_get_action_exploration_samples_valid_actions():
    np.random.seed(0)
    config = AgentConfig(eps_start=1.0, eps_final=1.0)
    agent = DQNAgent(config=config, device="cpu", log_device=False)
    actions = [agent.get_action(np.array([0.0, 0.0], dtype=np.float32)) for _ in range(200)]

    assert all(0 <= action < agent.action_space for action in actions)
    assert len(set(actions)) >= 2
