import numpy as np
import torch

from agent import DQNAgent


def test_get_action_greedy_when_epsilon_zero():
    agent = DQNAgent(epsilon=0.0, device="cpu", log_device=False)
    with torch.no_grad():
        for param in agent.qnet.parameters():
            param.zero_()
        agent.qnet.l3.bias.copy_(torch.tensor([0.0, 1.0, 2.0]))

    action = agent.get_action(np.array([0.0, 0.0], dtype=np.float32))

    assert action == 2


def test_get_action_exploration_when_epsilon_one():
    np.random.seed(0)
    agent = DQNAgent(epsilon=1.0, device="cpu", log_device=False)
    actions = [agent.get_action(np.array([0.0, 0.0], dtype=np.float32)) for _ in range(200)]

    assert all(0 <= action < agent.action_space for action in actions)
    assert len(set(actions)) >= 2
