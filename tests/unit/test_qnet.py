import torch

from agent import QNet


def test_qnet_forward_single_state_shape():
    qnet = QNet(action_size=3)
    state = torch.tensor([0.1, -0.2], dtype=torch.float32)

    output = qnet(state)

    assert output.shape == (1, 3)


def test_qnet_forward_batch_shape():
    qnet = QNet(action_size=3)
    batch = torch.randn(5, 2)

    output = qnet(batch)

    assert output.shape == (5, 3)
