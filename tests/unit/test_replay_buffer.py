import numpy as np

from agent import ReplayBuffer


def test_replay_buffer_batch_shapes_and_types():
    buffer = ReplayBuffer(buffer_size=20, batch_size=4)
    for i in range(10):
        state = np.array([i * 0.1, i * 0.2], dtype=np.float32)
        action = i % 3
        reward = float(i)
        next_state = state + 0.01
        done = i % 2 == 0
        buffer.add(state, action, reward, next_state, done)

    state, action, reward, next_state, done = buffer.get_batch()

    assert state.shape == (4, 2)
    assert action.shape == (4,)
    assert reward.shape == (4,)
    assert next_state.shape == (4, 2)
    assert done.shape == (4,)
    assert done.dtype == np.int32
