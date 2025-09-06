import random
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.action_size = action_size
        self.l1 = nn.Linear(2, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.epsilon = 0.1
        self.gamma = 0.99
        self.lr = 0.0005
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_space = 3

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_space).to(self.device)
        self.target_qnet = QNet(self.action_space).to(self.device)
        self.sync_qnet()
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def sync_qnet(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.qnet(s)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        s  = torch.as_tensor(state,      dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(action,     dtype=torch.int64,   device=self.device)
        r  = torch.as_tensor(reward,     dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(done,       dtype=torch.float32, device=self.device)


        qs = self.qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_qs = self.target_qnet(ns)
            next_q = next_qs.max(dim=1)[0]
            target = r + (1 - d) * self.gamma * next_q


        self.qnet.zero_grad(set_to_none=True)
        loss = nn.MSELoss()(qs, target)
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def get_batch(self):
        samples = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in samples])
        action = np.array([x[1] for x in samples])
        reward = np.array([x[2] for x in samples])
        next_state = np.stack([x[3] for x in samples])
        done = np.array([x[4] for x in samples]).astype(np.int32)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
