import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class AgentConfig:
    gamma: float = 0.98
    lr: float = 0.004
    buffer_size: int = 10_000
    batch_size: int = 128
    action_space: int = 3
    hidden_dim: int = 256
    n_timesteps: int = 120_000
    train_start: int = 1_000
    train_freq: int = 16
    gradient_steps: int = 8
    max_grad_norm: float = 10.0
    eps_start: float = 1.0
    eps_final: float = 0.07
    exploration_fraction: float = 0.2
    target_sync_every: int = 600


class QNet(nn.Module):
    def __init__(self, action_size: int, hidden_dim: int = 256):
        super().__init__()
        self.action_size = action_size
        self.l1 = nn.Linear(2, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        device: str | None = None,
        log_device: bool = True,
    ):
        self.config = config or AgentConfig()
        self.gamma = self.config.gamma
        self.lr = self.config.lr
        self.buffer_size = self.config.buffer_size
        self.batch_size = self.config.batch_size
        self.action_space = self.config.action_space
        self.n_timesteps = self.config.n_timesteps
        self.train_start = self.config.train_start
        self.train_freq = self.config.train_freq
        self.gradient_steps = self.config.gradient_steps
        self.max_grad_norm = self.config.max_grad_norm
        self.eps_start = self.config.eps_start
        self.eps_final = self.config.eps_final
        self.exploration_fraction = self.config.exploration_fraction
        self.target_sync_every = self.config.target_sync_every
        self.exploration_steps = int(self.n_timesteps * self.exploration_fraction)
        self.global_step = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if log_device:
            print(f"Using device: {self.device}")

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_space, hidden_dim=self.config.hidden_dim).to(self.device)
        self.target_qnet = QNet(self.action_space, hidden_dim=self.config.hidden_dim).to(
            self.device
        )
        self.sync_qnet()
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

    def epsilon(self) -> float:
        if self.global_step >= self.exploration_steps:
            return self.eps_final
        fraction = self.global_step / max(1, self.exploration_steps)
        return self.eps_start + (self.eps_final - self.eps_start) * fraction

    def sync_qnet(self) -> None:
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def _next_state_values(self, next_state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.target_qnet(next_state).max(dim=1).values

    def get_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon():
            return int(np.random.choice(self.action_space))
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.qnet(s)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self, state, action, reward, next_state, done) -> None:
        self.global_step += 1
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.train_start:
            return
        if self.global_step % self.train_freq != 0:
            return

        for _ in range(self.gradient_steps):
            state, action, reward, next_state, done = self.replay_buffer.get_batch()

            s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            a = torch.as_tensor(action, dtype=torch.int64, device=self.device)
            r = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            ns = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
            d = torch.as_tensor(done, dtype=torch.float32, device=self.device)

            q_values = self.qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = self._next_state_values(ns)
                target = r + (1 - d) * self.gamma * next_q

            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss_fn(q_values, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.max_grad_norm)
            self.optimizer.step()

        if self.global_step % self.target_sync_every == 0:
            self.sync_qnet()


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
