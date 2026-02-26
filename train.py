import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from agent import DQNAgent


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 1000
    trials: int = 100
    sync_interval: int = 20
    env_id: str = "MountainCar-v0"
    render_mode: str | None = "rgb_array"
    epsilon: float = 0.1
    gamma: float = 0.99
    lr: float = 0.0005
    buffer_size: int = 10000
    batch_size: int = 32
    action_space: int = 3
    reward_shaping_scale: float = 10.0
    reward_shaping_offset: float = 0.5
    device: str | None = None
    seed: int | None = None
    log_device: bool = True
    log_progress: bool = True


@dataclass(frozen=True)
class TrainingResult:
    reward_histories: list[list[float]]
    reward_means: list[float]
    elapsed_sec: float


def run_training(config: TrainingConfig) -> TrainingResult:
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    env_kwargs = {}
    if config.render_mode is not None:
        env_kwargs["render_mode"] = config.render_mode

    env = gym.make(config.env_id, **env_kwargs)
    start = time.time()
    reward_histories: list[list[float]] = []

    try:
        for trial in range(config.trials):
            agent = DQNAgent(
                epsilon=config.epsilon,
                gamma=config.gamma,
                lr=config.lr,
                buffer_size=config.buffer_size,
                batch_size=config.batch_size,
                action_space=config.action_space,
                device=config.device,
                log_device=config.log_device,
            )
            reward_history: list[float] = []

            for episode in range(config.episodes):
                if config.seed is not None:
                    episode_seed = config.seed + (trial * config.episodes) + episode
                    state, _ = env.reset(seed=episode_seed)
                else:
                    state, _ = env.reset()
                done = False
                total_reward = 0.0

                while not done:
                    action = agent.get_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    origin_reward = reward
                    reward += abs(
                        (next_state[0] + config.reward_shaping_offset) * next_state[1]
                    ) * config.reward_shaping_scale
                    done = terminated or truncated

                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += origin_reward

                reward_history.append(float(total_reward))
                if episode % config.sync_interval == 0:
                    agent.sync_qnet()

            reward_histories.append(reward_history)
            if config.log_progress:
                print(trial)
    finally:
        env.close()

    elapsed_sec = time.time() - start
    reward_means: list[float] = []
    if reward_histories and reward_histories[0]:
        reward_means = [sum(col) / len(col) for col in zip(*reward_histories)]

    return TrainingResult(
        reward_histories=reward_histories,
        reward_means=reward_means,
        elapsed_sec=elapsed_sec,
    )
