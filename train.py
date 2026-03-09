import time
import random
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch

from agent import AgentConfig, DQNAgent


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 1200
    trials: int = 3
    env_id: str = "MountainCar-v0"
    render_mode: str | None = None
    agent_config: AgentConfig = field(default_factory=AgentConfig)
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
        random.seed(config.seed)
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
                config=config.agent_config,
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
                    done = terminated or truncated

                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += float(reward)

                reward_history.append(float(total_reward))

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
