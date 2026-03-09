import time
import random
from dataclasses import dataclass, field
from itertools import zip_longest

import gymnasium as gym
import numpy as np
import torch

from agent import AgentConfig, DQNAgent


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 120_000
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
    timesteps_per_trial: list[int]
    elapsed_sec: float


def _mean_reward_histories(reward_histories: list[list[float]]) -> list[float]:
    reward_means: list[float] = []
    for col in zip_longest(*reward_histories, fillvalue=None):
        values = [value for value in col if value is not None]
        if values:
            reward_means.append(sum(values) / len(values))
    return reward_means


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
    timesteps_per_trial: list[int] = []

    try:
        for trial in range(config.trials):
            agent = DQNAgent(
                total_timesteps=config.total_timesteps,
                config=config.agent_config,
                device=config.device,
                log_device=config.log_device,
            )
            reward_history: list[float] = []
            total_steps = 0
            episode = 0

            while total_steps < config.total_timesteps:
                if config.seed is not None:
                    episode_seed = config.seed + (trial * config.total_timesteps) + episode
                    state, _ = env.reset(seed=episode_seed)
                else:
                    state, _ = env.reset()
                done = False
                total_reward = 0.0

                while not done and total_steps < config.total_timesteps:
                    action = agent.get_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    total_steps += 1
                    done = terminated or truncated

                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += float(reward)

                if not done:
                    break

                reward_history.append(float(total_reward))
                episode += 1

            reward_histories.append(reward_history)
            timesteps_per_trial.append(total_steps)
            if config.log_progress:
                print(trial)
    finally:
        env.close()

    elapsed_sec = time.time() - start
    reward_means = _mean_reward_histories(reward_histories)

    return TrainingResult(
        reward_histories=reward_histories,
        reward_means=reward_means,
        timesteps_per_trial=timesteps_per_trial,
        elapsed_sec=elapsed_sec,
    )
