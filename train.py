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
    eval_freq: int = 2_000
    n_eval_episodes: int = 20
    eval_window: int = 5
    eval_seed: int = 1_000_000
    log_device: bool = True
    log_progress: bool = True


@dataclass(frozen=True)
class TrainingResult:
    reward_histories: list[list[float]]
    reward_means: list[float]
    timesteps_per_trial: list[int]
    eval_timesteps: list[int]
    eval_reward_raw: list[float]
    eval_reward_smoothed: list[float]
    elapsed_sec: float


def _mean_reward_histories(reward_histories: list[list[float]]) -> list[float]:
    reward_means: list[float] = []
    for col in zip_longest(*reward_histories, fillvalue=None):
        values = [value for value in col if value is not None]
        if values:
            reward_means.append(sum(values) / len(values))
    return reward_means


def _aggregate_eval_histories(
    eval_histories: list[list[tuple[int, float]]],
) -> tuple[list[int], list[float]]:
    rewards_by_timestep: dict[int, list[float]] = {}
    for history in eval_histories:
        for timestep, reward in history:
            rewards_by_timestep.setdefault(timestep, []).append(reward)

    eval_timesteps = sorted(rewards_by_timestep)
    eval_reward_means = [
        sum(rewards_by_timestep[timestep]) / len(rewards_by_timestep[timestep])
        for timestep in eval_timesteps
    ]
    return eval_timesteps, eval_reward_means


def _smooth_rewards(rewards: list[float], window: int) -> list[float]:
    smoothed: list[float] = []
    for index in range(len(rewards)):
        start = max(0, index - window + 1)
        current_window = rewards[start : index + 1]
        smoothed.append(sum(current_window) / len(current_window))
    return smoothed


def _evaluate_agent(
    config: TrainingConfig,
    agent: DQNAgent,
    eval_rng: np.random.Generator,
) -> float:
    eval_env = gym.make(config.env_id)
    rewards: list[float] = []
    try:
        for _ in range(config.n_eval_episodes):
            eval_seed = int(eval_rng.integers(0, np.iinfo(np.int32).max))
            state, _ = eval_env.reset(seed=eval_seed)

            done = False
            total_reward = 0.0
            while not done:
                action = agent.get_greedy_action(state)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += float(reward)

            rewards.append(total_reward)
    finally:
        eval_env.close()

    return float(np.mean(rewards)) if rewards else 0.0


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
    eval_histories: list[list[tuple[int, float]]] = []

    try:
        for trial in range(config.trials):
            agent = DQNAgent(
                total_timesteps=config.total_timesteps,
                config=config.agent_config,
                device=config.device,
                log_device=config.log_device,
            )
            eval_rng = np.random.default_rng(config.eval_seed + trial)
            reward_history: list[float] = []
            eval_history: list[tuple[int, float]] = []
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
                    if total_steps % config.eval_freq == 0:
                        eval_reward = _evaluate_agent(config, agent, eval_rng)
                        eval_history.append((total_steps, eval_reward))
                    state = next_state
                    total_reward += float(reward)

                if not done:
                    break

                reward_history.append(float(total_reward))
                episode += 1

            reward_histories.append(reward_history)
            timesteps_per_trial.append(total_steps)
            eval_histories.append(eval_history)
            if config.log_progress:
                print(trial)
    finally:
        env.close()

    elapsed_sec = time.time() - start
    reward_means = _mean_reward_histories(reward_histories)
    eval_timesteps, eval_reward_raw = _aggregate_eval_histories(eval_histories)
    eval_reward_smoothed = _smooth_rewards(eval_reward_raw, config.eval_window)

    return TrainingResult(
        reward_histories=reward_histories,
        reward_means=reward_means,
        timesteps_per_trial=timesteps_per_trial,
        eval_timesteps=eval_timesteps,
        eval_reward_raw=eval_reward_raw,
        eval_reward_smoothed=eval_reward_smoothed,
        elapsed_sec=elapsed_sec,
    )
