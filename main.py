import gymnasium as gym
import matplotlib.pyplot as plt
from agent import DQNAgent

episodes = 300
sync_interval = 20
env = gym.make("MountainCar-v0", render_mode="human")
agent = DQNAgent()
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        reward += abs(next_state[0] * (next_state[1] + 0.5))
        done = terminated | truncated

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    reward_history.append(total_reward)
    if episode % sync_interval == 0:
        agent.sync_qnet()

plt.plot(reward_history)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.show()