import gymnasium as gym
import matplotlib.pyplot as plt
from agent import DQNAgent
import time

start = time.time()

episodes = 1000
sync_interval = 20
env = gym.make("MountainCar-v0", render_mode="rgb_array")
reward_histories = []
for trial in range(100):
    agent = DQNAgent()
    reward_history = []
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            origin_reward = reward
            reward += abs((next_state[0] + 0.5) * next_state[1])* 10
            done = terminated | truncated

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += origin_reward
        reward_history.append(total_reward)
        if episode % sync_interval == 0:
            agent.sync_qnet()
    reward_histories.append(reward_history)
    print(trial)


end = time.time()

print(f"실행 시간: {end - start:.4f}초")

reward_means = [sum(col)/len(col) for col in zip(*reward_histories)]

plt.plot(reward_means)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.show()