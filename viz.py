from pathlib import Path

import matplotlib.pyplot as plt


def save_reward_plot(reward_means: list[float], path: str, title: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(reward_means)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
