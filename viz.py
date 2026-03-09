from pathlib import Path

import matplotlib.pyplot as plt


def save_reward_plot(
    x_values: list[float],
    raw_values: list[float],
    smoothed_values: list[float],
    path: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(x_values, raw_values, label="Raw Eval Reward", alpha=0.45)
    plt.plot(x_values, smoothed_values, label="Smoothed Eval Reward", linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
