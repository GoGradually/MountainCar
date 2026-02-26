from viz import save_reward_plot


def test_save_reward_plot_creates_png(tmp_path):
    plot_path = tmp_path / "reward.png"

    save_reward_plot([-200.0, -180.0, -150.0], str(plot_path), "Quick Check")

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
