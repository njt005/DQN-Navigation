import typing as t
import numpy as np
import matplotlib.pyplot as plt


def moving_average(a: np.ndarray, n: int) -> np.ndarray:
    """moving_average
    Calculates the moving average of the input array

    Parameters
    ----------
    a : np.ndarray
    n : int
        moving average window length

    Returns
    -------
    np.ndarray
        Moving average of input array
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def create_reward_plot(
    scores: np.ndarray,
    smooth_n: t.Optional[int] = None,
    save_path: t.Optional[str] = None,
) -> None:
    """create_reward_plot
    Create plot of rewards during training

    Parameters
    ----------
    scores : np.ndarray
        rewards that agent received
    smooth_n : t.Optional[int], optional
        moving average bin length to smooth curve
    save_path : t.Optional[str], optional
        path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
    episodes = np.arange(len(scores))
    ax.plot(episodes, scores, c="k", lw=0.5)

    if smooth_n is not None:
        scores = moving_average(scores, n=smooth_n)
        episodes = np.arange(len(scores))
        ax.plot(episodes, scores, c="#6699cc", lw=1)

    ax.set_ylabel("Score")
    ax.set_xlabel("Episode")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
