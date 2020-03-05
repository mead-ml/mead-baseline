from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


__all__ = ["confidence_histogram"]


def confidence_histogram(
    left_bins: np.array,
    bin_counts: np.array,
    acc: Optional[float] = None,
    avg_conf: Optional[float] = None,
    x_ticks: Optional[np.array] = None,
    y_ticks: Optional[np.array] = None,
    title: Optional[str] = "Confidence Distribution",
    y_label: Optional[str] = "% of Samples",
    x_label: Optional[str] = "Confidence",
    conf_label: Optional[str] = "Avg. Confidence",
    acc_label: Optional[str] = "Accuracy",
    label_y: float = 0.4,
    conf_spacer: float = 0.015,
    acc_spacer: float = -0.07,
    edge: str = "k",
    color: str = "b",
    fig_size: int = 3,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
) -> Tuple[Figure, Axes]:
    """Plot a reliability graph a la https://arxiv.org/abs/1706.04599

    :param left_bins: The bins that define the groups, they should specify the left
        edge of the bin
    :param accs: The average accuracy for examples in that bin
    :param x_ticks: Where to show the ticks along the x axis. If not provided it
        will display every other tick, pass an empty list to skip adding ticks
    :param y_ticks: Where to show the ticks along the y axis. If not provided it
        uses default matplotlib ticks, padd an empty list to skip adding ticks
    :param title: The title for the axes, pass None to skip adding a title to the axes
    :param y_label: A label for the y axes, pass None to skip adding a label
    :param x_label: A label for the x axes, pass None to skip adding a label
    :param conf_label: A label to put on the line representing the average confidence
    :param acc_label: A label to put on the line representing the accuract
    :param label_y: Where to start the above labels alon the y axis, 0 is the bottom, 1 is the top
    :param conf_spacer: How far off the line to put the confidence label, use positive to move
        right of the line and negative to move to the left
    :param acc_spacer: How far off the line to put the accuracy label, use positive to move right
        of the line and negative to move to the left
    :param edge: The color to use for the edges of the output bins
    :param color: The color to use for the main section of the output bins
    :param fig_size: The figure size, only used if creating the figure/axis from scratch
    :param ax: An Axes object that we can create the graph in, if None it will be created
    :param fig: A figure that holds the axes object

    :returns: `Tuple(fig, ax)` If the figure and ax are passed in they are returned, if they
        were not passed in the created ones are returned. The don't use the figure at all but
        because we want to return both in case we created them.
    """
    # If niether a figure non-axes are passed in we create them.
    if ax is None and fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
    # If the axes was not passed in but the figure was we get the current active
    # axes for the figure.
    if ax is None and fig is not None:
        ax = fig.gca()
    # If the axes is provided but the figure we look up the figure on the axes
    if ax is not None and fig is None:
        fig = ax.figure
    widths = np.diff(left_bins, append=1)
    ax.bar(
        left_bins,
        bin_counts / np.sum(bin_counts),
        align='edge',
        width=widths,
        edgecolor=edge,
        color=color
    )
    trans = ax.get_xaxis_transform()
    if avg_conf is not None:
        ax.axvline(avg_conf, ls='--', c='.3')
        if conf_label is not None:
            ax.text(avg_conf + conf_spacer, label_y, conf_label, rotation=90, transform=trans)
    if acc is not None:
        ax.axvline(acc, ls='--', c='.3')
        if acc_label is not None:
            ax.text(acc + acc_spacer, label_y, acc_label, rotation=90, transform=trans)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if x_ticks is None:
        x_ticks = [y for i, y in enumerate(left_bins) if i % 2 == 0] + [1.0]
    ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    return fig, ax


def _demo():
    small_bins = np.arange(0, 1, step=0.05)
    counts = np.zeros_like(small_bins)
    prev = 0
    for i in range(len(small_bins)):
        add = np.random.randint(10, 15 * (i + 1))
        counts[i] = prev + add
        prev = counts[i]
    counts[-1] += np.random.randint(50, 1000)
    x_ticks = np.arange(0, 1.1, step=0.2)
    ACC = 0.72
    CONF = 0.84

    f, a = confidence_histogram(small_bins, counts, x_ticks=x_ticks, acc=ACC, avg_conf=CONF)
    plt.show()


if __name__ == "__main__":
    _demo()
