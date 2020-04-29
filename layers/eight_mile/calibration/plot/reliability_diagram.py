from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


__all__ = ["reliability_diagram", "reliability_curve"]


def reliability_diagram(
    accs: np.array,
    confs: np.array,
    left_bins: np.array,
    num_classes: Optional[int] = None,
    x_ticks: Optional[np.array] = None,
    y_ticks: Optional[np.array] = None,
    title: Optional[str] = "Reliability Graph",
    y_label: Optional[str] = "Accuracy",
    x_label: Optional[str] = "Confidence",
    gap_label: Optional[str] = "Gap",
    output_label: Optional[str] = "Outputs",
    gap_edge: str = "r",
    gap_color: str = "r",
    gap_alpha: float = 0.25,
    gap_hatch: str = "/",
    output_edge: str = "k",
    output_color: str = "b",
    fig_size: int = 3,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
) -> Tuple[Figure, Axes]:
    """Plot a reliability graph a la https://arxiv.org/abs/1706.04599

    :param accs: The average accuracy for examples in that bin
    :param congs: The average confidence for examples in that bin
    :param left_bins: The bins that define the groups, they should specify the left
        edge of the bin
    :param x_ticks: Where to show the ticks along the x axis. If not provided it
        will display every other tick, pass an empty list to skip adding ticks
    :param y_ticks: Where to show the ticks along the y axis. If not provided it
        uses default matplotlib ticks, padd an empty list to skip adding ticks
    :param title: The title for the axes, pass None to skip adding a title to the axes
    :param y_label: A label for the y axes, pass None to skip adding a label
    :param x_label: A label for the x axes, pass None to skip adding a label
    :param gap_label: A label for the gap bins, pass None to skip adding a label
    :param output_label: A label for the output bins, pass None to skip adding a label
    :param gap_edge: The color to use for the edges of the gap bins
    :param gap_color: The color to use for the main section of the gap bins
    :param gap_hatch: The hatching pattern for the gap, useful when a bin a under confident
    :param output_edge: The color to use for the edges of the output bins
    :param output_color: The color to use for the main section of the output bins
    :param fig_size: The figure size, only used if creating the figure/axis from scratch
    :param ax: An Axes object that we can create the graph in, if None it will be created
    :param fig: A figure that holds the axes object

    :returns: `Tuple(fig, ax)` If the figure and ax are passed in they are returned, if they
        were not passed in the created ones are returned. The don't use the figure at all but
        because we want to return both in case we created them.
    """
    # If neither a figure non-axes are passed in we create them.
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
    expected = left_bins + (widths / 2)
    expected = np.where(confs != 0, confs, expected)
    if num_classes is not None:
        # This is the minimum possible confidence. This is maximum entropy. If your confidence is
        # lower than this and something else has to have a larger confidence
        min_conf = 1.0 / num_classes
        # These bins are lowest confidence allowed in the bin so the highest confidence is the left edge
        # of the bin above it. So if we do a compare to the bins we will see that the first bin will always
        # be false. But we can use a np.roll and force the last bin to be true to simulate as if we compared
        # to the right bin
        possible_mask = np.roll(left_bins > min_conf, shift=-1)
        possible_mask[-1] = True
        # Mask out the expected values for bins that we can't put values in. Otherwise we will see all these
        # red bars we can't do anything about.
        expected *= possible_mask
    ax.bar(
        left_bins,
        accs,
        align='edge',
        width=widths,
        label=output_label,
        edgecolor=output_edge,
        color=output_color
    )
    ax.bar(
        left_bins,
        expected - accs,
        bottom=accs,
        align='edge',
        width=widths,
        label=gap_label,
        edgecolor=gap_edge,
        color=gap_color,
        alpha=gap_alpha,
        hatch=gap_hatch,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if x_ticks is None:
        x_ticks = [y for i, y in enumerate(left_bins) if i % 2 == 0] + [1.0]
    ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if not (gap_label is None and output_label is None):
        ax.legend()
    return fig, ax


def reliability_curve(
    accs: np.array,
    confs: np.array,
    num_classes: Optional[int] = None,
    x_ticks: Optional[np.array] = None,
    y_ticks: Optional[np.array] = None,
    title: Optional[str] = "Reliability Curve",
    y_label: Optional[str] = "Accuracy",
    x_label: Optional[str] = "Confidence",
    label: Optional[str] = None,
    color: str = "tab:orange",
    line_style: str = "o-",
    fig_size: int = 3,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
) -> Tuple[Figure, Axes]:
    """Plot a reliability curve a la https://scikit-learn.org/stable/modules/calibration.html#calibration

    Note:
        When there is a bin that had no samples in it the average confidence will be zero
        and the average accuracy will be zero too. If this is the case we skip plotting
        that point.

    :param confs: The average confidence in a bin
    :param accs: The average accuracy for examples in that bin
    :param x_ticks: Where to show the ticks along the x axis. If not provided it
        will display every other tick, pass an empty list to skip adding ticks
    :param y_ticks: Where to show the ticks along the y axis. If not provided it
        uses default matplotlib ticks, padd an empty list to skip adding ticks
    :param title: The title for the axes, pass None to skip adding a title to the axes
    :param y_label: A label for the y axes, pass None to skip adding a label
    :param x_label: A label for the x axes, pass None to skip adding a label
    :param label: A label for the plot we are making.
    :param color: A color for the line we are making.
    :param line_style: The stle of line to draw.
    :param fig_size: The figure size, only used if creating the figure/axis from scratch
    :param ax: An Axes object that we can create the graph in, if None it will be created
    :param fig: A figure that holds the axes object

    :returns: `Tuple(fig, ax)` If the figure and ax are passed in they are returned, if they
        were not passed in the created ones are returned. The don't use the figure at all but
        because we want to return both in case we created them.
    """
    # If neither a figure non-axes are passed in we create them.
    if ax is None and fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
    # If the axes was not passed in but the figure was we get the current active
    # axes for the figure.
    if ax is None and fig is not None:
        ax = fig.gca()
    # If the axes is provided but the figure we look up the figure on the axes
    if ax is not None and fig is None:
        fig = ax.figure
    valid = (confs != 0) | (accs != 0)
    ax.plot(
        confs[valid], accs[valid], line_style, color=color, label=label
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if x_ticks is None:
        x_ticks = [y for i, y in enumerate(np.arange(0, 1 + 1e-8, 1. / len(confs))) if i % 2 == 0]
    ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if label is not None:
        ax.legend()
    return fig, ax


def _demo():
    f, axs = plt.subplots(2, 2, figsize=(4, 4))
    data = np.array([0, 0, .15, .25, .28, .40, .38, .42, .56, .84])
    x = np.arange(0, 1, 1 / len(data))
    reliability_diagram(data, x + 0.05, x, num_classes=10, ax=axs[0][0], title="Reliability Diagram (10 classes)")
    data[5] = 0
    reliability_diagram(data, x + 0.05, x, num_classes=100, ax=axs[0][1], title="Reliability Diagram (100 classes)")
    conf = np.array([
        0.07167232, 0.16942227, 0.26184   , 0.35503329, 0.44979852,
        0.55041263, 0.64713617, 0.73971958, 0.83297067, 0.92842073
    ])
    acc = np.array([
        0.        , 0.00589623, 0.00789084, 0.03013831, 0.18707902,
        0.64843534, 0.92417131, 0.98329403, 0.98977505, 1.
    ])
    reliability_curve(acc, conf, label="SVC", ax=axs[1][0])
    conf = np.array([
        0.07167232, 0.16942227, 0.26184   , 0.35503329, 0.44979852,
        0.55041263, 0.64713617, 0, 0.83297067, 0.92842073
    ])
    acc = np.array([
        0.        , 0.00589623, 0.00789084, 0.03013831, 0.18707902,
        0.64843534, 0.92417131, 0, 0.98977505, 1.
    ])
    reliability_curve(acc * 0.5, conf, label="NB", ax=axs[1][1], color="b")
    reliability_curve(acc, conf, label="SVC", ax=axs[1][1], title="Reliability Curve, Missing bins")
    plt.show()


if __name__ == "__main__":
    _demo()
