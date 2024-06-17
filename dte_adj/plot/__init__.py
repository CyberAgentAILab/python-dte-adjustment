import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axis as axis
from typing import Optional


def plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    upper_bands: np.ndarray,
    lower_bands: np.ndarray,
    ax: Optional[axis.Axis] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    """Visualize distributional parameters and their confidence intervals.

    Args:
        x_values (np.Array): values to be used for x axis.
        y_values (np.Array): Expected distributional parameters.
        upper_bands (np.Array): Upper band for the distributional parameters.
        lower_bands (np.Array): Lower band for the distributional parameters.
        ax (matplotlib.axes.Axes, optional): Target axes instance. If None, a new figure and axes will be created.
        title (str, optional): Axes title.
        xlabel (str, optional): X-axis title label.
        ylabel (str, optional): Y-axis title label.

    Returns:
        matplotlib.axes.Axes: The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x_values, y_values, label="Values", color="blue")
    ax.fill_between(
        x_values,
        lower_bands,
        upper_bands,
        color="gray",
        alpha=0.3,
        label="Confidence Interval",
    )

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.legend()

    return ax
