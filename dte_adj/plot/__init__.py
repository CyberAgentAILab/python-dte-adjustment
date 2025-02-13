import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axis as axis
from typing import Optional


def plot(
    X: np.ndarray,
    means: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    chart_type: str = "line",
    color: str = "green",
    ax: Optional[axis.Axis] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    """Visualize distributional parameters and their confidence intervals.

    Args:
        X (np.Array): values to be used for x axis.
        means (np.Array): Expected distributional parameters.
        lower_bounds (np.Array): Lower bound for the distributional parameters.
        upper_bounds (np.Array): Upper bound for the distributional parameters.
        chart_type (str): Chart type of the plotting. Available values are line or bar.
        color (str): The color of lines or bars.
        ax (matplotlib.axes.Axes, optional): Target axes instance. If None, a new figure and axes will be created.
        title (str, optional): Axes title.
        xlabel (str, optional): X-axis title label.
        ylabel (str, optional): Y-axis title label.

    Returns:
        matplotlib.axes.Axes: The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if chart_type == "line":
        ax.plot(X, means, label="Values", color=color)
        ax.fill_between(
            X,
            lower_bounds,
            upper_bounds,
            color=color,
            alpha=0.3,
            label="Confidence Interval",
        )
    elif chart_type == "bar":
        ax.bar(
            X,
            means,
            yerr=[
                np.maximum(means - lower_bounds, 0),
                np.maximum(upper_bounds - means, 0),
            ],
            capsize=5,
            color=color,
            width=(X.max() - X.min()) / len(X),
        )
    else:
        raise ValueError(f"Chart type {chart_type} is not supported")

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return ax
