# Author: Sylvain Estebe

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from pyhgf.plots.graphviz.plot_nodes import plot_nodes

if TYPE_CHECKING:
    from pyhgf.model import Network


def plot_trajectories_sim(
    network: "Network",
    ci: bool = True,
    show_surprise: bool = True,
    show_posterior: bool = False,
    show_total_surprise: bool = False,
    figsize: Tuple[int, int] = (18, 9),
    axs: Optional[Union[List, Axes]] = None,
) -> Axes:
    """Plot simulation trajectories for nodes.

    After drawing the nodes via `plot_nodes`, this function extracts prediction
    trajectories from the network's predictions and overlays them on the plot.
    Each trajectory for each node is stored in a DataFrame column named
    "trajectory_<traj>_node_<node>".

    Parameters
    ----------
    network : Network
        Main Network instance.
    ci : bool, default True
        Display the confidence interval.
    show_surprise : bool, default True
        Plot the surprise for each node.
    show_posterior : bool, default False
        Overlay posterior (mean and precision) curves.
    show_total_surprise : bool, default False
        Plot total surprise in a separate panel.
    figsize : tuple of int, default (18, 9)
        Figure size.
    axs : list or Axes, optional
        Axes on which to plot. A new figure is created if None.

    Returns
    -------
    Axes
        Matplotlib axes used for plotting.

    """
    # Get a DataFrame with additional information, if any
    trajectories_df = network.to_pandas()

    # Create a DataFrame for the prediction trajectories
    df_list = []
    for node in range(network.n_nodes):
        # Get the "mean" predictions for the current node
        data = network.predictions[node]["mean"]
        # Iterate over each trajectory (row in the data)
        for traj in range(data.shape[0]):
            col_name = f"trajectory_{traj}_node_{node}"
            df_list.append(pd.DataFrame({col_name: data[traj, :]}))
    df_predictions = pd.concat(df_list, axis=1)
    trajectories_sim = df_predictions

    # Number of nodes based on the network's structure
    n_nodes = len(network.edges)

    # Create or get the axes
    if axs is None:
        n_rows = n_nodes + 1 if show_total_surprise else n_nodes
        fig, axs = plt.subplots(nrows=n_rows, figsize=figsize, sharex=True)
        if n_nodes == 1:
            axs = [axs]

    # 1. Plot nodes using plot_nodes (reverse axes order so node 0 is at the bottom)
    ax_i = n_nodes - 1
    for node_idx in range(n_nodes):
        if node_idx in network.input_idxs:
            _show_posterior = True
            color = "#4c72b0"
        else:
            _show_posterior = show_posterior
            color = (
                "#55a868"
                if (network.edges[node_idx].volatility_children is None)
                else "#c44e52"
            )

        plot_nodes(
            network=network,
            node_idxs=node_idx,
            axs=axs[ax_i],
            color=color,
            show_surprise=show_surprise,
            show_posterior=_show_posterior,
            ci=ci,
        )
        ax_i -= 1

    # 2. Overlay simulation trajectories (drawn on top)
    time_offset = len(
        trajectories_df
    )  # Alternatively, use trajectories_df.index.max() + 1 if needed

    for node in range(network.n_nodes):
        ax_idx = n_nodes - node - 1
        # Select columns for the current node
        node_cols = [col for col in trajectories_sim.columns if f"node_{node}" in col]
        for col in node_cols:
            shifted_index = trajectories_sim.index + time_offset
            axs[ax_idx].plot(
                shifted_index,
                trajectories_sim[col],
                ls="-",
                color="#2a2a2a",
                linewidth=1,
                zorder=2,
            )

    # 3. Plot total surprise if requested
    if show_total_surprise:
        surprise_ax = axs[n_nodes].twinx()
        surprise_ax.fill_between(
            x=trajectories_df.time,
            y1=trajectories_df.total_surprise,
            y2=trajectories_df.total_surprise.min(),
            label="Surprise",
            color="#7f7f7f",
            alpha=0.2,
        )
        surprise_ax.plot(
            trajectories_df.time,
            trajectories_df.total_surprise,
            color="#2a2a2a",
            linewidth=0.5,
            zorder=-1,
            label="Surprise",
        )
        sp = trajectories_df.total_surprise.sum()
        surprise_ax.set_title(f"Total surprise: {sp:.2f}", loc="right")
        surprise_ax.set_ylabel("Surprise")

    axs[n_nodes - 1].set_xlabel("Time")

    return axs
