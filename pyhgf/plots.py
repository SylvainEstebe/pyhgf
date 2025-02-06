# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from graphviz.sources import Source

    from pyhgf.model import Network


def plot_trajectories(
    network: "Network",
    ci: bool = True,
    show_surprise: bool = True,
    show_posterior: bool = False,
    show_total_surprise: bool = False,
    figsize: Tuple[int, int] = (18, 9),
    axs: Optional[Union[List, Axes]] = None,
) -> Axes:
    r"""Plot the trajectories of the nodes' sufficient statistics and surprise.

    This function will plot the expected mean and precision (converted into standard
    deviation) and the surprise at each level of the node structure.

    Parameters
    ----------
    network :
        An instance of the main Network class.
    ci :
        Show the uncertainty around the values estimates (standard deviation).
    show_surprise :
        If `True` plot each node's surprise together with sufficient statistics.
        If `False`, only the input node's surprise is depicted.
    show_posterior :
        If `True`, plot the posterior mean and precision on the top of expected mean and
        precision. Defaults to `False`.
    show_total_surprise :
        If `True`, plot the sum of surprises across all nodes in the bottom panel.
        Defaults to `False`.
    figsize :
        The width and height of the figure. Defaults to `(18, 9)` for a two-level model,
        or to `(18, 12)` for a three-level model.
    axs :
        A list of Matplotlib axes instances where to draw the trajectories. This should
        correspond to the number of nodes in the structure. The default is `None`
        (create a new figure).

    Returns
    -------
    axs :
        The Matplotlib axes instances where to plot the trajectories.

    Examples
    --------
    Visualization of nodes' trajectories from a three-level continuous HGF model.

    .. plot::

        from pyhgf import load_data
        from pyhgf.model import HGF

        # Set up standard 3-level HGF for continuous inputs
        hgf = HGF(
            n_levels=3,
            model_type="continuous",
            initial_mean={"1": 1.04, "2": 1.0, "3": 1.0},
            initial_precision={"1": 1e4, "2": 1e1, "3": 1e1},
            tonic_volatility={"1": -13.0, "2": -2.0, "3": -2.0},
            tonic_drift={"1": 0.0, "2": 0.0, "3": 0.0},
            volatility_coupling={"1": 1.0, "2": 1.0},
        )

        # Read USD-CHF data
        timeserie = load_data("continuous")

        # Feed input
        hgf.input_data(input_data=timeserie)

        # Plot
        hgf.plot_trajectories();

    Visualization of nodes' trajectories from a three-level binary HGF model.

    .. plot::

        from pyhgf import load_data
        from pyhgf.model import HGF
        import jax.numpy as jnp

        # Read binary input
        u, _ = load_data("binary")

        three_levels_hgf = HGF(
            n_levels=3,
            model_type="binary",
            initial_mean={"1": .0, "2": .5, "3": 0.},
            initial_precision={"1": .0, "2": 1e4, "3": 1e1},
            tonic_volatility={"1": None, "2": -6.0, "3": -2.0},
            tonic_drift={"1": None, "2": 0.0, "3": 0.0},
            volatility_coupling={"1": None, "2": 1.0},
            eta0=0.0,
            eta1=1.0,
            binary_precision = jnp.inf,
        )

        # Feed input
        three_levels_hgf = three_levels_hgf.input_data(u)

        # Plot
        three_levels_hgf.plot_trajectories();

    """
    trajectories_df = network.to_pandas()
    n_nodes = len(network.edges)

    if axs is None:
        _, axs = plt.subplots(
            nrows=n_nodes + 1 if show_total_surprise else n_nodes,
            figsize=figsize,
            sharex=True,
        )

    # plot all nodes
    # --------------
    ax_i = n_nodes - 1
    for node_idx in range(n_nodes):

        if node_idx in network.input_idxs:
            _show_posterior = True
            color = "#4c72b0"
        else:
            _show_posterior = show_posterior
            color = (
                "#55a868"
                if network.edges[node_idx].volatility_children is None
                else "#c44e52"
            )

        # use different colors for each node
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

    # plot the total surprise of the model
    # ------------------------------------
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


def plot_correlations(network: "Network") -> Axes:
    """Plot the heatmap correlation of the sufficient statistics trajectories.

    Parameters
    ----------
    network :
        An instance of the HGF model.

    Returns
    -------
    axs :
        The Matplotlib axe instance containing the heatmap of parameters trajectories
        correlation.

    """
    trajectories_df = network.to_pandas()
    trajectories_df = pd.concat(
        [
            trajectories_df[["time"]],
            trajectories_df[
                [
                    f"x_{i}_mean"
                    for i in range(len(network.edges))
                    if i in network.input_idxs
                ]
            ],
            trajectories_df.filter(regex="expected"),
            trajectories_df.filter(regex="surprise"),
        ],
        axis=1,
    )

    correlation_mat = trajectories_df.corr()
    ax = sns.heatmap(
        correlation_mat,
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        linewidths=2,
        square=True,
    )
    ax.set_title("Correlations between the model trajectories")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", size=8)
    ax.set_yticklabels(ax.get_yticklabels(), size=8)

    return ax


def plot_nodes(
    network: "Network",
    node_idxs: Union[int, List[int]],
    ci: bool = True,
    show_surprise: bool = True,
    show_posterior: bool = False,
    figsize: Tuple[int, int] = (12, 5),
    color: Optional[Union[Tuple, str]] = None,
    axs: Optional[Union[List, plt.Axes]] = None,
):
    """Plot the trajectory.

    Parameters
    ----------
    network : Network
        The network containing nodes whose trajectories will be plotted.
    node_idxs : int or list of int
        Index or list of indices of nodes to plot.
    ci : bool, optional (default=True)
        Whether to display confidence intervals around the expected mean.
    show_surprise : bool, optional (default=True)
        Whether to show surprise (currently unused in the function).
    show_posterior : bool, optional (default=False)
        Whether to display posterior mean estimates.
    figsize : tuple of int, optional (default=(12, 5))
        Size of the figure for plotting.
    color : tuple or str, optional (default=None)
        Color for the plotted lines.
    axs : list or plt.Axes, optional (default=None)
        Predefined matplotlib axes for plotting. If None, new axes are created.

    """
    if not isinstance(node_idxs, list):
        node_idxs = [node_idxs]
    trajectories_df = network.to_pandas()

    if axs is None:
        _, axs = plt.subplots(nrows=len(node_idxs), figsize=figsize, sharex=True)
    if isinstance(node_idxs, int) or len(node_idxs) == 1:
        axs = [axs]

    # Iterate over each node index to plot
    for i, node_idx in enumerate(node_idxs):
        ax = axs[i]

        # Dictionary to track unique legend labels and avoid duplicates
        handles_labels = {}

        # Loop through each trajectory and plot the corresponding data
        for traj, df_traj in trajectories_df.groupby("trajectory"):
            if node_idx in network.input_idxs:
                # Plot expected mean for input nodes
                (line,) = ax.plot(
                    df_traj.time,
                    df_traj[f"x_{node_idx}_expected_mean"],
                    label="Expected mean",
                    color=color,
                    linewidth=1,
                    zorder=2,
                    alpha=0.8,
                )
                ax.set_ylabel(rf"$\mu_{{{node_idx}}}$")
                handles_labels["Expected mean"] = line

                # If the node is continuous
                if network.edges[node_idx].node_type == 2:
                    scatter = ax.scatter(
                        x=df_traj.time,
                        y=df_traj[f"x_{node_idx}_mean"],
                        s=3,
                        label="Input",
                        color="#2a2a2a",
                        zorder=2,
                        alpha=0.8,
                    )
                    handles_labels["Input"] = scatter
                    if ci:
                        precision = df_traj[f"x_{node_idx}_expected_precision"]
                        sd = np.sqrt(1 / precision)
                        y1 = df_traj[f"x_{node_idx}_expected_mean"] - sd
                        y2 = df_traj[f"x_{node_idx}_expected_mean"] + sd
                        fill = ax.fill_between(
                            df_traj["time"], y1, y2, alpha=0.2, color=color, zorder=2
                        )
                        handles_labels["Confidence Interval"] = fill

                ax.set_title(f"Input Node {node_idx}", loc="left")

            else:
                # Plot state nodes (non-input)
                (line,) = ax.plot(
                    df_traj.time,
                    df_traj[f"x_{node_idx}_expected_mean"],
                    label="Expected mean",
                    color=color,
                    linewidth=1,
                    zorder=2,
                    alpha=0.8,
                )
                handles_labels["Expected mean"] = line
                ax.set_ylabel(rf"$\mu_{{{node_idx}}}$")

                if ci:
                    precision = df_traj[f"x_{node_idx}_expected_precision"]
                    sd = np.sqrt(1 / precision)
                    y1 = df_traj[f"x_{node_idx}_expected_mean"] - sd
                    y2 = df_traj[f"x_{node_idx}_expected_mean"] + sd
                    fill = ax.fill_between(
                        df_traj["time"], y1, y2, alpha=0.2, color=color, zorder=2
                    )
                    handles_labels["Confidence Interval"] = fill

                if show_posterior:
                    scatter = ax.scatter(
                        x=df_traj.time,
                        y=df_traj[f"x_{node_idx}_mean"],
                        s=3,
                        label="Posterior",
                        color="#2a2a2a",
                        zorder=2,
                        alpha=0.5,
                    )
                    handles_labels["Posterior"] = scatter

                ax.set_title(f"State Node {node_idx}", loc="left")

        ax.set_xlabel("Time")

        # Ensure the legend appears only once per subplot
        unique_handles = list(handles_labels.values())
        unique_labels = list(handles_labels.keys())
        ax.legend(unique_handles, unique_labels, loc="upper left")


def plot_network(network: "Network") -> "Source":
    """Visualization of node network using GraphViz.

    Parameters
    ----------
    network :
        An instance of main Network class.

    Notes
    -----
    This function requires [Graphviz](https://github.com/xflr6/graphviz) to be
    installed to work correctly.

    """
    try:
        import graphviz
    except ImportError:
        print(
            (
                "Graphviz is required to plot networks. "
                "See https://pypi.org/project/graphviz/"
            )
        )

    graphviz_structure = graphviz.Digraph("hgf-nodes", comment="Nodes structure")

    graphviz_structure.attr("node", shape="circle")

    # create the rest of nodes
    for idx in range(len(network.edges)):

        style = "filled" if idx in network.input_idxs else ""

        if network.edges[idx].node_type == 1:
            # binary state node
            graphviz_structure.node(
                f"x_{idx}", label=str(idx), shape="square", style=style
            )

        elif network.edges[idx].node_type == 2:
            # Continuous state nore
            graphviz_structure.node(
                f"x_{idx}", label=str(idx), shape="circle", style=style
            )

        elif network.edges[idx].node_type == 3:
            # Exponential family state nore
            graphviz_structure.node(
                f"x_{idx}",
                label=f"EF-{idx}",
                style="filled",
                shape="circle",
                fillcolor="#ced6e4",
            )

        elif network.edges[idx].node_type == 4:
            # Dirichlet Process state node
            graphviz_structure.node(
                f"x_{idx}",
                label=f"DP-{idx}",
                style="filled",
                shape="doublecircle",
                fillcolor="#e2d8c1",
            )

        elif network.edges[idx].node_type == 5:
            # Categorical state node
            graphviz_structure.node(
                f"x_{idx}",
                label=f"Ca-{idx}",
                style=style,
                shape="diamond",
                fillcolor="#e2d8c1",
            )

    # connect value parents
    for i, index in enumerate(network.edges):
        value_parents = index.value_parents

        if value_parents is not None:
            for value_parents_idx in value_parents:

                # get the coupling function from the value parent
                child_idx = network.edges[value_parents_idx].value_children.index(i)
                coupling_fn = network.edges[value_parents_idx].coupling_fn[child_idx]
                graphviz_structure.edge(
                    f"x_{value_parents_idx}",
                    f"x_{i}",
                    color="black" if coupling_fn is None else "black:invis:black",
                )

    # connect volatility parents
    for i, index in enumerate(network.edges):
        volatility_parents = index.volatility_parents

        if volatility_parents is not None:
            for volatility_parents_idx in volatility_parents:
                graphviz_structure.edge(
                    f"x_{volatility_parents_idx}",
                    f"x_{i}",
                    color="gray",
                    style="dashed",
                    arrowhead="dot",
                )

    # unflat the structure to better handle large/uneven networks
    graphviz_structure = graphviz_structure.unflatten(stagger=3)

    return graphviz_structure
