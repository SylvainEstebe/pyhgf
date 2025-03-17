# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>


from functools import partial
from typing import Any, Dict, Tuple, cast

from jax import jit, random
from jax.typing import ArrayLike

from pyhgf.typing import Attributes, Edges, UpdateSequence
from pyhgf.updates.observation import set_observation
from pyhgf.utils.sample_node_distribution import sample_node_distribution


@partial(
    jit, static_argnames=("update_sequence", "edges", "input_idxs", "sophisticated")
)
def beliefs_propagation(
    attributes: Attributes,
    inputs: Tuple[ArrayLike, ...],
    update_sequence: UpdateSequence,
    edges: Edges,
    input_idxs: Tuple[int],
    sophisticated: bool,
    rng_key: random.PRNGKey = None,
) -> Tuple[Dict, Dict]:
    """Update the networks parameters after observing new data point(s).

    This function performs the beliefs propagation step. Belief propagation consists in:
    1. A prediction sequence, from the leaves of the graph to the roots.
    2. The assignation of new observations to target nodes (usually the roots of the
    network)
    3. An inference step alternating between prediction errors and posterior updates,
    starting from the roots of the network to the leaves.
    This function returns a tuple of two new `parameter_structure` (i.e. the carryover
    and the accumulated in the context of :py:func:`jax.lax.scan`).

    Parameters
    ----------
    attributes :
        The dictionaries of nodes' parameters. This variable is updated and returned
        after the beliefs propagation step.
    inputs :
        A tuple of n by time steps arrays containing the new observation(s), the time
        steps as well as a boolean mask for observed values. The new observations are a
        tuple of array, with length equal to the number of input nodes. Each input node
        can receive observations  The time steps are the last
        column of the array, the default is unit incrementation.
    update_sequence :
        The sequence of updates that will be applied to the node structure.
    edges :
        Information on the network's edges.
    sophisticated :
        A boolean indicating whether the network use sophisticated inference or not.
    rng_key :
        Random key for the random number generator.
    input_idxs :
        List input indexes.

    Returns
    -------
    attributes, attributes :
        A tuple of parameters structure (carryover and accumulated).

    """
    prediction_steps, update_steps = update_sequence

    if sophisticated:
        time_step = (
            inputs  # In sophisticated mode, inputs serve directly as the time step.
        )
    else:
        values_tuple, observed_tuple, time_step = inputs

    # Assign the time_step (or input data) to the attributes.
    attributes[-1]["time_step"] = time_step

    # 1. Prediction sequence (common for both modes)
    for node_idx, update_fn in prediction_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)

    # 2. Handle observations
    if sophisticated:
        # Inline handling of observation for each input node
        for node_idx in input_idxs:
            # Sample the node distribution
            sampled_value, rng_key = sample_node_distribution(
                cast(
                    Dict[int, dict[Any, Any]], attributes
                ),  # cast for type compatibility
                node_idx,
                rng_key,
                edges[node_idx].node_type,
            )
            # Set the observation (using a constant observation flag, here set as 1)
            attributes = set_observation(
                attributes=attributes,
                node_idx=node_idx,
                values=sampled_value,
                observed=1,
            )
    else:
        # Unpack observation data and update each input node.
        for values, observed, node_idx in zip(values_tuple, observed_tuple, input_idxs):
            attributes = set_observation(
                attributes=attributes,
                node_idx=node_idx,
                values=values.squeeze(),
                observed=observed,
            )

    # 3. Update sequence (common for both modes)
    for node_idx, update_fn in update_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)

    return (
        attributes,
        attributes,
    )  # ("carryover", "accumulated")
