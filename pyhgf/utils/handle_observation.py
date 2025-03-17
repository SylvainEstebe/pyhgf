from typing import Any, Dict, Tuple, cast

from jax import random

from pyhgf.typing import Attributes, Edges
from pyhgf.updates.observation import set_observation
from pyhgf.utils.sample_node_distribution import sample_node_distribution


def handle_observation(
    attributes: Attributes,
    node_idx: int,
    rng_key: random.PRNGKey,
    edges: Edges,
) -> Tuple[Attributes, random.PRNGKey]:
    """Handle the observation for a specific node.

    Parameters
    ----------
    attributes : Attributes
        The dictionary of node parameters.
    node_idx : int
        The index of the node being observed.
    rng_key : random.PRNGKey
        Random number generator key.
    edges : Edges
        Network edges containing node type and connections.

    Returns
    -------
    updated_attributes : Attributes
        Updated attributes dictionary.
    rng_key : random.PRNGKey
        Updated RNG key.

    """
    # Sample the node distribution
    sampled_value, rng_key_new = sample_node_distribution(
        cast(
            Dict[int, dict[Any, Any]], attributes
        ),  # explicit cast for type compatibility
        node_idx,
        rng_key,
        edges[node_idx].node_type,
    )

    # Set the observation (with a constant observation flag, here set as 1)
    updated_attributes = set_observation(
        attributes=attributes,
        node_idx=node_idx,
        values=sampled_value,
        observed=1,
    )

    return updated_attributes, rng_key_new
