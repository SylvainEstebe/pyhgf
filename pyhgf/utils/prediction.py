from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import jit, lax, random
from jax.typing import ArrayLike

from pyhgf.typing import Attributes, Edges, UpdateSequence
from pyhgf.updates.observation import set_observation


def sample_node_distribution(
    attributes: Dict[int, dict],
    node_idx: int,
    rng_key: random.PRNGKey,
    model_type: int,
) -> Tuple[float, random.PRNGKey]:
    """Sample a value from the distribution of the specified node.

    Parameters
    ----------
    attributes : Dict[int, dict]
        The dictionary of node parameters, keyed by node index.
    node_idx : int
        The index of the child nodes whose distribution is to be sampled.
    rng_key : random.PRNGKey
        A PRNG key for random number generation.
    model_type : int
        Specifies the distribution type (e.g., discrete=1, continuous=2).

    Returns
    -------
    sample : float
        The sampled value from the node's distribution.
    rng_key : random.PRNGKey
        Updated PRNG key.

    """
    # Fetch parameters from attributes
    node_attr = attributes[node_idx]
    mu = node_attr.get("expected_mean", None)
    precision = node_attr.get("expected_precision", None)

    sigma = 1.0 / jnp.sqrt(precision)

    # Split RNG key
    rng_key, subkey = random.split(rng_key)

    # Sample conditionally based on model type
    sample = lax.cond(
        model_type == 2,
        lambda _: random.normal(subkey) * sigma + mu,  # Continuous distribution
        lambda _: jnp.float32(random.bernoulli(subkey)),  # Discrete (Bernoulli)
        operand=None,
    )
    return sample, rng_key


def scan_sampling(
    attributes: Attributes,
    node_idx: int,
    rng_key: random.PRNGKey,
    model_type: int,
    num_samples: int,
) -> ArrayLike:
    """Generate multiple samples from a node's distribution using JAX's `scan` function.

    Parameters
    ----------
    attributes : Attributes
        The dictionary of nodes' parameters.
    node_idx : int
        The index of the node to sample from.
    rng_key : random.PRNGKey
        A PRNG key used for random number generation.
    model_type : int
        Specifies how to generate samples (e.g., 1 for discrete, 2 for continuous).
    num_samples : int
        The number of samples to generate.

    Returns
    -------
    samples : ArrayLike
        An array of shape (num_samples,) containing the generated samples for the
        specified node's distribution.

    """

    def scan_fn(carry, _):
        rng_key_in = carry
        sample, rng_key_out = sample_node_distribution(
            attributes, node_idx, rng_key_in, model_type
        )
        return rng_key_out, sample

    _, samples = lax.scan(scan_fn, rng_key, None, length=num_samples)
    return samples


def handle_observation(
    attributes: Attributes,
    node_idx: int,
    rng_key: random.PRNGKey,
    sophisticated: bool,
    edges: Edges,
) -> Tuple[Attributes, random.PRNGKey]:
    """Handle the observation for a specific node based on the sophistication flag.

    Parameters
    ----------
    attributes : Attributes
        The dictionary of node parameters.
    node_idx : int
        The index of the node being observed.
    rng_key : random.PRNGKey
        Random number generator key.
    sophisticated : bool
        Whether to use sophisticated sampling.
    edges : Edges
        Network edges containing node type and connections.

    Returns
    -------
    updated_attributes : Attributes
        Updated attributes dictionary.
    rng_key : random.PRNGKey
        Updated RNG key.

    """

    def sophisticated_mode(_):
        sampled_value, rng_key_new = sample_node_distribution(
            attributes, node_idx, rng_key, edges[node_idx].node_type
        )
        return sampled_value, 1, rng_key_new

    def non_sophisticated_mode(_):
        return jnp.nan, 0, rng_key

    if sophisticated:
        sampled_value, observed, rng_key_out = sophisticated_mode(None)
    else:
        sampled_value, observed, rng_key_out = non_sophisticated_mode(None)

    updated_attributes = set_observation(
        attributes=attributes,
        node_idx=node_idx,
        values=sampled_value,
        observed=observed,
    )

    return updated_attributes, rng_key_out


@partial(
    jit, static_argnames=("update_sequence", "edges", "input_idxs", "sophisticated")
)
def inference_prediction(
    attributes: Attributes,
    inputs: Tuple[ArrayLike, ...],
    update_sequence: UpdateSequence,
    edges: Edges,
    input_idxs: Tuple[int],
    sophisticated: bool,
    rng_key: random.PRNGKey,
) -> Tuple[Dict, Dict, random.PRNGKey]:
    """Perform prediction steps with his own distribution of child node.

    Parameters
    ----------
    attributes : Attributes
        The dictionary of node parameters.
    inputs : Tuple[ArrayLike, ...]
        Input data (e.g., time step or observations).
    update_sequence : UpdateSequence
        Two lists: (prediction_steps, update_steps).
    edges : Edges
        Network edges containing node types and connections.
    input_idxs : Tuple[int]
        Indices of nodes receiving observations at this time step.
    sophisticated : bool
        Whether to sample observations from his own node or not.
    rng_key : random.PRNGKey
        Random number generator key to be used for sampling.

    Returns
    -------
    updated_attributes : Dict
        The attributes after all prediction, observation, and update steps.
    duplicated_attributes : Dict
        A duplicate of updated attributes (as required by your interface).
    rng_key : random.PRNGKey
        Updated RNG key after sampling.

    """
    # Use the passed rng_key without reinitializing it.
    prediction_steps, update_steps = update_sequence

    # Assign time_step (e.g. input data) to a special key in attributes.
    attributes[-1]["time_step"] = inputs

    # Run prediction steps.
    for node_idx, update_fn in prediction_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)

    # Handle observations on specified input nodes.
    for node_idx in input_idxs:
        attributes, rng_key = handle_observation(
            attributes, node_idx, rng_key, sophisticated, edges
        )

    # Run update steps.
    for node_idx, update_fn in update_steps:
        attributes = update_fn(attributes=attributes, node_idx=node_idx, edges=edges)

    return attributes, attributes, rng_key
