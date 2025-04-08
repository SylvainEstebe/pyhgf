from typing import Optional

import numpy as np
from jax import random, vmap
from jax.lax import scan
from jax.random import PRNGKey
from jax.tree_util import Partial

from pyhgf.utils import beliefs_propagation, get_update_sequence


def predict(
    Network,
    n_predictions: int,
    time_steps: Optional[np.ndarray] = None,
    rng_key: random.PRNGKey = PRNGKey(0),
    overwrite: bool = False,
    update_type: str = "eHGF",
):
    """Generate n_predictions from the current state of the network in generative mode.

    Parameters
    ----------
    Network : object
        The network instance used for generating predictions.
    n_predictions : int
        Number of predictions to generate.
    time_steps : Optional[np.ndarray], optional
        Array of time steps. If None, defaults to an array of ones of length 100.
    rng_key : random.PRNGKey, optional
        Random number generator key, by default PRNGKey(0).
    overwrite : bool, optional
        Whether to overwrite the existing generative scan function
    update_type : str, optional
        Type of update to use for belief propagation, by default "eHGF".

    Returns
    -------
    dict
        Dictionary of predictions
        corresponding to the number of predictions.

    """
    # Set the time step vector. If none is provided, default to a vector of ones.
    if time_steps is None:
        time_steps = np.ones(100)

    # Ensure that the network's input dimension is initialized.
    if not Network.input_dim:
        Network.get_input_dimension()

    # Create the update sequence if it does not exist.
    if Network.update_sequence is None:
        Network.update_sequence = get_update_sequence(
            network=Network, update_type=update_type
        )

    # Create the generative scan function if it doesn't exist or if overwrite is True.
    if (
        (not hasattr(Network, "scan_fn_generatif"))
        or (Network.scan_fn_generatif is None)
        or overwrite
    ):
        Network.scan_fn_generatif = Partial(
            beliefs_propagation,
            update_sequence=Network.update_sequence,
            edges=Network.edges,
            input_idxs=Network.input_idxs,
            observations="generative",
            action_fn=Network.action_fn,
        )

    # Save the initial state so that every prediction starts from the same point.
    initial_state = Network.last_attributes
    print(f"Using initial state: {initial_state}")

    # Define a function to perform a single prediction using a given RNG key.
    def single_prediction(rng_key_single):
        # Split the RNG key for each time step.
        rng_keys = random.split(rng_key_single, num=len(time_steps))

        # Prepare placeholders for the inputs and observations.
        # In generative mode, these are not used so we set them to None.
        values_tuple = tuple([None] * len(Network.input_idxs))
        observed_tuple = tuple([None] * len(Network.input_idxs))
        inputs = (values_tuple, observed_tuple, time_steps, rng_keys)

        # Execute the belief propagation using scan, starting from the initial state.
        # This returns the final state (last_attributes) and the node trajectories.
        last_attributes, node_trajectories = scan(
            Network.scan_fn_generatif,
            initial_state,
            inputs,
        )
        # Return only the node trajectories.
        return node_trajectories

    # Generate a batch of RNG keys, one for each prediction.
    rng_keys_batch = random.split(rng_key, num=n_predictions)

    # Use vmap to vectorize the single_prediction function over the batch of RNG keys.
    # This will return a dictionary of arrays.
    predictions = vmap(single_prediction)(rng_keys_batch)

    return predictions
