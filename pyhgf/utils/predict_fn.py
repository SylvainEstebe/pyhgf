# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>
# Author: Sylvain Estebe

from typing import TYPE_CHECKING

from jax import random, vmap
from jax.lax import scan
from jax.tree_util import Partial
from jax.typing import ArrayLike

from pyhgf.typing import Attributes

if TYPE_CHECKING:
    from pyhgf.model import Network


def predict_fn(
    network: "Network",
    time_steps: ArrayLike,
    n_predictions: int = 1,
    rng_key: ArrayLike = random.key(0),
    overwrite: bool = False,
    update_type: str = "eHGF",
) -> Attributes:
    """Generate n_predictions from the current state of the network in generative mode.

    Parameters
    ----------
    network :
        The network instance used for generating predictions.
    time_steps :
        Array of time steps.
    n_predictions :
        Number of predictions to generate. Defaults to 1.
    rng_key :
        Random number generator key, by default PRNGKey(0).
    overwrite :
        Whether to overwrite the existing generative scan function
    update_type :
        Type of update to use for belief propagation, by default "eHGF".

    Returns
    -------
    dict
        Dictionary of predictions
        corresponding to the number of predictions.

    """
    from pyhgf.utils import beliefs_propagation, get_update_sequence

    # Ensure that the network's input dimension is initialized.
    if not network.input_dim:
        network.get_input_dimension()

    # Create the update sequence if it does not exist.
    if network.update_sequence is None:
        network.update_sequence = get_update_sequence(
            network=network, update_type=update_type
        )

    # Create the generative scan function if it doesn't exist or if overwrite is True.
    if (network.scan_fn_generatif is None) or overwrite:
        network.scan_fn_generatif = Partial(
            beliefs_propagation,
            update_sequence=network.update_sequence,
            edges=network.edges,
            input_idxs=network.input_idxs,
            observations="generative",
            action_fn=network.action_fn,
        )

    # Save the initial state so that every prediction starts from the same point.
    initial_state = network.last_attributes

    print(f"Using initial state: {initial_state}")

    # Define a function to perform a single prediction using a given RNG key.
    def single_prediction(rng_key_single):
        # Split the RNG key for each time step.
        rng_keys = random.split(rng_key_single, num=len(time_steps))

        # Prepare placeholders for the inputs and observations.
        # In generative mode, these are not used so we set them to None.
        values_tuple = tuple([None] * len(network.input_idxs))
        observed_tuple = tuple([None] * len(network.input_idxs))
        inputs = (values_tuple, observed_tuple, time_steps, rng_keys)

        # Execute the belief propagation using scan, starting from the initial state.
        # This returns the final state (last_attributes) and the node trajectories.
        _, node_trajectories = scan(
            network.scan_fn_generatif,
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
