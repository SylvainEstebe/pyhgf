# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.interpreters.xla import DeviceArray
from jax.lax import dynamic_slice
from jax.scipy.stats import norm


def gaussian_surprise(hgf_results: Dict, response_function_parameters):
    """The sum of the gaussian surprise along the time series (continuous HGF).

    Parameters
    ----------
    hgf_results : dict
        Dictionary containing the HGF results.
    response_function_parameters : None
        No additional parameters are required.

    Return
    ------
    surprise : DeviceArray
        The model surprise given the input data.

    """
    _, results = hgf_results["final"]

    # Fill surprises with zeros if invalid input
    this_surprise = jnp.where(
        jnp.any(jnp.isnan(hgf_results["data"][1:, :]), axis=1), 0.0, results["surprise"]
    )

    # Return an infinite surprise if the model cannot fit
    this_surprise = jnp.where(jnp.isnan(this_surprise), jnp.inf, this_surprise)

    # Sum the surprise for this model
    surprise = jnp.sum(this_surprise)

    return surprise


def hrd_behaviors(
    hgf_results: Dict,
    response_function_parameters: Tuple[
        np.ndarray, np.ndarray, np.ndarray, float, np.ndarray
    ],
) -> DeviceArray:
    """The binary response function for the HRD task.

    The response function use the average of heart rate beliefs (here represented by
    the first level of the HGF model mu_1 over 3 heartbeats ans use this estimate
    to produce a response. The response probability is generated by a cumulative normal
    distribution where $loc = heart rate belief - sound frequency$.

    Parameters
    ----------
    hgf_results : dict
        Dictionary containing the HGF results.
    response_function_parameters : tuple
        Additional parameters used to compute the log probability from the HGF model:

        - triggers_idx : np.ndarray
            The triggers indexs (sample).
        - tones : np.ndarray
            The frequencies at each trial, converted to log(RR) (ms).
        - decisions : np.ndarray
            The participant decision (boolean) where `1` stands for Slower and `0` for
            Faster. The coding is different from what is used by the Psi staircase as
            the input use different units (log(RR), in miliseconds).
        - sigma_tone : float
            The precision of the tone perception. Defaults to `0.0`.

    Return
    ------
    surprise : DeviceArray
        The model evidence given the participant's behavior.

    """
    (
        triggers_idx,
        tones,
        decisions,
        sigma_tone,
        time_vector,
    ) = response_function_parameters

    # The estimated heart rate (mu_1)
    mu_1 = hgf_results["final"][0][1][0]["mu"]

    # The precision around the heart rate restimate
    pi_1 = hgf_results["final"][0][1][0]["pi"]

    # The surprise along the entire time series
    model_surprise = hgf_results["final"][1]["surprise"]

    # The time vector
    time = hgf_results["data"][:, 1] * 1000

    # Interpolate the model trajectories to 1000 Hz
    new_mu1 = jnp.interp(
        x=time_vector,
        xp=time[1:],
        fp=mu_1,
    )
    new_pi1 = jnp.interp(
        x=time_vector,
        xp=time[1:],
        fp=pi_1,
    )

    # Extract the values of mu_1 and pi_1 for each trial
    # The heart rate belief and the precision of the heart rate belief
    # --------------------------------------------------

    # First define a function to extract values for one trigger
    # (Use the average over the 5 seconds after the trigger)
    def extract(trigger: int, new_mu1=new_mu1, new_pi1=new_pi1):
        return (
            dynamic_slice(new_mu1, (trigger,), (5000,)).mean(),
            dynamic_slice(new_pi1, (trigger,), (5000,)).mean(),
        )

    hr, precision = vmap(extract)(triggers_idx)

    # The probability of answering Slower
    cdf = norm.cdf(
        0,
        loc=hr - tones,
        scale=jnp.sqrt(1 / precision + (sigma_tone**2)),
    )

    # Force probabilities to be greater or smaller than 0 or 1
    cdf = jnp.where(cdf == 1.0, 1 - 1e-6, cdf)
    cdf = jnp.where(cdf == 0.0, 1e-6, cdf)

    # The surprise (sum of the log(p)) of the model given the answers
    surprise = jnp.where(decisions, -jnp.log(cdf), -jnp.log(1 - cdf)).sum()

    # Return an infinite surprise if the model could not fit in the first place
    surprise = jnp.where(jnp.any(jnp.isnan(model_surprise)), jnp.inf, surprise)

    return surprise


def hrd_ideal_oberver(
    hgf_results: Dict,
    response_function_parameters: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> DeviceArray:
    """The binary response function for the HRD task given by an ideal observer.

    Parameters
    ----------
    hgf_results : dict
        Dictionary containing the HGF results.
    response_function_parameters : tuple
        Additional parameters used to compute the log probability from the HGF model:

       - triggers_idx : np.ndarray
            The triggers indexs (sample).
        - tones : np.ndarray
            The frequencies of the tones presented at each trial.
        - decisions : np.ndarray
            The participant decision (boolean) where `1` stands for Slower and `0` for
            Faster. The coding is different from what is used by the Psi staircase as
            the input use different units (log(RR), in miliseconds).

    Return
    ------
    surprise : DeviceArray
        The model evidence given the participant's behavior.

    """
    (
        triggers_idx,
        tones,
        condition,
    ) = response_function_parameters

    # The estimated heart rate (mu_1)
    mu_1 = hgf_results["final"][0][1][0]["mu"]

    # The surprise along the entire time series
    model_surprise = hgf_results["final"][1]["surprise"]

    # The time vector
    time = hgf_results["data"][:, 1]

    # Extract the values of mu_1 and pi_1 for each trial
    # The heart rate belief and the precision of the heart rate belief
    # --------------------------------------------------

    # First define a function to extract values for one trigger
    @jit
    def extract(trigger: int, mu_1=mu_1, time=time):
        idx = jnp.sum(time <= trigger) + 1
        return dynamic_slice(mu_1, (idx,), (3,)).mean()

    hr = vmap(extract)(triggers_idx)

    # The probability of answering Slower
    cdf = norm.cdf(
        0,
        loc=hr - tones,
        scale=1e-6,
    )

    # Force probabilities to be greater or smaller than 0 or 1
    cdf = jnp.where(cdf == 1.0, 1 - 1e-6, cdf)
    cdf = jnp.where(cdf == 0.0, 1e-6, cdf)

    # The surprise (negative sum of the log(p)) of the model given the answers
    surprise = jnp.where(condition, -jnp.log(cdf), -jnp.log(1 - cdf)).sum()

    # Return an infinite surprise if the model could not fit in the first place
    surprise = jnp.where(jnp.any(jnp.isnan(model_surprise)), jnp.inf, surprise)

    return surprise
