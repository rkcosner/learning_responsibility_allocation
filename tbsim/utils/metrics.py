"""
Adapted from https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/evaluation/metrics.py
"""


from typing import Callable

import numpy as np


metric_signature = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _assert_shapes(ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray) -> None:
    """
    Check the shapes of args required by metrics
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(timesteps)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(timesteps) with the availability for each gt timesteps
    Returns:
    """
    assert len(pred.shape) == 4, f"expected 3D (BxMxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert ground_truth.shape == (
        batch_size,
        future_len,
        num_coords,
    ), f"expected 2D (Batch x Time x Coords) array for gt, got {ground_truth.shape}"
    assert confidences.shape == (batch_size, num_modes,), f"expected 2D (Batch x Modes) array for confidences, got {confidences.shape}"
    assert np.allclose(np.sum(confidences, axis=1), 1), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len,), f"expected 1D (Time) array for avails, got {avails.shape}"
    # assert all data are valid
    assert np.isfinite(pred).all(), "invalid value found in pred"
    assert np.isfinite(ground_truth).all(), "invalid value found in gt"
    assert np.isfinite(confidences).all(), "invalid value found in confidences"
    assert np.isfinite(avails).all(), "invalid value found in avails"


def batch_neg_multi_log_likelihood(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray
) -> np.ndarray:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    For more details about used loss function and reformulation, please see
    https://github.com/lyft/l5kit/blob/master/competition.md.
    Args:
        ground_truth (np.ndarray): array of shape (batchsize)x(timesteps)x(2D coords)
        pred (np.ndarray): array of shape (batchsize)x(modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape ((batchsize)xmodes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batchsize)x(timesteps) with the availability for each gt timesteps
    Returns:
        np.ndarray: negative log-likelihood for this batch, an array of float numbers
    """
    _assert_shapes(ground_truth, pred, confidences, avails)

    ground_truth = np.expand_dims(ground_truth, 1)  # add modes
    avails = avails[:, np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((ground_truth - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        error = np.log(confidences) - 0.5 * np.sum(error, axis=-1)  # reduce timesteps

    # use max aggregator on modes for numerical stability
    max_value = np.max(error, axis=-1, keepdims=True)  # error are negative at this point, so max() gives the minimum one
    error = -np.log(np.sum(np.exp(error - max_value), axis=-1)) - max_value  # reduce modes
    return error


def batch_rmse(ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray) -> np.ndarray:
    """
    Return the root mean squared error, computed using the stable nll
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(timesteps)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(timesteps) with the availability for each gt timesteps
    Returns:
        np.ndarray: negative log-likelihood for this batch, an array of float numbers
    """
    nll = batch_neg_multi_log_likelihood(ground_truth, pred, confidences, avails)
    _, _, future_len, _ = pred.shape

    return np.sqrt(2 * nll / future_len)


def batch_prob_true_mode(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray
) -> np.ndarray:
    """
    Return the probability of the true mode
    Args:
        ground_truth (np.ndarray): array of shape (timesteps)x(2D coords)
        pred (np.ndarray): array of shape (modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (timesteps) with the availability for each gt timesteps
    Returns:
        np.ndarray: a (modes) numpy array
    """
    _assert_shapes(ground_truth, pred, confidences, avails)

    ground_truth = np.expand_dims(ground_truth, 1)  # add modes
    avails = avails[:, np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((ground_truth - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        error = np.log(confidences) - 0.5 * np.sum(error, axis=-1)  # reduce timesteps

    # use max aggregator on modes for numerical stability
    max_value = np.max(error, axis=-1, keepdims=True)  # error are negative at this point, so max() gives the minimum one

    error = np.exp(error - max_value) / np.sum(np.exp(error - max_value))
    return error


def batch_time_displace(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray
) -> np.ndarray:
    """
    Return the displacement at timesteps T
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(timesteps)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(timesteps)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(timesteps) with the availability for each gt timesteps
    Returns:
        np.ndarray: a (batch)x(timesteps) numpy array
    """
    true_mode_error = batch_prob_true_mode(ground_truth, pred, confidences, avails)
    true_mode_error = true_mode_error[:, :, np.newaxis]  # add timesteps axis

    ground_truth = np.expand_dims(ground_truth, 1)  # add modes
    avails = avails[:, np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((ground_truth - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability
    return np.sum(true_mode_error * np.sqrt(error), axis=1)  # reduce modes


def batch_average_displacement_error(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray, mode: str="mean"
) -> np.ndarray:
    """
    Returns the average displacement error (ADE), which is the average displacement over all timesteps.
    During calculation, confidences are ignored, and two modes are available:
        - oracle: only consider the best hypothesis
        - mean: average over all hypotheses
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(time) with the availability for each gt timestep
        mode (str): calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)
    Returns:
        np.ndarray: average displacement error (ADE) of the batch, an array of float numbers
    """
    _assert_shapes(ground_truth, pred, confidences, avails)

    ground_truth = np.expand_dims(ground_truth, 1)  # add modes
    avails = avails[:, np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((ground_truth - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability
    error = error ** 0.5  # calculate root of error (= L2 norm)
    error = np.mean(error, axis=-1)  # average over timesteps

    if mode == "oracle":
        error = np.min(error, axis=1)  # use best hypothesis
    elif mode == "mean":
        error = np.mean(error, axis=1)  # average over hypotheses
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error


def batch_final_displacement_error(
        ground_truth: np.ndarray, pred: np.ndarray, confidences: np.ndarray, avails: np.ndarray, mode: str="mean"
) -> np.ndarray:
    """
    Returns the final displacement error (FDE), which is the displacement calculated at the last timestep.
    During calculation, confidences are ignored, and two modes are available:
        - oracle: only consider the best hypothesis
        - mean: average over all hypotheses
    Args:
        ground_truth (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(modes)x(time)x(2D coords)
        confidences (np.ndarray): array of shape (batch)x(modes) with a confidence for each mode in each sample
        avails (np.ndarray): array of shape (batch)x(time) with the availability for each gt timestep
        mode (str): calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)
    Returns:
        np.ndarray: final displacement error (FDE) of the batch, an array of float numbers
    """
    _assert_shapes(ground_truth, pred, confidences, avails)

    ground_truth = np.expand_dims(ground_truth, 1)  # add modes
    avails = avails[:, np.newaxis, :, np.newaxis]  # add modes and cords

    error = np.sum(((ground_truth - pred) * avails) ** 2, axis=-1)  # reduce coords and use availability
    error = error ** 0.5  # calculate root of error (= L2 norm)
    error = error[:, :, -1]  # use last timestep

    if mode == "oracle":
        error = np.min(error, axis=-1)  # use best hypothesis
    elif mode == "mean":
        error = np.mean(error, axis=-1)  # average over hypotheses
    else:
        raise ValueError(f"mode: {mode} not valid")

    return error


def single_mode_metrics(metrics_func, ground_truth: np.ndarray, pred: np.ndarray, avails: np.ndarray):
    """
    Run a metrics with single mode by inserting a mode dimension

    Args:
        ground_truth (np.ndarray): array of shape (batch)x(time)x(2D coords)
        pred (np.ndarray): array of shape (batch)x(time)x(2D coords)
        avails (np.ndarray): array of shape (batch)x(time) with the availability for each gt timestep
        mode (str): Optional, set to None when not applicable
            calculation mode - options are 'mean' (average over hypotheses) and 'oracle' (use best hypotheses)
    Returns:
        np.ndarray: metrics values
    """
    pred = pred[:, None]
    conf = np.ones((pred.shape[0], 1))
    kwargs = dict(ground_truth=ground_truth, pred=pred, confidences=conf, avails=avails)
    return metrics_func(**kwargs)