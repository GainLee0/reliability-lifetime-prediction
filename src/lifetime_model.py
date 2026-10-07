import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def stretched_exponential(t, a, b):
    """
    Stretched-exponential degradation model.

    Parameters
    ----------
    t : array-like
        Time values.
    a : float
        Degradation-rate parameter.
    b : float
        Shape parameter.

    Returns
    -------
    np.ndarray
        Predicted normalized response.
    """
    t = np.asarray(t, dtype=float)
    return np.exp(-a * np.power(t, b))


def calculate_t95(a, b):
    """
    Calculate T95, defined as the time at which
    the normalized response reaches 95% of its initial value.
    """
    if a <= 0 or b <= 0:
        return np.nan

    return float(
        np.power(
            np.log(1 / 0.95) / a,
            1.0 / b
        )
    )


def fit_lifetime_model(time, values):
    """
    Fit a stretched-exponential degradation model to one sample.

    The function:
    1. Removes invalid values
    2. Sorts observations by time
    3. Normalizes the response to the initial measurement
    4. Fits model parameters a and b
    5. Calculates R² and T95

    Parameters
    ----------
    time : array-like
        Measurement times.
    values : array-like
        Observed degradation measurements.

    Returns
    -------
    dict
        Model parameters, predictions, normalized observations,
        R² score, and estimated T95.
    """

    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    # Remove invalid observations
    valid_mask = np.isfinite(time) & np.isfinite(values)
    time = time[valid_mask]
    values = values[valid_mask]

    # Sort chronologically
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    values = values[sort_idx]

    # Check for baseline measurement
    baseline_idx = np.where(time == 0)[0]

    if len(baseline_idx) == 0:
        raise ValueError(
            "A baseline measurement at time = 0 is required."
        )

    baseline = values[baseline_idx[0]]

    if baseline == 0:
        raise ValueError(
            "Baseline measurement cannot be zero."
        )

    # Normalize relative to initial value
    normalized_values = values / baseline

    # Nonlinear model fitting
    try:
        params, covariance = curve_fit(
            stretched_exponential,
            time,
            normalized_values,
            p0=[0.0002, 0.7],
            bounds=([1e-8, 0.1], [1.0, 2.0]),
            maxfev=10000
        )

        a, b = params

        predicted_values = stretched_exponential(
            time,
            a,
            b
        )

        r2 = r2_score(
            normalized_values,
            predicted_values
        )

        t95 = calculate_t95(a, b)

        return {
            "a": float(a),
            "b": float(b),
            "r2": float(r2),
            "t95": float(t95),
            "normalized_values": normalized_values,
            "predicted_values": predicted_values,
            "covariance": covariance
        }

    except (RuntimeError, ValueError):
        return {
            "a": np.nan,
            "b": np.nan,
            "r2": np.nan,
            "t95": np.nan,
            "normalized_values": normalized_values,
            "predicted_values": np.full_like(
                normalized_values,
                np.nan
            ),
            "covariance": None
        }
