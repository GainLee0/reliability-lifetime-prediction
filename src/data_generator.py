import numpy as np
import pandas as pd


def stretched_exponential(t, a, b):
    """Stretched-exponential degradation model."""
    return np.exp(-a * np.power(t, b))


def generate_synthetic_data(
    n_samples=100,
    max_time=1000,
    n_timepoints=51,
    random_seed=42
):
    """
    Generate fully synthetic degradation time-series data.

    Each sample has slightly different degradation parameters
    and random measurement noise.
    """
    rng = np.random.default_rng(random_seed)

    time = np.linspace(0, max_time, n_timepoints)

    data = {"Time": time}

    for i in range(1, n_samples + 1):
        # Synthetic sample-to-sample variation
        a = rng.normal(loc=0.00025, scale=0.000025)
        b = rng.normal(loc=0.65, scale=0.04)

        # Keep parameters physically reasonable
        a = max(a, 0.00005)
        b = max(b, 0.20)

        true_curve = stretched_exponential(time, a, b)

        # Add small synthetic measurement noise
        noise = rng.normal(loc=0.0, scale=0.0015, size=len(time))
        observed_curve = true_curve + noise

        # Force initial point to exactly 1.0
        observed_curve[0] = 1.0

        data[f"Sample_{i:03d}"] = observed_curve

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/synthetic_degradation.csv", index=False)

    print(df.head())
    print(f"\nGenerated dataset shape: {df.shape}")
