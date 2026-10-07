import pandas as pd
import numpy as np

from lifetime_model import fit_lifetime_model


def analyze_dataset(csv_path):
    """
    Fit the lifetime model to every sample in the dataset.

    Parameters
    ----------
    csv_path : str
        Path to synthetic degradation CSV file.

    Returns
    -------
    pd.DataFrame
        Sample-level fitted parameters and model metrics.
    """

    df = pd.read_csv(csv_path)

    time = df["Time"].to_numpy()
    sample_columns = [col for col in df.columns if col != "Time"]

    results = []

    for sample in sample_columns:
        fit_result = fit_lifetime_model(
            time,
            df[sample].to_numpy()
        )

        results.append({
            "sample": sample,
            "a": fit_result["a"],
            "b": fit_result["b"],
            "r2": fit_result["r2"],
            "t95": fit_result["t95"]
        })

    return pd.DataFrame(results)


def summarize_results(results_df):
    """
    Calculate summary statistics for sample-level T95 estimates.
    """

    valid_t95 = results_df["t95"].dropna()

    mean_t95 = valid_t95.mean()
    std_t95 = valid_t95.std()
    cv_t95 = std_t95 / mean_t95 * 100

    lower_bound = mean_t95 - 3 * std_t95
    upper_bound = mean_t95 + 3 * std_t95

    outliers = results_df[
        (results_df["t95"] < lower_bound) |
        (results_df["t95"] > upper_bound)
    ]

    summary = {
        "mean_t95": mean_t95,
        "std_t95": std_t95,
        "cv_t95_percent": cv_t95,
        "mean_r2": results_df["r2"].mean(),
        "n_samples": len(results_df),
        "n_valid_t95": len(valid_t95),
        "n_outliers": len(outliers)
    }

    return summary, outliers


if __name__ == "__main__":

    results_df = analyze_dataset(
        "data/synthetic_degradation.csv"
    )

    summary, outliers = summarize_results(results_df)

    print("\nSample-level fitting results:")
    print(results_df.head())

    print("\nSummary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\nPotential outliers:")
    if len(outliers) == 0:
        print("No outliers detected.")
    else:
        print(outliers[["sample", "t95", "r2"]])

    results_df.to_csv(
        "data/lifetime_results.csv",
        index=False
    )