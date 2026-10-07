import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifetime_model import fit_lifetime_model, stretched_exponential


def build_aggregate_dataset(df):
    """
    Pool normalized observations from all samples.
    """

    time = df["Time"].to_numpy()
    sample_columns = [col for col in df.columns if col != "Time"]

    x_all = []
    y_all = []

    for sample in sample_columns:
        values = df[sample].to_numpy()

        if values[0] == 0:
            continue

        normalized = values / values[0]

        x_all.extend(time)
        y_all.extend(normalized)

    return np.asarray(x_all), np.asarray(y_all)


def plot_results(
    data_path="data/synthetic_degradation.csv",
    results_path="data/lifetime_results.csv",
    output_path="figures/lifetime_summary.png"
):
    """
    Create aggregate degradation and T95 distribution plots.
    """

    df = pd.read_csv(data_path)
    results_df = pd.read_csv(results_path)

    # Build pooled dataset
    x_all, y_all = build_aggregate_dataset(df)

    # Fit aggregate model
    aggregate_result = fit_lifetime_model(
        x_all,
        y_all
    )

    aggregate_t95 = aggregate_result["t95"]

    # Smooth curve for visualization
    x_plot = np.linspace(
        x_all.min(),
        x_all.max(),
        400
    )

    y_plot = stretched_exponential(
        x_plot,
        aggregate_result["a"],
        aggregate_result["b"]
    )

    # Create figure
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    # Plot 1: degradation data + aggregate model
    axes[0].scatter(
        x_all,
        y_all * 100,
        s=8,
        alpha=0.15,
        label="Sample observations"
    )

    axes[0].plot(
        x_plot,
        y_plot * 100,
        linewidth=2,
        label="Aggregate model"
    )

    axes[0].set_xlabel("Time [hours]")
    axes[0].set_ylabel("Normalized response [%]")
    axes[0].set_title("Degradation Data and Aggregate Model")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Plot 2: T95 distribution
    axes[1].hist(
        results_df["t95"].dropna(),
        bins=12,
        alpha=0.75
    )

    axes[1].axvline(
        aggregate_t95,
        linewidth=2,
        linestyle="--",
        label=f"Aggregate T95 = {aggregate_t95:.0f} h"
    )

    axes[1].set_xlabel("T95 Lifetime [hours]")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Sample-Level T95 Distribution")
    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Aggregate model: "
        f"exp(-{aggregate_result['a']:.6f} * t^{aggregate_result['b']:.4f})"
    )

    print(
        f"Aggregate R²: {aggregate_result['r2']:.4f}"
    )

    print(
        f"Aggregate T95: {aggregate_t95:.1f} hours"
    )


if __name__ == "__main__":
    plot_results()