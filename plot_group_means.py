"""Replicate Figure 6: group means with 95% confidence intervals.

Group means are calculated from czi_annotation_measurements.csv. The confidence
intervals are mean +/- t * SD / sqrt(n) with n - 1 degrees of freedom, treating
every value as independent, as in the original figure (they do not account for
clustering within slices and patients - see
mixed_effects_results/mixed_effects_models.csv for the multilevel model
results).

For cell area, each value is an individual cell (two per measurement), as in
the original Figure 6B.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from swarm_plots import (
    CLASS_ORDER,
    METRIC_LABELS,
    METRIC_UNITS,
    build_group_palette,
    load_area_data,
    load_data,
)


PANELS = [
    ("A", "full_line_length", "Nucleus to nucleus distance", "µm"),
    ("B", "cell_area", METRIC_LABELS["cell_area"], METRIC_UNITS["cell_area"]),
    (
        "C",
        "boundary_length",
        METRIC_LABELS["boundary_length"],
        METRIC_UNITS["boundary_length"],
    ),
]


def group_means_with_ci(df, metric, level=0.95):
    summary = (
        df.dropna(subset=[metric])
        .groupby("af_type")[metric]
        .agg(["count", "mean", "std"])
        .loc[CLASS_ORDER]
    )
    n = summary["count"]
    t_crit = stats.t.ppf(0.5 + level / 2, df=n - 1)
    summary["half_width"] = t_crit * summary["std"] / np.sqrt(n)
    return summary


def plot_panel(ax, summary, title, unit, letter):
    x = np.arange(len(CLASS_ORDER))
    group_palette = build_group_palette()

    for i, af_type in enumerate(CLASS_ORDER):
        ax.errorbar(
            x[i],
            summary.loc[af_type, "mean"],
            yerr=summary.loc[af_type, "half_width"],
            fmt="o",
            color=group_palette[af_type],
            markeredgecolor="black",
            markersize=12,
            elinewidth=2,
            capsize=16,
            capthick=2,
            zorder=3,
        )

    ax.set_xticks(x, CLASS_ORDER)
    ax.set_xlim(-0.6, len(CLASS_ORDER) - 0.4)
    ax.set_title(title, fontsize=30, fontweight="bold")
    ax.set_xlabel("Group", fontsize=25, labelpad=14)
    ax.set_ylabel(f"Mean ± 95% CI ({unit})", fontsize=23)
    ax.tick_params(axis="x", labelsize=23)
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(axis="both", length=8, width=1.5)
    ax.text(
        -0.2,
        1.02,
        letter,
        transform=ax.transAxes,
        fontsize=37,
        fontweight="bold",
        va="bottom",
    )


def main():
    df = load_data()
    data_by_metric = {
        "full_line_length": df,
        "cell_area": load_area_data(df),
        "boundary_length": df,
    }

    fig, axes = plt.subplots(1, 3, figsize=(30, 8), constrained_layout=True)

    for ax, (letter, metric, title, unit) in zip(axes, PANELS):
        summary = group_means_with_ci(data_by_metric[metric], metric)
        print(f"{metric}:\n{summary.round(2)}\n")
        plot_panel(ax, summary, title, unit, letter)

    save_path = "group_means_95ci.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved -> {pdf_path}")
    plt.show()


if __name__ == "__main__":
    main()
