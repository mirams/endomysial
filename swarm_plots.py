import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CLASS_ORDER = ["Control", "Paroxysmal AF", "Persistent AF"]
CLASS_DISPLAY = {
    "Control": "Control",
    "Paro AF": "Paroxysmal AF",
    "Pers AF": "Persistent AF",
}
CLASS_BG_COLOUR = {
    "Control": "#4C72B0",
    "Paroxysmal AF": "#55A868",
    "Persistent AF": "#C44E52",
}
CLASS_PALETTES = {
    "Control": "Blues",
    "Paroxysmal AF": "Greens",
    "Persistent AF": "Reds",
}
METRIC = "boundary_length"
METRIC_LABELS = {"boundary_length": "Boundary to boundary distance",
                 "cell_area": "Cell area"}
METRIC_UNITS = {"boundary_length": "µm", "cell_area": "µm²"}


def _axis_label(ylabel, metric):
    unit = METRIC_UNITS.get(metric)
    return f"{ylabel} ({unit})" if unit else ylabel


def build_patient_palette(patient_metadata):
    patient_colour = {}

    for af_type in CLASS_ORDER:
        patient_ids = patient_metadata.loc[
            patient_metadata["af_type"] == af_type,
            "top_level_folder",
        ].tolist()

        if not patient_ids:
            continue

        # Use darker shades so swarm points remain
        # distinct from class backgrounds.
        palette = sns.color_palette(
            CLASS_PALETTES[af_type],
            n_colors=len(patient_ids) + 3,
        )[-len(patient_ids):]
        patient_colour.update(dict(zip(patient_ids, palette)))

    return patient_colour


def build_group_palette():
    return {
        af_type: sns.color_palette(CLASS_PALETTES[af_type], n_colors=6)[-1]
        for af_type in CLASS_ORDER
    }


def add_class_backgrounds(ax, ordered_labels, label_to_class):
    start_index = 0

    while start_index < len(ordered_labels):
        af_type = label_to_class[ordered_labels[start_index]]
        end_index = start_index

        while end_index + 1 < len(ordered_labels):
            next_label = ordered_labels[end_index + 1]
            if label_to_class[next_label] != af_type:
                break
            end_index += 1

        ax.axvspan(
            start_index - 0.5,
            end_index + 0.5,
            color=CLASS_BG_COLOUR[af_type],
            alpha=0.10,
            zorder=0,
        )

        if end_index < len(ordered_labels) - 1:
            ax.axvline(
                end_index + 0.5,
                color=CLASS_BG_COLOUR[af_type],
                linewidth=1.2,
                linestyle="--",
                alpha=0.65,
                zorder=1,
            )

        start_index = end_index + 1


def patient_labels(df):
    return dict(zip(df["top_level_folder"], df["patient_label"]))


def build_patient_handles(patient_order, patient_colour, patient_to_label):
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=patient_colour[patient_id],
            markersize=14,
            label=f"Patient {patient_to_label[patient_id]}",
        )
        for patient_id in patient_order
    ]


def build_class_handles():
    return [
        mpatches.Patch(
            facecolor=CLASS_BG_COLOUR[af_type],
            alpha=0.25,
            label=af_type,
        )
        for af_type in CLASS_ORDER
    ]


def add_legends(ax, patient_order, patient_colour, patient_to_label):
    patient_handles = build_patient_handles(
        patient_order, patient_colour, patient_to_label
    )
    class_handles = build_class_handles()

    patient_legend = ax.legend(
        handles=patient_handles,
        title="Patient",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        framealpha=0.9,
        fontsize=16,
    )
    ax.add_artist(patient_legend)

    ax.legend(
        handles=class_handles,
        title="Group",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.18),
        borderaxespad=0,
        framealpha=0.9,
        fontsize=16,
    )


def add_central_legend(
    legend_ax, patient_order, patient_colour, patient_to_label
):
    legend_ax.axis("off")
    spacer = mpatches.Patch(facecolor="none", edgecolor="none", label="")
    handles = (
        build_patient_handles(patient_order, patient_colour, patient_to_label)
        + [spacer]
        + build_class_handles()
    )
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=-(-len(handles) // 2),
        mode="expand",
        bbox_to_anchor=(0, 0, 1, 1),
        frameon=False,
        fontsize=18,
        columnspacing=1.0,
        handletextpad=0.4,
    )


def load_data():
    df = pd.read_csv("czi_annotation_measurements.csv")
    class_df = pd.read_csv("Data/patient_classes.txt").rename(
        columns={
            "Patient number": "top_level_folder",
            "Label_number": "patient_label",
            "Group": "af_type_raw",
        }
    )
    class_df["af_type"] = class_df["af_type_raw"].map(CLASS_DISPLAY)

    if class_df["af_type"].isna().any():
        unknown_groups = sorted(
            class_df.loc[class_df["af_type"].isna(), "af_type_raw"].unique()
        )
        raise ValueError(
            "Unknown patient groups in Data/patient_classes.txt: "
            f"{unknown_groups}"
        )

    df = df.merge(
        class_df[["top_level_folder", "patient_label", "af_type"]],
        on="top_level_folder",
        how="left",
    )

    if df["af_type"].isna().any():
        missing_patients = sorted(
            df.loc[df["af_type"].isna(), "top_level_folder"].unique()
        )
        raise ValueError(
            f"Missing patient class assignments for: {missing_patients}"
        )

    df["af_rank"] = df["af_type"].map(
        {group: index for index, group in enumerate(CLASS_ORDER)}
    )
    df["group"] = (
        df["patient_label"].astype(str)
        + " / S"
        + df["slice_number"].astype(str)
    )
    df = df.sort_values(
        ["af_rank", "top_level_folder", "slice_number", "annotation_number"]
    )

    return df


def prepare_orders(df):
    patient_metadata = (
        df[["top_level_folder", "af_type", "af_rank"]]
        .drop_duplicates()
        .sort_values(["af_rank", "top_level_folder"])
    )
    patient_order = patient_metadata["top_level_folder"].tolist()
    # Keep the dataframe's existing sort order:
    # group first, then patient, then slice.
    group_order = df["group"].drop_duplicates().tolist()
    patient_colour = build_patient_palette(patient_metadata)
    group_to_colour = {
        group_label: patient_colour[patient_id]
        for group_label, patient_id in zip(df["group"], df["top_level_folder"])
    }
    group_to_colour = {
        group_label: group_to_colour[group_label]
        for group_label in group_order
    }
    group_to_class = {
        group_label: df.loc[df["group"] == group_label, "af_type"].iloc[0]
        for group_label in group_order
    }
    patient_to_class = dict(
        zip(patient_metadata["top_level_folder"], patient_metadata["af_type"])
    )

    return (
        patient_order,
        group_order,
        patient_colour,
        group_to_colour,
        group_to_class,
        patient_to_class,
    )


def plot_by_slice(
    df,
    patient_order,
    group_order,
    patient_colour,
    group_to_colour,
    group_to_class,
    metric=METRIC,
    ylabel=None,
    size=7.5,
    save_path="swarm_plots_boundary.png",
    ax=None,
    show_legend=True,
):
    ylabel = ylabel or METRIC_LABELS.get(
        metric, metric.replace("_", " ").title())
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=(max(14, len(group_order) * 0.6), 5),
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    add_class_backgrounds(ax, group_order, group_to_class)

    sns.swarmplot(
        data=df,
        x="group",
        y=metric,
        hue="group",
        order=group_order,
        hue_order=group_order,
        palette=group_to_colour,
        size=size,
        legend=False,
        ax=ax,
    )
    sns.boxplot(
        data=df,
        x="group",
        y=metric,
        hue="group",
        order=group_order,
        hue_order=group_order,
        width=0.4,
        showfliers=False,
        linewidth=0.8,
        boxprops=dict(alpha=0.25),
        medianprops=dict(color="black", linewidth=1.5),
        palette=group_to_colour,
        legend=False,
        ax=ax,
    )

    prev_patient = None
    for i, group_label in enumerate(group_order):
        patient_id = group_label.split(" /")[0]
        if prev_patient is not None and patient_id != prev_patient:
            ax.axvline(
                i - 0.5,
                color="grey",
                linewidth=0.8,
                linestyle="--",
                alpha=0.7,
                zorder=2,
            )
        prev_patient = patient_id

    # ax.set_title(ylabel, fontsize=13, fontweight="bold")
    ax.set_xlabel("Patient / Slice", fontsize=20)
    ax.set_ylabel(_axis_label(ylabel, metric), fontsize=20)
    ax.tick_params(axis="x", rotation=55, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    if show_legend:
        add_legends(ax, patient_order, patient_colour, patient_labels(df))

    if standalone:
        fig.suptitle(
            "CZI Annotation Measurements - Swarm Plots by Patient & Slice",
            fontsize=28,
            fontweight="bold",
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
        plt.show()
    return ax


def plot_by_patient(
    df,
    patient_order,
    patient_colour,
    patient_to_class,
    metric=METRIC,
    ylabel=None,
    size=7.5,
    save_path="swarm_plots_boundary_by_patient.png",
    ax=None,
    show_legend=True,
):
    ylabel = ylabel or METRIC_LABELS.get(
        metric, metric.replace("_", " ").title())
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=(max(8, len(patient_order) * 0.8), 5),
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    add_class_backgrounds(ax, patient_order, patient_to_class)

    sns.swarmplot(
        data=df,
        x="top_level_folder",
        y=metric,
        hue="top_level_folder",
        order=patient_order,
        hue_order=patient_order,
        palette=patient_colour,
        size=size,
        legend=False,
        ax=ax,
    )
    sns.boxplot(
        data=df,
        x="top_level_folder",
        y=metric,
        hue="top_level_folder",
        order=patient_order,
        hue_order=patient_order,
        width=0.4,
        showfliers=False,
        linewidth=0.8,
        boxprops=dict(alpha=0.25),
        medianprops=dict(color="black", linewidth=1.5),
        palette=patient_colour,
        legend=False,
        ax=ax,
    )

    patient_to_label = patient_labels(df)
    ax.set_xticks(
        range(len(patient_order)),
        [patient_to_label[patient_id] for patient_id in patient_order],
    )

    ax.set_title(
        f"by Patient",
        fontsize=26,
        fontweight="bold",
    )
    ax.set_xlabel("Patient", fontsize=20)
    ax.set_ylabel(_axis_label(ylabel, metric), fontsize=20)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=16)
    if show_legend:
        add_legends(ax, patient_order, patient_colour, patient_to_label)

    if standalone:
        fig.suptitle(
            "CZI Annotation Measurements - Swarm Plots by Patient",
            fontsize=28,
            fontweight="bold",
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
        plt.show()
    return ax


def plot_by_group(
    df,
    metric=METRIC,
    ylabel=None,
    size=7.5,
    save_path="swarm_plots_boundary_by_group.png",
    ax=None,
):
    ylabel = ylabel or METRIC_LABELS.get(
        metric, metric.replace("_", " ").title())
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=(8, 5),
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    group_palette = build_group_palette()

    sns.boxplot(
        data=df,
        x="af_type",
        y=metric,
        hue="af_type",
        order=CLASS_ORDER,
        hue_order=CLASS_ORDER,
        width=0.42,
        showfliers=False,
        linewidth=0.9,
        boxprops=dict(alpha=0.28),
        whiskerprops=dict(alpha=0.4),
        capprops=dict(alpha=0.4),
        medianprops=dict(color="black", linewidth=1.6),
        palette=group_palette,
        legend=False,
        ax=ax,
    )
    sns.swarmplot(
        data=df,
        x="af_type",
        y=metric,
        hue="af_type",
        order=CLASS_ORDER,
        hue_order=CLASS_ORDER,
        palette=group_palette,
        size=size,
        alpha=0.5,
        legend=False,
        ax=ax,
    )

    ax.set_title(
        f"by Group",
        fontsize=26,
        fontweight="bold",
    )
    ax.set_xlabel("Group", fontsize=20)
    ax.set_ylabel(_axis_label(ylabel, metric), fontsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=16)

    if standalone:
        fig.suptitle(
            "CZI Annotation Measurements - Swarm Plot by Group",
            fontsize=28,
            fontweight="bold",
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
        plt.show()
    return ax


def plot_combined_swarm_panels(
    df,
    patient_order,
    group_order,
    patient_colour,
    group_to_colour,
    group_to_class,
    patient_to_class,
    metric=METRIC,
    ylabel=None,
    size=7.5,
    save_path="swarm_plots_combined.png",
):
    ylabel = ylabel or METRIC_LABELS.get(
        metric, metric.replace("_", " ").title())

    fig = plt.figure(figsize=(26, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.35, 1])
    ax_slice = fig.add_subplot(gs[0, :])
    legend_ax = fig.add_subplot(gs[1, :])
    ax_patient = fig.add_subplot(gs[2, 0])
    ax_group = fig.add_subplot(gs[2, 1])

    plot_by_slice(
        df,
        patient_order,
        group_order,
        patient_colour,
        group_to_colour,
        group_to_class,
        metric=metric,
        ylabel=ylabel,
        size=size,
        ax=ax_slice,
        show_legend=False,
    )
    plot_by_patient(
        df,
        patient_order,
        patient_colour,
        patient_to_class,
        metric=metric,
        ylabel=ylabel,
        size=size,
        ax=ax_patient,
        show_legend=False,
    )
    plot_by_group(
        df,
        metric=metric,
        ylabel=ylabel,
        size=size,
        ax=ax_group,
    )

    add_central_legend(
        legend_ax, patient_order, patient_colour, patient_labels(df)
    )

    fig.suptitle(
        f"{ylabel} swarm plots",
        fontsize=32,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    pdf_path = save_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved -> {pdf_path}")
    plt.show()


def plot_mean_area_vs_boundary_panels(
    df,
    patient_order,
    group_order,
    patient_colour,
    group_to_colour,
    save_path="scatter_mean_area_vs_boundary_panels.png",
):
    scatter_df = df.copy()
    scatter_df["mean_cell_area"] = scatter_df[["cell_a_area", "cell_b_area"]].mean(
        axis=1
    )
    scatter_df = scatter_df.dropna(
        subset=["boundary_length", "mean_cell_area"]
    )

    group_palette = build_group_palette()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    sns.scatterplot(
        data=scatter_df,
        x="boundary_length",
        y="mean_cell_area",
        hue="group",
        hue_order=group_order,
        palette=group_to_colour,
        s=36,
        alpha=0.85,
        linewidth=0,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Coloured by Patient / Slice",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Inter-boundary Distance (µm)")
    axes[0].set_ylabel("Mean Cell Area (µm²)")

    sns.scatterplot(
        data=scatter_df,
        x="boundary_length",
        y="mean_cell_area",
        hue="top_level_folder",
        hue_order=patient_order,
        palette=patient_colour,
        s=36,
        alpha=0.85,
        linewidth=0,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Coloured by Patient", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Inter-boundary Distance (µm)")
    axes[1].set_ylabel("Mean Cell Area (µm²)")

    sns.scatterplot(
        data=scatter_df,
        x="boundary_length",
        y="mean_cell_area",
        hue="af_type",
        hue_order=CLASS_ORDER,
        palette=group_palette,
        s=38,
        alpha=0.9,
        linewidth=0,
        ax=axes[2],
    )
    axes[2].set_title("Coloured by Group", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Inter-boundary Distance (µm)")
    axes[2].set_ylabel("Mean Cell Area (µm²)")
    axes[2].legend(
        title="Group",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        framealpha=0.9,
        fontsize=16,
    )

    fig.suptitle(
        "Mean Cell Area vs Inter-boundary Distance",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    plt.show()


def load_area_data(df):
    """Melt cell_a_area and cell_b_area into a single cell_area column."""
    id_cols = [c for c in df.columns if c not in (
        "cell_a_area", "cell_b_area")]
    return (
        df.melt(
            id_vars=id_cols,
            value_vars=["cell_a_area", "cell_b_area"],
            var_name="cell_label",
            value_name="cell_area",
        )
        .dropna(subset=["cell_area"])
        .reset_index(drop=True)
    )


def main():
    df = load_data()
    (
        patient_order,
        group_order,
        patient_colour,
        group_to_colour,
        group_to_class,
        patient_to_class,
    ) = prepare_orders(df)

    plot_mean_area_vs_boundary_panels(
        df,
        patient_order,
        group_order,
        patient_colour,
        group_to_colour,
    )

    plot_combined_swarm_panels(
        df,
        patient_order,
        group_order,
        patient_colour,
        group_to_colour,
        group_to_class,
        patient_to_class,
        save_path="swarm_plots_boundary_combined.png",
    )

    area_df = load_area_data(df)
    (
        patient_order,
        group_order,
        patient_colour,
        group_to_colour,
        group_to_class,
        patient_to_class,
    ) = prepare_orders(area_df)

    plot_combined_swarm_panels(
        area_df,
        patient_order,
        group_order,
        patient_colour,
        group_to_colour,
        group_to_class,
        patient_to_class,
        metric="cell_area",
        size=4.5,
        save_path="swarm_plots_cell_area_combined.png",
    )


if __name__ == "__main__":
    main()
