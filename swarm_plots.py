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
METRIC = "boundary_boundary_length"


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


def add_legends(ax, patient_order, patient_colour):
    patient_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=patient_colour[patient_id],
            markersize=8,
            label=f"Patient {patient_id}",
        )
        for patient_id in patient_order
    ]
    class_handles = [
        mpatches.Patch(
            facecolor=CLASS_BG_COLOUR[af_type],
            alpha=0.25,
            label=af_type,
        )
        for af_type in CLASS_ORDER
    ]

    patient_legend = ax.legend(
        handles=patient_handles,
        title="Patient",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        framealpha=0.9,
        fontsize=8,
    )
    ax.add_artist(patient_legend)

    ax.legend(
        handles=class_handles,
        title="Group",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.18),
        borderaxespad=0,
        framealpha=0.9,
        fontsize=8,
    )


def load_data():
    df = pd.read_csv("czi_annotation_measurements.csv")
    class_df = pd.read_csv("Data/patient_classes.txt").rename(
        columns={
            "Patient number": "top_level_folder",
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
        class_df[["top_level_folder", "af_type"]],
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
        df["top_level_folder"].astype(str)
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
    save_path="swarm_plots.png",
):
    ylabel = ylabel or metric.replace("_", " ").title()
    fig, ax = plt.subplots(
        figsize=(max(14, len(group_order) * 0.6), 5),
        constrained_layout=True,
    )

    add_class_backgrounds(ax, group_order, group_to_class)

    sns.swarmplot(
        data=df,
        x="group",
        y=metric,
        hue="group",
        order=group_order,
        hue_order=group_order,
        palette=group_to_colour,
        size=5,
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
        fliersize=0,
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

    ax.set_title(ylabel, fontsize=13, fontweight="bold")
    ax.set_xlabel("Patient / Slice", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    add_legends(ax, patient_order, patient_colour)

    fig.suptitle(
        "CZI Annotation Measurements - Swarm Plots by Patient & Slice",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    plt.show()


def plot_by_patient(
    df,
    patient_order,
    patient_colour,
    patient_to_class,
    metric=METRIC,
    ylabel=None,
    save_path="swarm_plots_by_patient.png",
):
    ylabel = ylabel or metric.replace("_", " ").title()
    fig, ax = plt.subplots(
        figsize=(max(8, len(patient_order) * 0.8), 5),
        constrained_layout=True,
    )

    add_class_backgrounds(ax, patient_order, patient_to_class)

    sns.swarmplot(
        data=df,
        x="top_level_folder",
        y=metric,
        hue="top_level_folder",
        order=patient_order,
        hue_order=patient_order,
        palette=patient_colour,
        size=5,
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
        fliersize=0,
        linewidth=0.8,
        boxprops=dict(alpha=0.25),
        medianprops=dict(color="black", linewidth=1.5),
        palette=patient_colour,
        legend=False,
        ax=ax,
    )

    ax.set_title(
        f"{ylabel} by Patient",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Patient", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    add_legends(ax, patient_order, patient_colour)

    fig.suptitle(
        "CZI Annotation Measurements - Swarm Plots by Patient",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    plt.show()


def plot_by_group(df, metric=METRIC, ylabel=None, save_path="swarm_plots_by_group.png"):
    ylabel = ylabel or metric.replace("_", " ").title()
    fig, ax = plt.subplots(
        figsize=(8, 5),
        constrained_layout=True,
    )

    group_palette = {
        af_type: sns.color_palette(CLASS_PALETTES[af_type], n_colors=6)[-1]
        for af_type in CLASS_ORDER
    }

    sns.swarmplot(
        data=df,
        x="af_type",
        y=metric,
        hue="af_type",
        order=CLASS_ORDER,
        hue_order=CLASS_ORDER,
        palette=group_palette,
        size=5,
        legend=False,
        ax=ax,
    )
    sns.boxplot(
        data=df,
        x="af_type",
        y=metric,
        hue="af_type",
        order=CLASS_ORDER,
        hue_order=CLASS_ORDER,
        width=0.42,
        fliersize=0,
        linewidth=0.9,
        boxprops=dict(alpha=0.28),
        medianprops=dict(color="black", linewidth=1.6),
        palette=group_palette,
        legend=False,
        ax=ax,
    )

    ax.set_title(
        f"{ylabel} by Group",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Group", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="x", labelsize=10)

    handles = [
        mpatches.Patch(
            facecolor=group_palette[af_type],
            alpha=0.35,
            label=af_type,
        )
        for af_type in CLASS_ORDER
    ]

    fig.suptitle(
        "CZI Annotation Measurements - Swarm Plot by Group",
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

    plot_by_slice(
        df,
        patient_order,
        group_order,
        patient_colour,
        group_to_colour,
        group_to_class,
    )
    plot_by_patient(df, patient_order, patient_colour, patient_to_class)
    plot_by_group(df)

    area_df = load_area_data(df)
    (
        patient_order,
        group_order,
        patient_colour,
        group_to_colour,
        group_to_class,
        patient_to_class,
    ) = prepare_orders(area_df)

    plot_by_slice(
        area_df,
        patient_order,
        group_order,
        patient_colour,
        group_to_colour,
        group_to_class,
        metric="cell_area",
        ylabel="Cell Area (µm²)",
        save_path="swarm_plots_cell_area.png",
    )
    plot_by_patient(
        area_df,
        patient_order,
        patient_colour,
        patient_to_class,
        metric="cell_area",
        ylabel="Cell Area (µm²)",
        save_path="swarm_plots_cell_area_by_patient.png",
    )
    plot_by_group(
        area_df,
        metric="cell_area",
        ylabel="Cell Area (µm²)",
        save_path="swarm_plots_cell_area_by_group.png",
    )


if __name__ == "__main__":
    main()
