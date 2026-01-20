import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns

# ------------------------
# Style defaults (self-contained)
# ------------------------
tab10_colors = sns.color_palette('tab10')
colors = [mcolors.to_hex(tab10_colors[0]),mcolors.to_hex(tab10_colors[1]),mcolors.to_hex(tab10_colors[2])]
edgecolor = "black"

def create_multi_csv_bar_chart_arxiv(
    csv_files,
    x_labels_list,
    subplot_titles,
    figsize=(16, 6),
    ylim=(0, 4),
    ylabel="Speedup",
):
    """
    Creates multiple grouped bar charts from different CSV files with shared y-axis.
    """

    legend_font = 36+12
    xlabel_font = 40+12
    ylabel_font = 42+12
    subtitle_font = 36

    num_plots = len(csv_files)

    # width ratios by number of x-ticks in each subplot
    width_ratios = [len(labels) for labels in x_labels_list]

    fig, axes = plt.subplots(
        1,
        num_plots,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.0},
    )

    if num_plots == 1:
        axes = [axes]

    # Axes spine thickness
    thickness = 2.5
    for ax in axes:
        for p in ["left", "right", "top", "bottom"]:
            ax.spines[p].set_linewidth(thickness)
            ax.spines[p].set_color("black")

    # Draw subplots
    for i, (ax, csv_file, x_labels, subplot_title) in enumerate(
        zip(axes, csv_files, x_labels_list, subplot_titles)
    ):
        df = pd.read_csv(csv_file)

        # Expect these columns in each CSV
        cycles = df["static_coarse_cycles"].values
        interleave = df["static_interleave_cycles"].values
        dynamic = df["dynamic_cycles"].values

        x = np.arange(len(x_labels))
        width = 0.25

        # Bars (legend only on first axis)
        ax.bar(
            x - width,
            cycles,
            width,
            label="Static (Coarse)" if i == 0 else "",
            alpha=0.8,
            zorder=2,
            linewidth=3.0,
            edgecolor=edgecolor,
            color=colors[0],
        )
        ax.bar(
            x,
            interleave,
            width,
            label="Static (Interleave)" if i == 0 else "",
            alpha=0.8,
            zorder=2,
            linewidth=3.0,
            edgecolor=edgecolor,
            color=colors[1],
        )
        ax.bar(
            x + width,
            dynamic,
            width,
            label="Dynamic" if i == 0 else "",
            alpha=0.8,
            zorder=2,
            linewidth=3.0,
            edgecolor=edgecolor,
            color=colors[2],
        )

        ax.set_xticks(x)
        # ---- SPACING FIX 1: more space within multi-line tick labels ----
        ax.set_xticklabels(x_labels, ha="center", linespacing=1.8)

        # Grid on y only
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=1.5, axis="y")

        ax.tick_params(axis="y", labelsize=ylabel_font)
        ax.tick_params(axis="x", labelsize=xlabel_font)

        # ---- SPACING FIX 2: more space between tick labels and "B=.." label ----
        ax.set_xlabel(subplot_title, fontsize=xlabel_font, fontweight="bold", labelpad=20)

        if i == 0:
            ax.set_ylabel(ylabel, fontsize=xlabel_font)
            # ax.yaxis.set_label_coords(-0.15, 0.25)  # ↓ move label downward
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    # Shared y limits and formatting
    axes[0].set_ylim(ylim)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.1f}"))

    # One legend for all
    handles, labels = axes[0].get_legend_handles_labels()
    axbox = axes[0].get_position()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=[axbox.x0 + 1.05 * axbox.width + 0.59, axbox.y0 + axbox.height + 0.29],
        ncol=3,
        fontsize=legend_font,
        borderaxespad=0.3,
        handletextpad=0.3,
        frameon=False,
    )

    plt.tight_layout()
    return fig


# ======================
# Inputs (adjust paths as needed)
# ======================
csv_files = [
    "./dynamic_par/batch16_sweep_ae.csv",
    "./dynamic_par/batch64_sweep_ae.csv",
    "./dynamic_par/batch80_sweep_ae.csv",
]

# Multi-line tick labels: first line (High/Med/Low), second line descriptor
x_labels_list = [
    # ["High", "Med\nKV cache length variation", "Low"],
    # ["High", "Med\nKV cache length variation", "Low"],
    # ["High", "Med\nKV cache length variation", "Low"],
    ["High", "Med\nKV $ Length Var.", "Low"],
    ["High", "Med\nKV $ Length Var.", "Low"],
    ["High", "Med\nKV $ Length Var.", "Low"],
]

subplot_titles = [
    "B=16",
    "B=64",
    "B=64+16",
]

# Create and show
fig = create_multi_csv_bar_chart_arxiv(
    csv_files=csv_files,
    x_labels_list=x_labels_list,
    subplot_titles=subplot_titles,
    figsize=(26, 6),
    ylim=(0.5, 3.5),
    ylabel="Normalized\nCycles",
)

plt.show()

# Save
fig.savefig("./dynamic_par/figure11.pdf", dpi=300, bbox_inches="tight")
fig.savefig("./dynamic_par/figure11.png", bbox_inches="tight")
