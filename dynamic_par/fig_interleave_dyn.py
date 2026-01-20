import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns

# ------------------------
# Style defaults (self-contained)
# ------------------------
tab10_colors = sns.color_palette('tab10')
colors = [mcolors.to_hex(tab10_colors[0]), mcolors.to_hex(tab10_colors[1]), mcolors.to_hex(tab10_colors[2])]
edgecolor = "black"

def create_speedup_bar_chart(
    csv_file,
    figsize=(14, 7),
    ylim=(0, 2.5),
    ylabel="Speedup",
):
    """
    Creates a bar chart showing speedup (static_interleave/dynamic) vs. stdev.
    Each row in the CSV becomes a separate bar.
    """

    legend_font = 36 + 12
    xlabel_font = 40 + 12
    ylabel_font = 42 + 12

    # Read CSV
    df = pd.read_csv(csv_file)

    # Calculate speedup for each row
    df['speedup'] = df['static_interleave_cycles'] / df['dynamic_cycles']

    # Sort by stdev ascending
    # df = df.sort_values('stdev', ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Axes spine thickness
    thickness = 2.5
    for p in ["left", "right", "top", "bottom"]:
        ax.spines[p].set_linewidth(thickness)
        ax.spines[p].set_color("black")

    # Create bars
    x = np.arange(len(df))
    width = 0.6

    bars = ax.bar(
        x,
        df['speedup'].values,
        width,
        alpha=0.8,
        zorder=2,
        linewidth=3.0,
        edgecolor=edgecolor,
        color=colors[1],  # Use the interleave color from reference
    )

    # Add text and arrows for bars exceeding y-axis limit
    for rect in bars:
        val = rect.get_height()
        if val > ylim[1]:
            x_pos = rect.get_x() + rect.get_width()/2
            ax.text(x_pos, ylim[1] - 0.3, str(round(val, 2)),
                    fontsize=xlabel_font*0.8, color="black", ha='center')
            ax.annotate("",
                xy=(x_pos, ylim[1] - 0.05),
                xytext=(x_pos, ylim[1] - 0.25),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))

    # Set x-axis labels - one label per group of 3 bars using kv_length_var
    # Group every 3 rows
    group_labels = []
    group_positions = []
    for i in range(0, len(df), 3):
        if i < len(df):
            # Use the kv_length_var from the first row in each group
            label = df.iloc[i]['kv_length_var'].capitalize()
            group_labels.append(label)
            # Position label at the center of the 3 bars
            group_positions.append(i + 1)  # Center of 3 bars (0, 1, 2 -> center at 1)

    ax.set_xticks(group_positions)
    ax.set_xticklabels(group_labels, ha="center", linespacing=1.8)

    # Add vertical lines to separate groups
    for i in range(3, len(df), 3):
        ax.axvline(x=i - 0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=1)

    # Grid on y only
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=1.5, axis="y")

    # Add horizontal line at y=1.0
    ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=2, alpha=0.7, zorder=1)

    # Tick parameters
    ax.tick_params(axis="y", labelsize=ylabel_font)
    ax.tick_params(axis="x", labelsize=xlabel_font)

    # Labels
    ax.set_xlabel("KV $ Length Var.", fontsize=xlabel_font, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=xlabel_font)
    ax.yaxis.set_label_coords(-0.13, 0.4)  # Move label down along y-axis (adjust 0.4 to move up/down)

    # Y limits and formatting
    ax.set_ylim(ylim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.1f}"))

    plt.tight_layout()
    return fig


# ======================
# Main execution
# ======================
csv_file = "./dynamic_par/batch64_interleave_dynamic.csv"

# Create and show
fig = create_speedup_bar_chart(
    csv_file=csv_file,
    figsize=(13, 7),
    ylim=(0.8, 1.75),
    ylabel="Speedup\n(vs. Static Intrlv.)",
)

plt.show()

# Save
fig.savefig("./dynamic_par/figure14.pdf", bbox_inches="tight")
