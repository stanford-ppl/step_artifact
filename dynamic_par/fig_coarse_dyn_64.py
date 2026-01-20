import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Color setup using tab20 palette
tab20_colors = sns.color_palette('tab20c')
colors = [mcolors.to_hex(tab20_colors[0]), mcolors.to_hex(tab20_colors[4])]
edgecolor = "black"

# Configuration
figsize = (8, 5)
legend_font = 40
xlabel_font = 40
ylabel_font = 40
tick_font = 40

# Path to your CSV file
csv_path = "./dynamic_par/batch_sweep_coarse_vs_dynamic.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Sort by batch (optional but recommended)
df = df.sort_values("batch")

# Create figure and axis
fig, ax = plt.subplots(figsize=figsize)

# Axes spine thickness
thickness = 2.5
for p in ["left", "right", "top", "bottom"]:
    ax.spines[p].set_linewidth(thickness)
    ax.spines[p].set_color("black")

# Plot lines with enhanced styling
ax.plot(df["batch"], df["static_coarse_cycles"],
        marker='o',
        label='Static\n(Coarse)',
        linewidth=3.5,
        markersize=12,
        color=colors[0],
        markeredgecolor=edgecolor,
        markeredgewidth=2.0,
        alpha=0.8)

ax.plot(df["batch"], df["dynamic_cycles"],
        marker='s',
        label='Dyn.',
        linewidth=3.5,
        markersize=12,
        color=colors[1],
        markeredgecolor=edgecolor,
        markeredgewidth=2.0,
        alpha=0.8)

# Labels and formatting
ax.set_xlabel('Batch Size', fontsize=xlabel_font)
ax.set_ylabel('Latency\n(Cycles)', fontsize=ylabel_font,labelpad=20)

# Set y-axis to log scale
ax.set_yscale('log')

# Use scientific notation for y-axis with 10^4 scale
# ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y/1e4)}' if y/1e4 == int(y/1e4) else f'{y/1e4:.1f}'))

from matplotlib.ticker import FixedLocator, FuncFormatter

scale = 1e4

# Specify exactly which y-ticks you want (in *actual* values)
yticks = [2e4, 4e4, 6e4]

ax.yaxis.set_major_locator(FixedLocator(yticks))
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda y, _: f'{int(y/scale)}')
)

# Optional: turn off minor ticks entirely (cleaner)
ax.yaxis.set_minor_locator(plt.NullLocator())

ax.text(0.1, 1.0, r'1e4',
        transform=ax.transAxes,
        fontsize=ylabel_font,
        va='bottom',
        ha='right')

# Set x-axis ticks to show all data points
ax.set_xticks(df['batch'])

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=1.5, axis='y')

# Tick parameters
# ax.tick_params(axis='y', labelsize=ylabel_font)
ax.tick_params(axis='y', which='both', labelsize=ylabel_font)

ax.tick_params(axis='x', labelsize=xlabel_font)

# Use colors from palette for arrow and text
annotation_color_1 = mcolors.to_hex(tab20_colors[4])

# Add annotation for x=16 showing speedup
row_16 = df[df['batch'] == 16].iloc[0]
static_val_16 = row_16['static_coarse_cycles']
dynamic_val_16 = row_16['dynamic_cycles']
speedup_16 = static_val_16 / dynamic_val_16

# Draw arrow between the two points at x=16
ax.annotate('',
            xy=(16, dynamic_val_16),
            xytext=(16, static_val_16),
            arrowprops=dict(arrowstyle='<->, head_width=0.6, head_length=0.8',
                          color=annotation_color_1, lw=2.5))

# Add text showing the speedup value for x=16
mid_y_16 = (static_val_16 + dynamic_val_16) / 2
ax.text(16 + 2, mid_y_16, f'{speedup_16:.2f}x',
        fontsize=xlabel_font,
        va='center',
        ha='left',
        color=annotation_color_1)

# Legend - top center with 2 columns (inside plot area)
# ax.legend(fontsize=legend_font,
#           frameon=False,
#           loc='upper center',
#           handletextpad=0.5,
#           bbox_to_anchor=(0.5, 1.0),
#           ncol=2)


# plt.tight_layout()
# --- Legend outside, above axes (no overlap) ---
handles, labels = ax.get_legend_handles_labels()

# Make room at the top for the legend
top_margin = 0.86   # try 0.88 → 0.83 range
fig.subplots_adjust(top=top_margin)  # tweak: smaller -> more room, larger -> less room

fig.legend(handles, labels,
           fontsize=legend_font,
           frameon=False,
           loc='upper center',
           bbox_to_anchor=(0.5, 1.2),  # ← move upward
           ncol=2,
           handletextpad=0.5,
           columnspacing=1.0)

# Use tight_layout but keep the reserved top space
plt.tight_layout(rect=[0, 0, 1, top_margin])

# Show plot
plt.savefig(f'./dynamic_par/figure15.pdf', bbox_inches='tight')


