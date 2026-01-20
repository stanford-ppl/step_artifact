import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

def create_pareto_plot(csv_files, subplot_titles, figsize=(20, 9),
                       cycles_scale=None, on_chip_scale=None,
                       cycles_scale_list=None, on_chip_scale_list=None,
                       x_ticks_list=None, y_ticks_list=None,
                       x_scale_list=None, y_scale_list=None,
                       output_file=None,
                       arrow_linestyle='dotted'):
    """
    Create side-by-side Pareto frontier plots from multiple CSV files.
    X-axis: On-chip memory usage
    Y-axis: Cycle count

    Parameters:
    csv_files (list): List of CSV file paths
    subplot_titles (list): Title for each subplot
    figsize (tuple): Figure size (width, height)
    cycles_scale (tuple, optional): Y-axis scale for cycles (applied to all plots if cycles_scale_list not provided)
    on_chip_scale (tuple, optional): X-axis scale for on-chip memory (applied to all plots if on_chip_scale_list not provided)
    cycles_scale_list (list of tuples, optional): Y-axis scale for each subplot individually
    on_chip_scale_list (list of tuples, optional): X-axis scale for each subplot individually
    x_ticks_list (list of lists, optional): X-axis tick values for each subplot individually
    y_ticks_list (list of lists, optional): Y-axis tick values for each subplot individually
    x_scale_list (list of numbers, optional): X-axis scale factors (e.g., 1e7, 1e8) for each subplot
    y_scale_list (list of numbers, optional): Y-axis scale factors (e.g., 1e6, 1e5) for each subplot
    output_file (str, optional): Path to save the combined plot
    """
    font_uniform = 44
    axis_font = font_uniform
    legend_font = font_uniform
    tick_font = font_uniform
    x_tick_size = font_uniform
    subtitle_font = font_uniform

    # Set up color palette
    color_palette = sns.color_palette(palette='tab20c')

    num_plots = len(csv_files)

    # Create subplots
    fig, axes = plt.subplots(1, num_plots, figsize=figsize, gridspec_kw={'wspace': 0.3})

    # Ensure axes is always a list
    if num_plots == 1:
        axes = [axes]

    # Process each subplot
    for plot_idx, (csv_file, subplot_title) in enumerate(zip(csv_files, subplot_titles)):
        ax = axes[plot_idx]

        # Read CSV file
        df = pd.read_csv(csv_file)

        # Separate static and dynamic tiling
        df_static = df[df['tile_N'] != 'tile=dynamic'].copy()
        df_dynamic = df[df['tile_N'] == 'tile=dynamic']

        # Extract data for static tiles
        tile_sizes = df_static['tile_N'].str.replace('tile=', '').astype(int).values
        static_cycles = df_static['cycles'].values
        static_on_chip = df_static['on_chip_mem'].values

        # Extract data for dynamic tiling
        dynamic_cycles = df_dynamic['cycles'].values[0] if len(df_dynamic) > 0 else None
        dynamic_on_chip = df_dynamic['on_chip_mem'].values[0] if len(df_dynamic) > 0 else None

        # Plot the Pareto curve (connecting static tile points)
        ax.plot(static_on_chip, static_cycles,
                color=color_palette[0], linewidth=3,
                marker='o', markersize=12,
                label='Static Tiling' if plot_idx == 0 else "",
                zorder=2)

        # Annotate each point with tile size
        for i, (x, y, tile) in enumerate(zip(static_on_chip, static_cycles, tile_sizes)):
            if i == 3:
                ax.annotate(f'{tile}',
                       xy=(x, y),
                       xytext=(-45, -45),
                    #    rotation=30,
                       textcoords='offset points',
                       fontsize=x_tick_size)
            elif i == 0:
                ax.annotate(f'{tile}',
                       xy=(x, y),
                       xytext=(15, -15),
                    #    rotation=30,
                       textcoords='offset points',
                       fontsize=x_tick_size)
            else:
                ax.annotate(f'{tile}',
                       xy=(x, y),
                       xytext=(10, 10),
                    #    rotation=30,
                       textcoords='offset points',
                       fontsize=x_tick_size)

        # Plot dynamic tiling point
        if dynamic_cycles is not None and dynamic_on_chip is not None:
            ax.scatter(dynamic_on_chip, dynamic_cycles,
                      color=color_palette[4], s=300,
                      marker='o',  linewidths=2,
                      label='Dynamic Tiling' if plot_idx == 0 else "",
                      zorder=3)

            # Add arrows only to the first plot
            if plot_idx == 0:
                # Find tile=16, tile=32, and tile=64 points
                tile_16_idx = None
                tile_32_idx = None
                tile_64_idx = None
                for i, tile in enumerate(tile_sizes):
                    if tile == 16:
                        tile_16_idx = i
                    elif tile == 32:
                        tile_32_idx = i
                    elif tile == 64:
                        tile_64_idx = i

                dyn_x = dynamic_on_chip
                dyn_y = dynamic_cycles

                # Arrow 1: Performance speedup (dynamic vs tile=16)
                if tile_16_idx is not None:
                    tile_16_x = static_on_chip[tile_16_idx]
                    tile_16_y = static_cycles[tile_16_idx]

                    # Calculate speedup
                    speedup = round(tile_16_y / dyn_y, 1)

                    # Add double-headed arrow with large arrowheads
                    ax.annotate('', xy=(tile_16_x, tile_16_y), xytext=(dyn_x, dyn_y),
                               arrowprops=dict(arrowstyle='<-', lw=4, color=color_palette[8],
                                             mutation_scale=40, shrinkA=10, shrinkB=15,
                                             linestyle=arrow_linestyle))

                    # Add speedup text at the midpoint of the arrow
                    mid_x = (tile_16_x * dyn_x) ** 0.5  # Geometric mean for log scale
                    mid_y = (tile_16_y * dyn_y) ** 0.5  # Geometric mean for log scale
                    speedup_ratio = round(tile_16_y / dyn_y, 2)
                    ax.text(mid_x*0.99, mid_y, f'{speedup_ratio}×',
                           fontsize=x_tick_size, #fontweight='bold',
                           ha='right', va='center', color=color_palette[8])

                # Arrow 2: Memory usage ratio (dynamic vs tile=32)
                if tile_32_idx is not None:
                    tile_32_x = static_on_chip[tile_32_idx]
                    tile_32_y = static_cycles[tile_32_idx]

                    # Calculate memory ratio
                    memory_ratio = round(tile_32_x / dyn_x, 2)

                    # Add double-headed arrow with large arrowheads
                    ax.annotate('', xy=(tile_32_x, tile_32_y), xytext=(dyn_x, dyn_y),
                               arrowprops=dict(arrowstyle='<-', lw=4, color=color_palette[8],
                                             mutation_scale=40, shrinkA=10, shrinkB=15,
                                             linestyle=arrow_linestyle))

                    # Add memory ratio text at the midpoint of the arrow
                    mid_x = (tile_32_x * dyn_x) ** 0.5  # Geometric mean for log scale
                    mid_y = (tile_32_y * dyn_y) ** 0.5  # Geometric mean for log scale
                    # Offset the text lower by adjusting the y position
                    ax.text(mid_x, mid_y * 0.9, f'{memory_ratio}×',
                           fontsize=x_tick_size, #fontweight='bold',
                           ha='center', va='top', color=color_palette[8])

            # Add arrows only to the second plot (Qwen)
            if plot_idx == 1:
                # Find tile=8 and tile=64 points
                tile_8_idx = None
                tile_64_idx = None
                for i, tile in enumerate(tile_sizes):
                    if tile == 8:
                        tile_8_idx = i
                    elif tile == 64:
                        tile_64_idx = i

                dyn_x = dynamic_on_chip
                dyn_y = dynamic_cycles

                # Arrow 3a: Arrow from tile=8 to dynamic (memory ratio)
                if tile_8_idx is not None:
                    tile_8_x = static_on_chip[tile_8_idx]
                    tile_8_y = static_cycles[tile_8_idx]

                    # Calculate memory ratio
                    memory_ratio_8 = round(tile_8_x / dyn_x, 2)

                    # Offset the arrow slightly in log space to avoid overlap
                    # Adjust starting point slightly higher in y
                    dyn_y_offset1 = dyn_y * 1.02
                    tile_8_y_offset1 = tile_8_y * 1.02

                    # Add arrow from dynamic to tile=8
                    ax.annotate('', xy=(tile_8_x, tile_8_y_offset1), xytext=(dyn_x, dyn_y_offset1),
                               arrowprops=dict(arrowstyle='<-', lw=4, color=color_palette[8],
                                             mutation_scale=40, shrinkA=10, shrinkB=20,
                                             linestyle=arrow_linestyle))

                    # Add memory ratio text below the arrow
                    mid_x = (tile_8_x * dyn_x) ** 0.5  # Geometric mean for log scale
                    mid_y = (tile_8_y_offset1 * dyn_y_offset1) ** 0.5  # Geometric mean for log scale
                    ax.text(mid_x* 0.95, mid_y * 1.35, f'{memory_ratio_8}×',
                           fontsize=x_tick_size, #fontweight='bold',
                           ha='center', va='top', color=color_palette[12])

                # Arrow 3b: Arrow from tile=8 to dynamic (cycle ratio)
                if tile_8_idx is not None:
                    tile_8_x = static_on_chip[tile_8_idx]
                    tile_8_y = static_cycles[tile_8_idx]

                    # Calculate cycle ratio
                    cycle_ratio_8 = round(tile_8_y / dyn_y, 2)

                    # Offset the arrow slightly lower in y to separate from first arrow
                    dyn_y_offset2 = dyn_y * 0.98
                    tile_8_y_offset2 = tile_8_y * 0.98

                    # Add arrow from dynamic to tile=8
                    ax.annotate('', xy=(tile_8_x, tile_8_y_offset2), xytext=(dyn_x, dyn_y_offset2),
                               arrowprops=dict(arrowstyle='<-', lw=4, color=color_palette[12],
                                             mutation_scale=40, shrinkA=10, shrinkB=10,
                                             linestyle=arrow_linestyle))

                    # Add cycle ratio text on top of the arrow
                    mid_x = (tile_8_x * dyn_x) ** 0.5  # Geometric mean for log scale
                    mid_y = (tile_8_y_offset2 * dyn_y_offset2) ** 0.5  # Geometric mean for log scale
                    ax.text(mid_x* 0.77, mid_y*1.02, f'{cycle_ratio_8}×',
                           fontsize=x_tick_size, #fontweight='bold',
                           ha='center', va='bottom', color=color_palette[8])

                # Arrow 4: Horizontal arrow from tile=64 to dynamic at tile=64 y-coordinate
                if tile_64_idx is not None:
                    tile_64_x = static_on_chip[tile_64_idx]
                    tile_64_y = static_cycles[tile_64_idx]

                    # Calculate memory ratio
                    memory_ratio_64 = round(tile_64_x / dyn_x, 2)

                    # Add horizontal arrow from (dyn_x, tile_64_y) to (tile_64_x, tile_64_y)
                    ax.annotate('', xy=(tile_64_x, tile_64_y), xytext=(dyn_x, tile_64_y),
                               arrowprops=dict(arrowstyle='<-', lw=4, color=color_palette[12],
                                             mutation_scale=40, shrinkA=10, shrinkB=20,
                                             linestyle=arrow_linestyle))

                    # Add memory ratio text on top of the arrow
                    mid_x = (tile_64_x * dyn_x) ** 0.5  # Geometric mean for log scale
                    ax.text(mid_x, tile_64_y, f'{memory_ratio_64}×',
                           fontsize=x_tick_size, #fontweight='bold',
                           ha='center', va='bottom', color=color_palette[12])

        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Remove minor tick labels (keep tick marks but hide labels)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        # Set specific tick values and format them
        # Use individual tick values if provided, otherwise use default
        if y_ticks_list is not None and plot_idx < len(y_ticks_list):
            y_ticks = y_ticks_list[plot_idx]
        else:
            y_ticks = [1e6, 2e6, 6e6]  # Default values

        # Determine y_scale: use provided scale if available, otherwise auto-detect from ticks
        if y_scale_list is not None and plot_idx < len(y_scale_list):
            y_scale = y_scale_list[plot_idx]
        else:
            # Determine scale factor from first tick value
            y_scale = 10 ** (len(str(int(y_ticks[0]))) - 1)

        if y_ticks is not None:
            ax.set_yticks(y_ticks)
            # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/y_scale)}'))
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, p, scale=y_scale: f'{int(x/scale)}')
            )

        if x_ticks_list is not None and plot_idx < len(x_ticks_list):
            x_ticks = x_ticks_list[plot_idx]
        else:
            x_ticks = [1e7, 2e7, 6e7]  # Default values

        # Determine x_scale: use provided scale if available, otherwise auto-detect from ticks
        if x_scale_list is not None and plot_idx < len(x_scale_list):
            x_scale = x_scale_list[plot_idx]
        else:
            # Determine scale factor from first tick value
            x_scale = 10 ** (len(str(int(x_ticks[0]))) - 1)

        if x_ticks is not None:
            ax.set_xticks(x_ticks)
            # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/x_scale)}'))
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, p, scale=x_scale: f'{int(x/scale)}')
            )
        # Set labels
        ax.set_xlabel(subplot_title, fontsize=axis_font, labelpad=60, fontweight='bold')

        # Only show y-label on leftmost plot
        if plot_idx == 0:
            ax.set_ylabel('Latency (Cycles)', fontsize=axis_font, labelpad=10)

        # Add scale indicator for y-axis
        y_exp = int(np.log10(y_scale))
        # y_scale_text = f'×10{chr(0x2070 + y_exp) if y_exp < 10 else "".join(chr(0x2070 + int(d)) if d != "1" else chr(0x00B9) for d in str(y_exp))}'
        # y_scale_text = f'10{chr(0x2070 + y_exp) if y_exp < 10 else "".join(chr(0x2070 + int(d)) if d != "1" else chr(0x00B9) for d in str(y_exp))}'
        y_scale_text = f'1e{str(y_exp)}'
        
        # Simpler approach: just use superscript digits
        # superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        # y_scale_text = f'×10{str(y_exp).translate(superscripts)}'
        # Position: (x, y) where x: left(-)/right(+), y: down(-)/up(+)
        # Adjust these values to move the y-axis scale indicator
        ax.text(0.15, 1.01, y_scale_text, transform=ax.transAxes,
                fontsize=tick_font, ha='right', va='bottom')

        # Add scale indicator for x-axis on each subplot
        x_exp = int(np.log10(x_scale))
        superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        # x_scale_text = f'×10{str(x_exp).translate(superscripts)}'
        x_scale_text = f'1e{str(x_exp)}'
        # Position: (x, y) where x: left(-)/right(+), y: down(-)/up(+)
        # Adjust these values to move the x-axis scale indicator
        ax.text(1.02, -0.02, x_scale_text, transform=ax.transAxes,
                fontsize=tick_font, ha='center', va='top')

        # Format tick labels with increased padding
        ax.tick_params(axis='both', labelsize=tick_font, pad=15)

        # Adjust offset text for scientific notation
        ax_xoffset_text = ax.xaxis.get_offset_text()
        ax_xoffset_text.set_fontsize(tick_font)
        
        # 1. Use set_y with a small value
        # 2. Change vertical alignment to 'bottom' to push it up from its anchor
        ax_xoffset_text.set_y(0.05) 
        ax_xoffset_text.set_verticalalignment('bottom')
        
        ax_xoffset_text.set_position((1.1, 0.5))

        ax_yoffset_text = ax.yaxis.get_offset_text()
        ax_yoffset_text.set_fontsize(tick_font)
        ax_yoffset_text.set_position((-0.08, 1.02))

        # Set scales if specified
        # Use individual scales if provided, otherwise use global scales
        if cycles_scale_list is not None and plot_idx < len(cycles_scale_list):
            if cycles_scale_list[plot_idx] is not None:
                ax.set_ylim(cycles_scale_list[plot_idx])
        elif cycles_scale is not None:
            ax.set_ylim(cycles_scale)

        if on_chip_scale_list is not None and plot_idx < len(on_chip_scale_list):
            if on_chip_scale_list[plot_idx] is not None:
                ax.set_xlim(on_chip_scale_list[plot_idx])
        elif on_chip_scale is not None:
            ax.set_xlim(on_chip_scale)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=1.5)

        # Set spine properties
        thickness = 2.5
        for spine in ['left', 'right', 'top', 'bottom']:
            ax.spines[spine].set_linewidth(thickness)
            ax.spines[spine].set_color('black')

    # Add legend (only once, positioned at top center)
    if num_plots > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                  bbox_to_anchor=(0.5, 1.32),  # legend position
                  ncol=2, fontsize=legend_font,
                  frameon=False)

    # Add shared x-axis label
    fig.text(0.5, -0.23, 'On-Chip Memory (Bytes)',
             ha='center', va='bottom', fontsize=axis_font)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)

    # Save plot if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Pareto plot saved to {output_file}")

    # Show plot
    plt.show()

    return fig


# Example usage
csv_files = [
    './dyn_tiling/figure_9_mixtral_b64_raw.csv',
    './dyn_tiling/figure_9_qwen_b64_raw.csv',
]

subplot_titles = [
    'Mixtral8x7B',
    'Qwen3-30B-A3B',
]

# Optional: Set axis scales
# Option 1: Use same scale for all plots
cycles_scale = (2e6, 8e6)
cycles_scale = None
on_chip_scale = None

# Option 2: Use different scales for each plot (overrides cycles_scale and on_chip_scale)

# cycles_scale_list = [(2e6, 8.5*1e6), (0.5*1e6, 1.1*1e6)]
# on_chip_scale_list = [(1.5*1e7, 4.5*1e7), (0.5*1e8, 3.5*1e8)]
cycles_scale_list = [(2e6,9e6), (5e5,1.1*1e6)]
on_chip_scale_list = [(1.35*1e7,4e7), (5e7,3.5*1e8)]

# Option 3: Use different tick values for each plot
# Example: First plot shows [1, 2, 6] and second plot shows [1, 3, 5]
# x_ticks_list = [[1e7, 2e7, 6e7], [1e8, 3e8, 5e8]]
# y_ticks_list = [[1e6, 2e6, 6e6], [5e5, 1e6, 2e6]]
y_ticks_list =  [[2e6, 6e6], [6e5, 8e5,1e6]]
x_ticks_list = [[2e7,3e7],[6e7,1e8,2e8]]

# Option 4: Manually specify scale factors for each plot
# Example: First plot uses 1e6 scale, second plot uses 1e5 scale
# x_scale_list = [1e7, 1e8]  # First plot: ×10⁷, Second plot: ×10⁸
# y_scale_list = [1e6, 1e5]  # First plot: ×10⁶, Second plot: ×10⁵
y_scale_list = [1e6, 1e5]
x_scale_list = [1e7, 1e7]

output_file = './dyn_tiling/figure9.pdf'

fig = create_pareto_plot(
    csv_files=csv_files,
    subplot_titles=subplot_titles,
    figsize=(18, 4.5),
    cycles_scale=cycles_scale,
    on_chip_scale=on_chip_scale,
    cycles_scale_list=cycles_scale_list,
    on_chip_scale_list=on_chip_scale_list,
    x_ticks_list=x_ticks_list,
    y_ticks_list=y_ticks_list,
    x_scale_list=x_scale_list,
    y_scale_list=y_scale_list,
    output_file=output_file
)
