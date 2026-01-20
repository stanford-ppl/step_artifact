import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors

def create_pareto_plot(csv_files, subplot_titles, figsize=(20, 9),
                       cycles_scale=None, on_chip_scale=None,
                       cycles_scale_list=None, on_chip_scale_list=None,
                       output_file=None):
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
    output_file (str, optional): Path to save the combined plot
    """
    font_uniform = 40
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
        static_on_chip = df_static['on_chip_mem'].values if plot_idx==0 else df_static['on_chip_mem_new'].values

        # Extract data for dynamic tiling
        dynamic_cycles = df_dynamic['cycles'].values[0] if len(df_dynamic) > 0 else None
        dynamic_on_chip = df_dynamic['on_chip_mem'].values[0] if len(df_dynamic) > 0 else None
        if plot_idx ==1:
            dynamic_on_chip = df_dynamic['on_chip_mem_new'].values[0] if len(df_dynamic) > 0 else None
            
        # Plot the Pareto curve (connecting static tile points)
        ax.plot(static_on_chip, static_cycles,
                color=color_palette[0], linewidth=3,
                marker='o', markersize=12,
                label='Static Tiling' if plot_idx == 0 else "",
                zorder=2)
        
        if plot_idx == 0:
            # Annotate each point with tile size
            for i, (x, y, tile) in enumerate(zip(static_on_chip, static_cycles, tile_sizes)):
                if i ==0:
                    ax.annotate(f'{tile}',
                        xy=(x, y),
                        xytext=(-60, -30),
                        #    rotation=30,
                        textcoords='offset points',
                        fontsize=x_tick_size)
                elif i ==2:
                    ax.annotate(f'{tile}',
                        xy=(x, y),
                        xytext=(5, 15),
                        #    rotation=30,
                        textcoords='offset points',
                        fontsize=x_tick_size)
                elif i==6:
                    ax.annotate(f'{tile}',
                        xy=(x, y),
                        xytext=(-80, 20),
                        #    rotation=30,
                        textcoords='offset points',
                        fontsize=x_tick_size)
                elif i in [4]:
                    ax.annotate(f'{tile}',
                        xy=(x, y),
                        xytext=(-20, 15),
                        #    rotation=30,
                        textcoords='offset points',
                        fontsize=x_tick_size)
        else:
            # Annotate each point with tile size
            for i, (x, y, tile) in enumerate(zip(static_on_chip, static_cycles, tile_sizes)):
                if i in [4]:
                    ax.annotate(f'{tile}',
                        xy=(x, y),
                        xytext=(-25, 20),
                        #    rotation=30,
                        textcoords='offset points',
                        fontsize=x_tick_size)
                elif i==6:
                    ax.annotate(f'{tile}',
                        xy=(x, y),
                        xytext=(-55, 20),
                        #    rotation=30,
                        textcoords='offset points',
                        fontsize=x_tick_size)
                elif i in [0,2]:
                    ax.annotate(f'{tile}',
                        xy=(x, y),
                        xytext=(10, -10),
                        #    rotation=30,
                        textcoords='offset points',
                        fontsize=x_tick_size)

        # Plot dynamic tiling point
        if dynamic_cycles is not None and dynamic_on_chip is not None:
            ax.scatter(dynamic_on_chip, dynamic_cycles,
                      color=color_palette[4], s=300,
                      marker='o', linewidths=2,
                      label='Dynamic Tiling' if plot_idx == 0 else "",
                      zorder=3)

            # Annotate dynamic point

            # ax.annotate('Dyn.',
            #            xy=(dynamic_on_chip, dynamic_cycles),
            #            xytext=(-105, -25),
            #            textcoords='offset points',
            #            fontsize=x_tick_size,
            #            fontweight='bold')

            # Add arrows between tile=128 and dynamic in first plot (Mixtral)
            if plot_idx == 0:
                tile_256_idx = np.where(tile_sizes == 256)[0]
                tile_512_idx = np.where(tile_sizes == 512)[0]
                if len(tile_256_idx) > 0:
                    idx = tile_256_idx[0]
                    tile_256_x = static_on_chip[idx]
                    tile_256_y = static_cycles[idx]

                    tile_512_x = static_on_chip[tile_512_idx[0]]

                    # Draw arrow from dynamic point to tile=256
                    ax.annotate('',
                               xy=(tile_256_x, tile_256_y),  # Arrow points to tile=256
                               xytext=(dynamic_on_chip, dynamic_cycles),  # Arrow starts from dynamic
                               arrowprops=dict(arrowstyle='<-,head_width=0.8,head_length=0.8',
                                             color=color_palette[8],
                                             lw=4,
                                             mutation_scale=15,
                                             linestyle='dotted',
                                             shrinkA=10,  # Shrink from start point
                                             shrinkB=10))  # Shrink from end point

                    # Calculate and display the ratio
                    ratio = round(tile_256_y / dynamic_cycles, 2)
                    # Position text at midpoint of arrow
                    mid_x = tile_512_x
                    mid_y = (tile_256_y + dynamic_cycles) / 2 - 0.2*1e7
                    ax.text(mid_x*1.1, mid_y, f'{ratio}x',
                           fontsize=x_tick_size,
                           color=color_palette[8],
                           fontweight='bold',
                           ha='center',
                           va='bottom')
                    
                # Find tile=128 point
                tile_128_idx = None
                for i, tile in enumerate(tile_sizes):
                    if tile == 128:
                        tile_128_idx = i

                dyn_x = dynamic_on_chip
                dyn_y = dynamic_cycles

                # Arrow 1: Memory usage ratio (dynamic vs tile=128)
                if tile_128_idx is not None:
                    tile_128_x = static_on_chip[tile_128_idx]
                    tile_128_y = static_cycles[tile_128_idx]

                    # Calculate memory ratio
                    memory_ratio = round(tile_128_x / dyn_x, 2)

                    # Offset the arrow slightly higher in y to separate from second arrow
                    dyn_y_offset1 = dyn_y * 1.02
                    tile_128_y_offset1 = tile_128_y * 1.02

                    # Add double-headed arrow with large arrowheads
                    # ax.annotate('', xy=(tile_128_x, tile_128_y_offset1), xytext=(dyn_x, dyn_y_offset1),
                    #            arrowprops=dict(arrowstyle='<-', lw=4, color=color_palette[8],
                    #                          mutation_scale=40, shrinkA=10, shrinkB=20,
                    #                          linestyle='dotted'))

                    # # Add memory ratio text
                    # mid_x = (tile_128_x * dyn_x) ** 0.5  # Geometric mean for log scale
                    # mid_y = (tile_128_y_offset1 * dyn_y_offset1) ** 0.5  # Geometric mean for log scale
                    # ax.text(mid_x * 0.4, mid_y * 1.35, f'{memory_ratio}×',
                    #        fontsize=x_tick_size,
                    #        ha='center', va='top', color=color_palette[8])

                # Arrow 2: Performance speedup (dynamic vs tile=128)
                if tile_128_idx is not None:
                    tile_128_x = static_on_chip[tile_128_idx]
                    tile_128_y = static_cycles[tile_128_idx]

                    # Calculate cycle ratio
                    cycle_ratio = round(tile_128_y / dyn_y, 2)

                    # Offset the arrow slightly lower in y to separate from first arrow
                    dyn_y_offset2 = dyn_y * 0.98
                    tile_128_y_offset2 = tile_128_y * 0.98

                    # Add double-headed arrow with large arrowheads
                    ax.annotate('', xy=(tile_128_x, tile_128_y_offset2), xytext=(dyn_x, dyn_y_offset2),
                               arrowprops=dict(arrowstyle='<-', lw=4, color=color_palette[8],
                                             mutation_scale=40, shrinkA=10, shrinkB=10,
                                             linestyle='dotted'))

                    # Add cycle ratio text
                    mid_x = (tile_128_x * dyn_x) ** 0.5  # Geometric mean for log scale
                    mid_y = (tile_128_y_offset2 * dyn_y_offset2) ** 0.5  # Geometric mean for log scale
                    ax.text(mid_x * 0.5, mid_y * 0.8, f'{cycle_ratio}×',
                           fontsize=x_tick_size,
                           ha='center', va='bottom', color=color_palette[8], fontweight='bold')

            # Add vertical and horizontal arrows in second plot
            if plot_idx == 1:
                # Find tile=64 and tile=1024 points
                tile_16_idx = np.where(tile_sizes == 16)[0]
                tile_64_idx = np.where(tile_sizes == 64)[0]
                tile_256_idx = np.where(tile_sizes == 256)[0]
                tile_1024_idx = np.where(tile_sizes == 1024)[0]
                if len(tile_64_idx) > 0:
                    idx_64 = tile_64_idx[0]
                    tile_16_x = static_on_chip[tile_16_idx[0]]
                    tile_64_x = static_on_chip[idx_64]
                    tile_64_y = static_cycles[idx_64]

                    # Draw vertical arrow from (dyn_x, dyn_y) to (dyn_x, tile_64_y)
                    ax.annotate('',
                               xy=(dynamic_on_chip, tile_64_y),  # Arrow points to tile_64_y
                               xytext=(dynamic_on_chip, dynamic_cycles),  # Arrow starts from dynamic
                               arrowprops=dict(arrowstyle='<-,head_width=0.8,head_length=0.8',
                                             color=color_palette[8],
                                             lw=4,
                                             linestyle='dotted',
                                             shrinkA=10,  # Shrink from start point
                                             shrinkB=0))  # Shrink from end point

                    # Draw horizontal line from (dyn_x, tile_64_y) to (tile_64_x, tile_64_y)
                    ax.plot([dynamic_on_chip-0.3*1e8, tile_64_x], [tile_64_y, tile_64_y],
                           color=color_palette[8], linewidth=3, linestyle='-', zorder=2)

                    # Calculate and display the vertical ratio
                    ratio_y = round(tile_64_y / dynamic_cycles, 2)
                    # Position text to the right of the arrow
                    text_x = tile_16_x -0.5 * 1e8 # Slightly to the right
                    text_y = (tile_64_y + dynamic_cycles) / 2
                    ax.text(text_x*0.8, text_y, f'{ratio_y}x',
                           fontsize=x_tick_size,
                           color=color_palette[8],
                           fontweight='bold',
                           ha='left',
                           va='center')

                if len(tile_1024_idx) > 0:
                    idx_1024 = tile_1024_idx[0]
                    tile_256_x = static_on_chip[tile_256_idx[0]]
                    tile_1024_x = static_on_chip[idx_1024]
                    tile_1024_y = static_cycles[idx_1024]

                    # Draw horizontal arrow from (dyn_x, dyn_y) to (tile_1024_x, dyn_y)
                    ax.annotate('',
                               xy=(tile_1024_x, dynamic_cycles),  # Arrow points to tile_1024_x
                               xytext=(dynamic_on_chip, dynamic_cycles),  # Arrow starts from dynamic
                               arrowprops=dict(arrowstyle='<-,head_width=0.8,head_length=0.8',
                                             color=color_palette[12],
                                             lw=4,
                                             linestyle='dotted',
                                             shrinkA=10,  # Shrink from start point
                                             shrinkB=0))  # Shrink from end point

                    # Draw vertical line from (tile_1024_x, tile_1024_y) to (tile_1024_x, dyn_y)
                    ax.plot([tile_1024_x, tile_1024_x], [tile_1024_y, dynamic_cycles-0.1*1e6],
                           color=color_palette[12], linewidth=4, linestyle='-', zorder=2)

                    # Calculate and display the horizontal ratio
                    ratio_x = round(tile_1024_x / dynamic_on_chip, 1)
                    # Position text above the arrow
                    text_x = tile_256_x
                    text_y = dynamic_cycles - 2.75*1e5  # Slightly below
                    ax.text(text_x, text_y, f'{ratio_x}x',
                           fontsize=x_tick_size,
                           color=color_palette[12],
                           fontweight='bold',
                           ha='center',
                           va='bottom')

        # Set labels
        ax.set_xlabel(subplot_title, fontsize=axis_font, labelpad=60, fontweight='bold')

        # Only show y-label on leftmost plot
        if plot_idx == 0:
            ax.set_ylabel('Latency (Cycles)', fontsize=axis_font, labelpad=10)

        # Format tick labels with increased padding
        # Make tick marks larger: major ticks (with labels) and minor ticks (without labels)
        ax.tick_params(axis='both', labelsize=tick_font, pad=15,
                      which='major', length=10, width=2.5)
        ax.tick_params(axis='both', which='minor', length=6, width=2)

        # Use LogLocator to show only powers of 10 for cleaner labels
        from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter

        # Set major locators to only show powers of 10
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))

        # Use LogFormatterSciNotation for major ticks
        ax.xaxis.set_major_formatter(LogFormatterSciNotation())
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())

        # Remove minor tick labels
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

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

        # Add grid with larger linewidth
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=2.5, which='major')

        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Kill minor y-ticks on second plot
        if plot_idx == 1:
            from matplotlib.ticker import NullLocator
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax.yaxis.set_minor_formatter(NullFormatter())
        
        # Set spine properties
        thickness = 2.5
        for spine in ['left', 'right', 'top', 'bottom']:
            ax.spines[spine].set_linewidth(thickness)
            ax.spines[spine].set_color('black')

    # Add legend (only once, positioned at top center)
    if num_plots > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                  bbox_to_anchor=(0.5, 1.2),  # legend position
                  ncol=2, fontsize=legend_font,
                  frameon=False)

    # Add shared x-axis label
    fig.text(0.5, -0.22, 'On-Chip Memory (Bytes)',
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
    './dyn_tiling/figure_10_mixtral_b1024_raw.csv',
    './dyn_tiling/figure_10_qwen_b1024_raw.csv'
]

subplot_titles = [
    # 'Qwen3-30B-A3B (2_2)',
    'Mixtral8x7B',
    'Qwen3-30B-A3B',
]

# Optional: Set axis scales
# Option 1: Use same scale for all plots
# cycles_scale = (2e6, 8e6)
cycles_scale = None
on_chip_scale = None

# Option 2: Use different scales for each plot (overrides cycles_scale and on_chip_scale)
# cycles_scale_list = [(2e6, 8e6), (2e6, 8e6)]  # [plot1_ylim, plot2_ylim]
# on_chip_scale_list = [(1.5e7, 4e7), (1.5e7, 4e7)]  # [plot1_xlim, plot2_xlim]

cycles_scale_list = [(6*1e6,7*1e7), (7*1e5, 7*1e6)]
on_chip_scale_list = [(1e7, 5*1e8), (1e8, 5*1e9)]
# cycles_scale_list = [(2e6, 8.5*1e6), (0.5*1e6, 1.05*1e6)]
# on_chip_scale_list = [(1.5*1e7, 4.5*1e7), (0.5*1e8, 1.8*1e8)]

output_file = './dyn_tiling/figure10.pdf'

fig = create_pareto_plot(
    csv_files=csv_files,
    subplot_titles=subplot_titles,
    figsize=(18, 4.5),
    cycles_scale=cycles_scale,
    on_chip_scale=on_chip_scale,
    cycles_scale_list=cycles_scale_list,
    on_chip_scale_list=on_chip_scale_list,
    output_file=output_file
)
