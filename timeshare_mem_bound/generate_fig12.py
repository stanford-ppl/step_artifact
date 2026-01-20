import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def cycle_raw_comp_utilization_multi(csv_files, subplot_titles, figsize=(20, 9), 
                                   fig_a_base=None, fig_a_compare_base=None,
                                   fig_b_base=None, fig_b_compare_base=None,
                                   utilization_scale=None, cycles_scale=None,
                                   util_hlines_list=None, cycles_hlines_list=None,
                                   output_file=None):
    """
    Create side-by-side multi-axis line charts from multiple CSV files.
    
    Parameters:
    csv_files (list): List of CSV file paths
    subplot_titles (list): Title for each subplot
    figsize (tuple): Figure size (width, height)
    utilization_scale (tuple, optional): Y-axis scale for utilization
    cycles_scale (tuple, optional): Y-axis scale for cycles
    util_hlines_list (list of lists, optional): Y-values for horizontal lines on utilization axis for each plot
    cycles_hlines_list (list of lists, optional): Y-values for horizontal lines on cycles axis for each plot
    output_file (str, optional): Path to save the combined plot
    """
    font_uniform = 44
    axis_font = font_uniform
    legend_font = font_uniform
    tick_font = font_uniform
    x_tick_size = font_uniform
    subtitle_font = font_uniform
    
    # Set up color palette
    color_palette = sns.color_palette(palette='tab20')
    
    num_plots = len(csv_files)
    
    # Create subplots
    fig, axes = plt.subplots(1, num_plots, figsize=figsize, gridspec_kw={'wspace': 0.1})
    
    # Ensure axes is always a list
    if num_plots == 1:
        axes = [axes]
    
    # Process each subplot
    for plot_idx, (csv_file, subplot_title) in enumerate(zip(csv_files, subplot_titles)):
        ax1 = axes[plot_idx]
        
        # Read CSV file
        df_combined = pd.read_csv(csv_file)
        
        # Create display names from n_par_region column
        
        display_names = ['128\n(1)', '64\n(2)', '32\n(4)', '16\n(8)', '8\n(16)', '4\n(32)']
        # Position of points on X axis
        x_pos = np.arange(len(display_names))
        
        # Remove individual x-axis labels - we'll add a shared one later
        if plot_idx == 0:
            ax1.set_xlabel('(a) Static Tiling', fontsize=axis_font, labelpad=85, fontweight='bold')
        else:
            ax1.set_xlabel('(b) Dynamic Tiling', fontsize=axis_font, labelpad=85, fontweight='bold')
        
        # Create first y-axis for utilization
        performance_cycles = df_combined[df_combined.iloc[:, 0] == 'performance(cycles)'].iloc[0, 1:].astype(int).tolist()
        compute_util = df_combined[df_combined.iloc[:, 0] == 'compute_util(%)'].iloc[0, 1:].astype(float).tolist()
        line1 = ax1.plot(x_pos, compute_util, 
                         color=color_palette[0], marker='o', linewidth=3, 
                         markersize=8, label=f'Compute Util.' if plot_idx == 0 else "")
        
        # Customize first y-axis (for utilization)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(display_names, ha='center', fontsize=x_tick_size)
        ax1.tick_params(axis='y', labelcolor=color_palette[0], labelsize=tick_font)

        
        
        # Only show y-label on leftmost plot
        if plot_idx == 0:
            ax1.set_ylabel('Utilization (%)', fontsize=axis_font, labelpad=10)
        
        # Create second y-axis for cycles
        ax2 = ax1.twinx()
        line2 = ax2.plot(x_pos, performance_cycles, 
                         color=color_palette[2], marker='s', linewidth=3, 
                         markersize=8, label='Cycles' if plot_idx == 0 else "")
        
        # Customize second y-axis (for cycles)
        ax2.tick_params(axis='y', labelcolor=color_palette[2], labelsize=tick_font)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax2_offset_text = ax2.yaxis.get_offset_text()
        ax2_offset_text.set_fontsize(tick_font)
        ax2_offset_text.set_position((0.99, 1.02))
        
        # Only show y-label on rightmost plot
        if plot_idx == num_plots - 1:
            ax2.set_ylabel('Performance (cycles)', fontsize=axis_font, labelpad=10)
        
        # Set subplot title
        # ax1.set_title(subplot_title, fontsize=subtitle_font, fontweight='bold')
        
        # ==================== Set Range ====================
        if utilization_scale is not None:
            ax1.set_ylim(utilization_scale)
        if cycles_scale is not None:
            ax2.set_ylim(cycles_scale)
        
        # ==================== Add Red Horizontal Lines ====================
        # Add horizontal lines for utilization axis
        if util_hlines_list and plot_idx < len(util_hlines_list) and util_hlines_list[plot_idx]:
            for i, y_val in enumerate(util_hlines_list[plot_idx]):
                ax1.axhline(y=y_val, color='red', linestyle='--', linewidth=2, alpha=0.8)
                ax1.text(x_pos[-1], y_val + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02, 
                        f'Target: {y_val}%', 
                        color='red', fontsize=20, ha='right', va='bottom')
        
        # Add horizontal lines for cycles axis
        if cycles_hlines_list and plot_idx < len(cycles_hlines_list) and cycles_hlines_list[plot_idx]:
            for i, y_val in enumerate(cycles_hlines_list[plot_idx]):
                ax2.axhline(y=y_val, color='red', linestyle='--', linewidth=2, alpha=0.8)
                if y_val == fig_a_compare_base:
                    percentile = 1
                    ax2.text(x_pos[0], y_val + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.02, 
                        f'   < {percentile}%', 
                        color='red', fontsize=x_tick_size, ha='left', va='bottom')

                elif y_val == fig_b_compare_base:
                    percentile = 5
                    ax2.text(x_pos[0], y_val + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.02, 
                        f'   ~ {percentile}%', 
                        color='red', fontsize=x_tick_size, ha='left', va='bottom')

                
                    
        
        # Add grid - Only horizontal grid lines
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Remove y-axis ticks and labels from non-edge plots for cleaner look
        if plot_idx > 0:
            ax1.tick_params(axis='y', left=False, labelleft=False)
        if plot_idx < num_plots - 1:
            ax2.tick_params(axis='y', right=False, labelright=False)
    
    # Add legend (only once, positioned at top)
    if num_plots > 0:
        # Get legend elements from the first plot
        lines1, labels1 = axes[0].get_legend_handles_labels()
        ax2_first = axes[0].get_shared_x_axes().get_siblings(axes[0])[0] if hasattr(axes[0], 'get_shared_x_axes') else None
        
        # Try to get the twinx axis from the first subplot
        lines2, labels2 = [], []
        for child in axes[0].get_figure().get_children():
            if hasattr(child, 'get_legend_handles_labels'):
                try:
                    temp_lines, temp_labels = child.get_legend_handles_labels()
                    if temp_labels and 'Cycles' in temp_labels:
                        lines2, labels2 = temp_lines, temp_labels
                        break
                except:
                    continue
        
        # If we couldn't find the cycles legend, create it manually
        if not lines2:
            # Create a dummy line for the legend
            dummy_line = plt.Line2D([0], [0], color=color_palette[2], marker='s', linewidth=3, markersize=8)
            lines2 = [dummy_line]
            labels2 = ['Cycles']
        
        all_handles = lines1 + lines2
        all_labels = labels1 + labels2
        fig.legend(all_handles, all_labels, loc='upper right', bbox_to_anchor=(0.91, 0.90), 
                  ncol=1, fontsize=legend_font)
    
    # Add shared x-axis label
    fig.text(0.5, -0.12, 'Parallel Regions (Experts per Region)', 
             ha='center', va='bottom', fontsize=axis_font)
    
    # Adjust layout
    plt.tight_layout()
    # Add some bottom padding for the shared x-label
    plt.subplots_adjust(bottom=0.15)
    
    # Save plot if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {output_file}")
    
    # Show plot
    # plt.show()
    
    return fig

# Your existing parameters
csv_file_1='./timeshare_mem_bound/fig_8_a.csv'
csv_file_2='./timeshare_mem_bound/fig_8_b.csv'

utilization_scale = (0,32)
cycles_scale = (0.6*1e6,2.1*1e6)


# Option 2: Use the new side-by-side function
csv_files = [csv_file_1, csv_file_2]
subplot_titles = ['N=32', 'N=Dynamic']

df = pd.read_csv(csv_file_1, index_col=0)
fig_a_base = df.loc['performance(cycles)', '128(1)']
fig_a_compare_base = df.loc['performance(cycles)', '32(4)']

df = pd.read_csv(csv_file_2, index_col=0)
fig_b_base = df.loc['performance(cycles)', '128(1)']
fig_b_compare_base = df.loc['performance(cycles)', '32(4)']

cycles_hlines_list = [[fig_a_base, fig_a_compare_base], [fig_b_base, fig_b_compare_base]]
output_file_combined = "./timeshare_mem_bound/figure12.pdf"

cycle_raw_comp_utilization_multi(
    csv_files=csv_files,
    subplot_titles=subplot_titles,
    figsize=(18, 8.5),
    fig_a_base=fig_a_base, fig_a_compare_base=fig_a_compare_base,
    fig_b_base=fig_b_base, fig_b_compare_base=fig_b_compare_base,
    utilization_scale=utilization_scale,
    cycles_scale=cycles_scale,
    cycles_hlines_list=cycles_hlines_list,
    output_file=output_file_combined
)