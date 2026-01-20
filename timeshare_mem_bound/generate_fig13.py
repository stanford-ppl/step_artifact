import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def create_two_plots_pdf(csv_files, output_pdf, cycles_scale=None, on_chip_scale=None, 
                         allocated_flops_scale=None, bandwidth_scale_2=None, cycles_scale_2=None):
    """
    Create a single PDF with two plots arranged vertically [2,1] with equal box sizes.
    
    Parameters:
    csv_file (str): Path to CSV file
    output_pdf (str): Path to save the PDF
    cycles_scale (tuple, optional): Y-axis scale for cycles (plot 1)
    on_chip_scale (tuple, optional): Y-axis scale for on-chip requirement
    allocated_flops_scale (tuple, optional): Y-axis scale for allocated FLOPS
    bandwidth_scale_2 (tuple, optional): Y-axis scale for bandwidth utilization (plot 2)
    cycles_scale_2 (tuple, optional): Y-axis scale for cycles (plot 2)
    """

    cycles_color_idx = 2
    on_chip_color_idx = 4
    allocated_flops_color_idx = 8
    bandwidth_color_idx = 18
    
    font_uniform = 44
    axis_font = font_uniform
    legend_font = font_uniform
    tick_font = font_uniform
    x_tick_size = font_uniform
    
    # Set up color palette
    color_palette = sns.color_palette(palette='tab20')
    # print(color_palette)
    
    # Read CSV file
    df_combined_a = pd.read_csv(csv_files[0])
    df_combined_b = pd.read_csv(csv_files[1])
    
    # Create display names from n_par_region column
    display_names = ['128\n(1)', '64\n(2)', '32\n(4)', '16\n(8)', '8\n(16)', '4\n(32)']
    
    # Position of points on X axis
    x_pos = np.arange(len(display_names))
    
    # Create figure with subplots - adjusted for vertical [2,1] layout
    # Increased hspace to accommodate shared legend
    fig = plt.figure(figsize=(9, 11))
    
    # Use gridspec for better control over subplot positioning
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 1, figure=fig, left=0.1, right=0.85, bottom=0.1, top=0.9, hspace=0.25)
    
    # ==================== FIRST SUBPLOT (TOP): Three-axis plot ====================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create first y-axis for cycles
    performance_cycles = df_combined_a[df_combined_a.iloc[:, 0] == 'performance(cycles)'].iloc[0, 1:].astype(int).tolist()

    line1 = ax1.plot(x_pos, performance_cycles, 
                     color=color_palette[cycles_color_idx], marker='o', linewidth=3, 
                     markersize=8, label=f'Cycles')
    
    # Customize first y-axis - NO X-AXIS LABELS for top plot
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([])  # Remove x-tick labels
    ax1.tick_params(axis='y', labelcolor=color_palette[cycles_color_idx], labelsize=tick_font)
    ax1.tick_params(axis='x', bottom=False)  # Remove x-tick marks
    
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax1_offset_text = ax1.yaxis.get_offset_text()
    ax1_offset_text.set_fontsize(tick_font)
    ax1_offset_text.set_position((-0.15, 1.02))
    ax1.set_ylabel('Performance\n(Cycles)', fontsize=axis_font, labelpad=10)
    
    # Create second y-axis for on_chip_requirement_bytes
    ax1_2 = ax1.twinx()
    on_chip_req_kb = df_combined_a[df_combined_a.iloc[:, 0] == 'on_chip_mem(KB)'].iloc[0, 1:].astype(float).tolist()
    line2 = ax1_2.plot(x_pos, on_chip_req_kb, 
                       color=color_palette[on_chip_color_idx], marker='s', linewidth=3, 
                       markersize=8, label=f'On-Chip\nMemory')
    
    # Customize second y-axis
    ax1_2.tick_params(axis='y', labelcolor=color_palette[on_chip_color_idx], labelsize=tick_font)
    ax1_2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax1_2_offset_text = ax1_2.yaxis.get_offset_text()
    ax1_2_offset_text.set_fontsize(tick_font)
    ax1_2_offset_text.set_position((0.8, 1.02))
    
    ax1_2.set_ylabel('Memory (KB)', fontsize=axis_font, labelpad=5)
    
    # Create third y-axis for allocated_flops
    ax1_3 = ax1.twinx()
    ax1_3.spines['right'].set_position(('outward', 120))  # Increased from 95 to 120

    allocated_flops = df_combined_a[df_combined_a.iloc[:, 0] == 'allocated_comp(flops/cycle)'].iloc[0, 1:].astype(float).tolist()
    
    line3 = ax1_3.plot(x_pos, allocated_flops, 
                       color=color_palette[allocated_flops_color_idx], marker='^', linewidth=3, 
                       markersize=8, label=f'Allocated\nCompute')
    
    # Customize third y-axis
    ax1_3.tick_params(axis='y', labelcolor=color_palette[allocated_flops_color_idx], labelsize=tick_font)
    ax1_3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax1_3_offset_text = ax1_3.yaxis.get_offset_text()
    ax1_3_offset_text.set_fontsize(tick_font)
    ax1_3_offset_text.set_position((1, 1.02))
    ax1_3.set_ylabel('Compute\n(FLOPs/cycle)', fontsize=axis_font, labelpad=10)
    
    # Set scales for first plot
    if cycles_scale is not None:
        ax1.set_ylim(cycles_scale)
    if on_chip_scale is not None:
        ax1_2.set_ylim(on_chip_scale)
    if allocated_flops_scale is not None:
        ax1_3.set_ylim(allocated_flops_scale)
    
    # NO individual legend for first plot - will be handled by shared legend
    
    ax1.grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines
    ax1.text(0.02, 0.95, '(a)', fontsize=x_tick_size, fontweight='bold', 
             transform=ax1.transAxes, va='top')
    
    # ==================== SECOND SUBPLOT (BOTTOM): Off-chip bandwidth plot ====================
    ax2 = fig.add_subplot(gs[1, 0])
    
    mem_bw = 1024  # (bytes/cycle)
    
    # Create first y-axis for cycles (now on the LEFT)
    performance_cycles = df_combined_b[df_combined_b.iloc[:, 0] == 'performance(cycles)'].iloc[0, 1:].astype(int).tolist()
    line5 = ax2.plot(x_pos, performance_cycles, 
                     color=color_palette[cycles_color_idx], marker='s', linewidth=3, 
                     markersize=8, label='Cycles')
    
    # Customize first y-axis - WITH X-AXIS LABELS for bottom plot
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(display_names, ha='center', fontsize=x_tick_size)
    ax2.tick_params(axis='y', labelcolor=color_palette[cycles_color_idx], labelsize=tick_font)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax2_offset_text = ax2.yaxis.get_offset_text()
    ax2_offset_text.set_fontsize(tick_font)
    ax2_offset_text.set_position((-0.1, 1.02))
    ax2.set_ylabel('Performance\n(Cycles)', fontsize=axis_font, labelpad=10)
    
    # Create second y-axis for bandwidth utilization (now on the RIGHT)
    ax2_2 = ax2.twinx()
    off_chip_bandwidth_utilization = df_combined_b[df_combined_b.iloc[:, 0] == 'off_chip_bandwidth_utilization(%)'].iloc[0, 1:].astype(float).tolist()

    line4 = ax2_2.plot(x_pos, off_chip_bandwidth_utilization, 
                       color=color_palette[bandwidth_color_idx], marker='o', linewidth=3, 
                       markersize=8, label=f'Off-chip\nBW Util.')
    
    # Customize second y-axis
    ax2_2.tick_params(axis='y', labelcolor=color_palette[bandwidth_color_idx], labelsize=tick_font)
    ax2_2.set_ylabel('Utilization (%)', fontsize=axis_font, labelpad=0)
    
    # Set scales for second plot
    if cycles_scale_2 is not None:
        ax2.set_ylim(cycles_scale_2)  # Now ax2 is cycles
    if bandwidth_scale_2 is not None:
        ax2_2.set_ylim(bandwidth_scale_2)  # Now ax2_2 is bandwidth
    
    # NO individual legend for second plot - will be handled by shared legend
    
    ax2.grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines
    ax2.text(0.02, 0.95, '(b)', fontsize=x_tick_size, fontweight='bold', 
             transform=ax2.transAxes, va='top')
    
    # ==================== SHARED LEGEND FOR ALL FOUR METRICS ====================
    # Collect all handles and labels from all axes
    lines1, labels1 = ax1.get_legend_handles_labels()      # Cycles from top plot
    lines2, labels2 = ax1_2.get_legend_handles_labels()    # On-Chip Memory
    lines3, labels3 = ax1_3.get_legend_handles_labels()    # Allocated FLOPS
    lines4, labels4 = ax2_2.get_legend_handles_labels()    # BW Utilization from bottom plot
    
    # Combine all handles and labels
    all_handles = lines1 + lines2 + lines3 + lines4
    all_labels = labels1 + labels2 + labels3 + labels4
    
    # Create single shared legend at the top of the figure
    fig.legend(all_handles, all_labels, 
              loc='upper center', 
              bbox_to_anchor=(1.38, 0.55),  # Top of figure
              ncol=1,  # One column - all four metrics in a vertical stack [4,1]
              fontsize=legend_font,
              frameon=True,
              fancybox=True,
              shadow=True)
    
    # Add shared x-axis label at the bottom
    fig.text(0.5, -0.11, 'Parallel Regions (Experts per Region)', 
             fontsize=x_tick_size+4, ha='center', va='bottom')
    
    # Save to PDF
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        print(f"Two plots saved to {output_pdf}")
    
    # Show the plot
    plt.show()
    plt.close()

# Example usage:
# create_two_plots_pdf('your_data.csv', 'two_plots.pdf')

outfile = "./timeshare_mem_bound/figure13.pdf"

on_chip_scale=(1*1e4,10*1e4)
allocated_flops_scale=(1*1e5,10*1e5)
cycles_scale=(0.5*1e6,2.5*1e6)
bandwidth_scale_2=(0,100)

# on_chip_scale=None
# allocated_flops_scale=None
# cycles_scale=None
# bandwidth_scale_2=None


create_two_plots_pdf(
    ['./timeshare_mem_bound/fig_9_a.csv', './timeshare_mem_bound/fig_9_b.csv'],
    outfile,
    cycles_scale=cycles_scale, 
    on_chip_scale=on_chip_scale, 
    allocated_flops_scale=allocated_flops_scale, 
    bandwidth_scale_2=bandwidth_scale_2, 
    cycles_scale_2=cycles_scale)