import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

tab10_colors = sns.color_palette('tab10')
colors = [mcolors.to_hex(tab10_colors[0]),mcolors.to_hex(tab10_colors[1]),mcolors.to_hex(tab10_colors[2])]
edgecolor = '#192133'

def transposed_create_multi_csv_bar_chart(csv_files, x_labels_list, subplot_titles, 
                              figsize=(16, 6),
                              ylim=(0, 4), ylabel="Speedup"):
    """
    Creates multiple grouped bar charts from different CSV files with shared y-axis.
    
    Parameters:
    csv_files (list): List of CSV file paths, one for each subplot
    x_labels_list (list of lists): X-axis labels for each subplot
    subplot_titles (list): Title for each subplot  
    figsize (tuple): Figure size (width, height)
    title (str): Overall figure title
    ylim (tuple): Y-axis limits
    ylabel (str): Y-axis label
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """

    legend_font = 32
    xlabel_font = 32
    ylabel_font = 30
    subtitle_font = 32
    
    num_plots = len(csv_files)
    
    # Calculate width ratios based on number of x-labels in each plot
    width_ratios = [len(labels) for labels in x_labels_list]
    
    # Create subplots with shared y-axis and better spacing
    fig, axes = plt.subplots(1, num_plots, figsize=figsize, sharey=True, 
                           gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.0})
    
    # Ensure axes is always a list
    if num_plots == 1:
        axes = [axes]
    thickness = 2.5
    for ax in axes:
        for p in ['left', 'right', 'top', 'bottom']:
            ax.spines[p].set_linewidth(thickness)
            ax.spines[p].set_color('black')
    
    # Process each subplot with its own CSV file
    for i, (ax, csv_file, x_labels, subplot_title) in enumerate(zip(axes, csv_files, x_labels_list, subplot_titles)):
        
        
        # Read CSV data for this specific plot
        df = pd.read_csv(csv_file)
        
        # Extract and normalize data (using last row as baseline)
        norm_cycles = df['cycles'].values[-1]
        norm_off_chip_traffic = df['off_chip_traffic'].values[-1]
        norm_on_chip_requirement = df['on_chip_mem'].values[-1]

        cycles = df['cycles'].values / norm_cycles
        off_chip_traffic = df['off_chip_traffic'].values / norm_off_chip_traffic
        on_chip_requirement = df['on_chip_mem'].values / norm_on_chip_requirement
        print(cycles,off_chip_traffic,on_chip_requirement)
        
        # Set up bar positions - now for 3 metric types
        metrics = ['Cycles', 'Off-chip\nTraffic', 'On-chip\nMem.']
        x = np.arange(len(metrics))
        width = 0.25
        
        # Create bars grouped by tile size (now these become the legend)
        tile_labels = ['Tile=16', 'Tile=64', 'Dynamic']
        if "b_1024" in csv_files[0]:
            tile_labels = ['Tile=256', 'Tile=1024', 'Dynamic']
        bars1 = ax.bar(x - width, [cycles[0], off_chip_traffic[0], on_chip_requirement[0]], width, 
                      label=tile_labels[0] if i == 0 else "", 
                      alpha=0.8, 
                      zorder=2,linewidth=3.0,edgecolor=edgecolor,
                      color=colors[0])
        bars2 = ax.bar(x, [cycles[1], off_chip_traffic[1], on_chip_requirement[1]], width, 
                      label=tile_labels[1] if i == 0 else "", 
                      alpha=0.8, 
                    
                      zorder=2,linewidth=3.0,edgecolor=edgecolor,
                      color=colors[1])
        bars3 = ax.bar(x + width, [cycles[2], off_chip_traffic[2], on_chip_requirement[2]], width, 
                      label=tile_labels[2] if i == 0 else "", 
                      alpha=0.8, 
                      zorder=2,linewidth=3.0,edgecolor=edgecolor,
                      color=colors[2])
        
        bars_group = [bars1, bars2, bars3]

        for bars in bars_group:
            for rect in bars:
                val = rect.get_height()
                if val > 2.5:
                    x_pos = rect.get_x() + rect.get_width()/2
                    ax.text(x_pos + 0.7*width, 2.125, str(round(val, 1)),
                            fontsize=xlabel_font*0.8, color="black")
                    ax.annotate("",
                        xy=(x_pos + 0.6*width, 2.5),
                        xytext=(x_pos + 1.0*width, 2.3),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2))


        if on_chip_requirement[1] > 2.5 and False:
            ax.text(x[2]+0.7*width, 2.2, str(round(on_chip_requirement[1],1)), fontsize=xlabel_font*0.8, color="black")
            ax.annotate("",
                xy=(x[2]+0.6*width, 2.5),        # arrow tip (where it points to)
                xytext=(x[2]+0.7*width, 2.3),  # arrow tail (where it starts)
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
        
        # ax.set_ylim(0.0, 2.5)
        # ax.set_yscale('log')


        # print(x,on_chip_requirement)
        
        # Customize subplot
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, ha='center')
        # ax.set_title(subplot_title, fontsize=subtitle_font, fontweight='bold')
        # ax.grid(True, alpha=0.3, axis='y')
        # ax.grid(True, alpha=0.3, linestyle='-', dashes=(6,6), linewidth=1.5, axis='y')
        # ax.grid(True, alpha=0.3, linestyle='-', linewidth=1.5, axis='y')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=1.5, axis='y')


        
        # ax.set_xlabel('Metric Type', fontsize=xlabel_font)
        ax.set_xlabel(subplot_title, fontsize=xlabel_font, fontweight='bold')
        
        
        # Only show y-label on leftmost plot
        if i == 0:
            print(i,ylabel)
            ax.set_ylabel(ylabel, fontsize=xlabel_font)
            
        # # Remove y-axis ticks and labels from non-leftmost plots for cleaner look
        # if i > 0:
        #     ax.tick_params(axis='y', left=False, labelleft=False)
        
        ax.tick_params(axis='x', labelsize=xlabel_font)
        # thickness = 3.0
        # for p in ['left', 'right', 'top', 'bottom']:
        #     ax.spines[p].set_linewidth(thickness)
        #     ax.spines[p].set_color('black')
        # ax.set_ylim(0, 2.5)
    axes[0].set_yscale('log')
    axes[0].set_yticks([1, 1.2,1.4,1.6,1.8,2,2.2,2.4])
    axes[0].set_yticklabels([str(t) for t in [1, 1.2,1.4,1.6,1.8,2,2.2,2.4]])
    axes[0].tick_params(axis='y', labelsize=ylabel_font)
    print(axes[0].get_ylim())
    ymin, ymax = axes[0].get_ylim()

    axes[0].set_ylim(0.9, 2.5)
    print("ymin,ymax",ymin,ymax)
    

    
        
    # Add overall title
    # fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    
    # Add legend (only once, positioned at top)
    if num_plots > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        print(handles, labels)
        axbox = axes[0].get_position()
        print("!",axbox,axbox.x0+1.05*axbox.width+0.2, axbox.y0+axbox.height-0.1)
        fig.legend(handles, labels,
                   bbox_to_anchor=[axbox.x0+1.05*axbox.width+0.325, axbox.y0+axbox.height+0.165],
                   ncol=3,fontsize=legend_font,borderaxespad=0.3,handletextpad=0.3,frameon=False,)
        
    
    return fig
# Example 1: Different CSV files for each model/dataset combination
csv_files = [
    './dyn_tiling/figure_6_mixtral_b64_raw.csv',  
    './dyn_tiling/figure_6_qwen_b64_raw.csv',         
]

x_labels_list = [
    ['Tile=16','Tile=64','Dynamic', ],
    ['Tile=16','Tile=64','Dynamic', ]
]

subplot_titles = [
    'Mixtral8x7B',
    'Qwen3-30B-A3B', 
]


# Create the multi-plot chart
fig = transposed_create_multi_csv_bar_chart(
    csv_files=csv_files,figsize=(16, 8*0.6),
    x_labels_list=x_labels_list, 
    subplot_titles=subplot_titles,
    # ylim=(0.7, 3),
    ylim=(0.7, 2.5),
    # ylabel="Ratio of Cycles or Bytes\n(Log Scale)"
    ylabel="Normalized Ratio\n(Log Scale)"
)

plt.show()


# You can also save the plots
fig.savefig('./dyn_tiling/figure6.pdf', dpi=300, bbox_inches='tight')
fig.savefig('./dyn_tiling/figure6.png', bbox_inches='tight')