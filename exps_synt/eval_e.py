import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('all_metrics_results_e.csv')

# Filter out 'entire' subset to focus on context ranges
df_filtered = df[df['subset_name'] != 'entire'].copy()

# Create a mapping for better labels and numerical values for x-axis
context_mapping = {
    '0-2000': ('0-2K', 1),
    '2000-4000': ('2K-4K', 3), 
    '0-4000': ('0-4K', 2),
    '4000-8000': ('4K-8K', 6)
}

df_filtered['context_label'] = df_filtered['subset_name'].map(lambda x: context_mapping[x][0])
df_filtered['context_numeric'] = df_filtered['subset_name'].map(lambda x: context_mapping[x][1])

def create_performance_comparison_plot():
    """Create a comparison plot similar to Pass@K metrics"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use all context ranges but in a meaningful order
    context_order = ['0-2K', '2K-4K', '0-4K', '4K-8K']
    context_positions = [0.5, 1.5, 2.5, 4.0]  # Spaced positions
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    models = df_filtered['model_name'].unique()
    
    for i, model in enumerate(models):
        model_data = df_filtered[df_filtered['model_name'] == model]
        
        y_values = []
        x_positions = []
        
        for j, context in enumerate(context_order):
            context_row = model_data[model_data['context_label'] == context]
            if not context_row.empty:
                y_values.append(context_row['f1_score'].iloc[0])
                x_positions.append(context_positions[j])
        
        ax.plot(x_positions, y_values,
                color=colors[i],
                marker='^',
                markersize=8,
                linewidth=2.5,
                label=model)
    
    # Styling to match the reference image
    ax.set_xlabel('Context Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax.set_title('Model F1-Score Performance Analysis', fontsize=16, fontweight='bold')
    
    # Grid
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Set limits and ticks
    ax.set_ylim(0.4, 0.7)
    ax.set_yticks(np.arange(0.4, 0.71, 0.1))
    ax.set_xticks(context_positions)
    ax.set_xticklabels(context_order)
    
    # Legend - positioned outside plot area
    ax.legend(fontsize=12,
          loc='upper right',         # Position inside
          bbox_to_anchor=(0.98, 0.98),  # Fine-tuned padding from top-left
          frameon=True,             # Draw box around legend
          facecolor='white',        # Background for contrast
          edgecolor='black',        # Border color
          fancybox=True,            # Rounded corners
          framealpha=0.8)           # Slight transparency 
    
    plt.tight_layout()
    plt.savefig('performance_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_analysis_table():
    """Print detailed analysis similar to performance metrics"""
    
    print("=" * 60)
    print("DETAILED F1-SCORE PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Create summary table
    summary_data = []
    
    for model in df_filtered['model_name'].unique():
        model_data = df_filtered[df_filtered['model_name'] == model]
        
        row_data = {'Model': model}
        for _, row in model_data.iterrows():
            row_data[row['context_label']] = f"{row['f1_score']:.3f}"
        
        # Calculate improvement from shortest to longest context
        shortest = model_data[model_data['context_label'] == '0-2K']['f1_score'].iloc[0]
        longest = model_data[model_data['context_label'] == '4K-8K']['f1_score'].iloc[0]
        improvement = longest - shortest
        row_data['Δ (4K-8K vs 0-2K)'] = f"{improvement:+.3f}"
        
        summary_data.append(row_data)
    
    # Print formatted table
    print(f"{'Model':<12} {'0-2K':<8} {'2K-4K':<8} {'0-4K':<8} {'4K-8K':<8} {'Δ (4K-8K vs 0-2K)':<15}")
    print("-" * 70)
    
    for data in summary_data:
        print(f"{data['Model']:<12} {data.get('0-2K', 'N/A'):<8} {data.get('2K-4K', 'N/A'):<8} "
              f"{data.get('0-4K', 'N/A'):<8} {data.get('4K-8K', 'N/A'):<8} {data.get('Δ (4K-8K vs 0-2K)', 'N/A'):<15}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("- Positive Δ indicates better performance with longer contexts")
    print("- Negative Δ indicates performance degradation with longer contexts")
    print("=" * 60)

# Main execution
if __name__ == "__main__":
    print("Creating HumanEval-style performance plots...")
    
    # Generate the plots
    create_performance_comparison_plot()
    
    # Print analysis
    create_detailed_analysis_table()
    
    print("\nPlots generated:")
    print("humaneval_style_f1_plot.png - Main comparison plot")
    print("context_progression_plot.png - Context window progression")
    print("performance_comparison_plot.png - Detailed performance analysis")
