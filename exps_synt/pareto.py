import matplotlib.pyplot as plt
import numpy as np

# Data
technical_mrr = [0.862, 0.151, 0.133, 0.853]
conceptual_mrr = [0.744, 0.284, 0.263, 0.501]
technical_acc = [0.899, 0.174, 0.156, 0.917]
conceptual_acc = [0.789, 0.339, 0.307, 0.661]

# Metrics and approaches
metrics = ['technical_mrr', 'conceptual_mrr', 'technical_acc', 'conceptual_acc']
approaches = ['our approach', 'recursive', 'fixed-size', 'BM25']

# Organize data for plotting
data = np.array([technical_mrr, conceptual_mrr, technical_acc, conceptual_acc])

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# X positions
x = np.arange(len(metrics))

# Colors and dash styles for each approach
colors = ['#2563eb', '#ea580c', '#6b7280', '#eab308']
dash_styles = [
    (0, ()),           # solid line
    (0, (5, 5)),       # dashed line
    (0, (3, 5, 1, 5)), # dashdotted line  
    (0, (1, 1))        # dotted line
]
markers = ['o', 's', '^', 'D']

# Create line plots with dash styles
for i, (approach, color, dash, marker) in enumerate(zip(approaches, colors, dash_styles, markers)):
    values = data[:, i]
    ax.plot(x, values, color=color, linestyle=dash, linewidth=3, 
            marker=marker, markersize=10, label=approach, 
            markerfacecolor=color, markeredgecolor='white', markeredgewidth=2)

# Customize the plot
ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
ax.set_title('Performance Comparison - Dash Line Styles', fontsize=16, fontweight='bold', pad=20)

# Set x-axis labels
ax.set_xticks(x)
ax.set_xticklabels(metrics)

# Set y-axis limits
ax.set_ylim(0, 1)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', color='gray')

# Add legend
ax.legend(loc='center right', framealpha=0.9, fancybox=True, shadow=True, fontsize=12)

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()

