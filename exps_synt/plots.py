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

# Colors for each approach
colors = ['#2563eb', '#ea580c', '#6b7280', '#eab308']
markers = ['o', 's', '^', 'D']

# Create solid line plots with different colors
for i, (approach, color, marker) in enumerate(zip(approaches, colors, markers)):
    values = data[:, i]
    ax.plot(x, values, color=color, linestyle='-', linewidth=2.5,
             marker=marker, markersize=8, label=approach, markerfacecolor=color,
             markeredgecolor='white', markeredgewidth=1)

# Customize the plot
ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison - Line Plot', fontsize=14, fontweight='bold', pad=20)

# Set x-axis labels
ax.set_xticks(x)
ax.set_xticklabels(metrics)

# Set y-axis limits and ticks with reduced spacing
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1.1, 0.1))  # Y-axis ticks every 0.1 instead of default spacing

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend
ax.legend(loc='center right', framealpha=0.9, fancybox=True, shadow=True)

# Improve layout
plt.tight_layout()

# Show the plot
plt.savefig("./retrieval_plots/solid_techniques.png")
plt.show()
