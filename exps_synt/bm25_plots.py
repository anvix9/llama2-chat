import matplotlib.pyplot as plt
import pandas as pd

# Updated data with labels
data = [
    {"technique": "bm25_card_v2_r3", "type": "card", "query": "technical", "accuracy": 0.9633, "mrr": 0.9373},
    {"technique": "bm25_card_v2_r5", "type": "card", "query": "technical", "accuracy": 0.9725, "mrr": 0.9396},
    {"technique": "bm25_card_v3_r3", "type": "card", "query": "conceptual", "accuracy": 0.8899, "mrr": 0.8502},
    {"technique": "bm25_card_v3_r5", "type": "card", "query": "conceptual", "accuracy": 0.9174, "mrr": 0.8557},
    {"technique": "bm25_abs_v2_r3", "type": "abs", "query": "technical", "accuracy": 0.8991, "mrr": 0.8456},
    {"technique": "bm25_abs_v2_r5", "type": "abs", "query": "technical", "accuracy": 0.9358, "mrr": 0.8543},
    {"technique": "bm25_abs_v3_r3", "type": "abs", "query": "conceptual", "accuracy": 0.6606, "mrr": 0.5627},
    {"technique": "bm25_abs_v3_r5", "type": "abs", "query": "conceptual", "accuracy": 0.7156, "mrr": 0.5751},
]

df = pd.DataFrame(data)

# Create cleaner technique labels
df['clean_label'] = df['technique'].str.replace('bm25_', '').str.replace('_', ' ').str.upper()
df['version'] = df['technique'].str.extract(r'(v[23])')
df['rank'] = df['technique'].str.extract(r'(r[35])')

def create_technical_plot():
    fig, ax1 = plt.subplots(figsize=(7, 5))
    df_tech = df[df["query"] == "technical"]
    
    tech_card = df_tech[df_tech['type'] == 'card']
    tech_abs = df_tech[df_tech['type'] == 'abs']
    
    ax1.plot(range(len(tech_card)), tech_card['accuracy'], 'o-', 
             color='#2E86AB', linewidth=3, markersize=10, label='Card - Accuracy')
    ax1.plot(range(len(tech_card)), tech_card['mrr'], 's--', 
             color='#2E86AB', linewidth=3, markersize=10, alpha=0.7, label='Card - MRR')
    
    ax1.plot(range(len(tech_abs)), tech_abs['accuracy'], 'o-', 
             color='#A23B72', linewidth=3, markersize=10, label='Abstract - Accuracy')
    ax1.plot(range(len(tech_abs)), tech_abs['mrr'], 's--', 
             color='#A23B72', linewidth=3, markersize=10, alpha=0.7, label='Abstract - MRR')
    
    ax1.set_title('Technical Queries Performance', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Rank Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.8, 1.0)
    ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(['@3', '@5'])
    return fig

def create_conceptual_plot():
    fig, ax2 = plt.subplots(figsize=(7, 5))
    df_conc = df[df["query"] == "conceptual"]
    
    conc_card = df_conc[df_conc['type'] == 'card']
    conc_abs = df_conc[df_conc['type'] == 'abs']
    
    ax2.plot(range(len(conc_card)), conc_card['accuracy'], 'o-', 
             color='#F18F01', linewidth=3, markersize=10, label='Card - Accuracy')
    ax2.plot(range(len(conc_card)), conc_card['mrr'], 's--', 
             color='#F18F01', linewidth=3, markersize=10, alpha=0.7, label='Card - MRR')
    
    ax2.plot(range(len(conc_abs)), conc_abs['accuracy'], 'o-', 
             color='#C73E1D', linewidth=3, markersize=10, label='Abstract - Accuracy')
    ax2.plot(range(len(conc_abs)), conc_abs['mrr'], 's--', 
             color='#C73E1D', linewidth=3, markersize=10, alpha=0.7, label='Abstract - MRR')
    
    ax2.set_title('Conceptual Queries Performance', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Rank Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylim(0.5, 1.0)
    ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(['@3', '@5'])
    return fig

fig_tech = create_technical_plot()
fig_tech.savefig("./retrieval_plots/bm25_technical_performance.png", bbox_inches='tight')

fig_conc = create_conceptual_plot()
fig_conc.savefig("./retrieval_plots/bm25_conceptual_performance.png", bbox_inches='tight')


