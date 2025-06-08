import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load all datasets
llama2_r3 = pd.read_csv("./llama2-chat-result-qa-wiki_r3.csv")
llama2_r7 = pd.read_csv("./llama2-chat-result-qa-wiki_r7.csv")

def compute_metrics(df):
    groups = df.groupby('id')
    accuracy, mrr, precision, recall = [], [], [], []

    for name, group in groups:
        group = group.sort_values('rank')
        correct_id = str(name).strip()
        preds = [str(pid).strip() for pid in group['predicted_id'].tolist()]
        k = len(preds)

        # Define a match: exact OR prefix match like "2303.1407" in "2303.14070" or "2303.1407v1"
        def is_match(pred):
            return pred == correct_id or pred.startswith(correct_id)

        matched_indices = [i for i, pred in enumerate(preds) if is_match(pred)]
        correct_found = 1 if matched_indices else 0
        accuracy.append(correct_found)
        recall.append(correct_found)

        if matched_indices:
            first_rank = matched_indices[0] + 1
            mrr.append(1 / first_rank)
        else:
            mrr.append(0.0)

        precision.append(1 / k if correct_found else 0.0)

    avg_precision = sum(precision) / len(precision)
    avg_recall = sum(recall) / len(recall)

    # F1 score
    if avg_precision + avg_recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

    return {
        'Accuracy': sum(accuracy) / len(accuracy),
        'MRR': sum(mrr) / len(mrr),
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F1': f1
    }

# Compute metrics for all datasets
datasets = {
    'llama2_r3': llama2_r3,
    'llama2_r7': llama2_r7,
#    'approach_v2_r5': approach_v2_r5,
}

# Compute results
# Compute and prepare results
results = {name: compute_metrics(df) for name, df in datasets.items()}

# Print results
for approach, metrics in results.items():
    print(f"Metrics for {approach}:")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  MRR:       {metrics['MRR']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1:        {metrics['F1']:.4f}")
    print()
