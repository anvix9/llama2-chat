import pandas as pd
import numpy as np

def compute_metrics_at_3(csv_file, model_name):
    """
    Compute precision, recall, F1 score, and MRR at @3 for model results.
    
    Args:
        csv_file (str): Path to the CSV file with model results
        model_name (str): Name of the model for reporting
        
    Returns:
        dict: Dictionary with precision, recall, F1 score, and MRR
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Calculate metrics for @3 (any of the three answers is correct)
    total_rows = len(df)
    
    # For each row, check if any of the three answers is True
    true_positives = sum((df['ans1'] | df['ans2'] | df['ans3']).astype(int))
    
    # All rows where all three answers are False are false negatives
    false_negatives = sum((~df['ans1'] & ~df['ans2'] & ~df['ans3']).astype(int))
    
    # Calculate precision and recall
    precision = true_positives / total_rows if total_rows > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate MRR
    reciprocal_ranks = []
    
    # Process each row for MRR
    for _, row in df.iterrows():
        # Get the positions of correct answers (True values)
        positions = [i+1 for i, ans in enumerate([row['ans1'], row['ans2'], row['ans3']]) if ans]
        
        # Calculate reciprocal rank (1/position of first correct answer)
        if positions:
            # First position where answer is correct
            first_correct = min(positions)
            reciprocal_rank = 1.0 / first_correct
        else:
            # No correct answers
            reciprocal_rank = 0.0
        
        reciprocal_ranks.append(reciprocal_rank)
    
    # Calculate MRR (mean of all reciprocal ranks)
    mrr = np.mean(reciprocal_ranks)
    
    # Additional statistics
    correct_at_position = {
        1: sum(1 for _, row in df.iterrows() if row['ans1']),
        2: sum(1 for _, row in df.iterrows() if not row['ans1'] and row['ans2']),
        3: sum(1 for _, row in df.iterrows() if not row['ans1'] and not row['ans2'] and row['ans3'])
    }
    
    return {
        'model_name': model_name,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mrr': mrr,
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'total_rows': total_rows,
        'correct_at_position': correct_at_position
    }

if __name__ == "__main__":
    # Compute metrics for both CSV files
    #results_llama3 = compute_metrics_at_3('./model_results_llama3.csv', 'Llama3')
    #results_llama3_l2 = compute_metrics_at_3('./model_results_llama3_l2.csv', 'Llama3_L2')

    results_llama2_chat = compute_metrics_at_3('./model_results_data0_llama2.csv', 'Llama2')
    results_llama2_l2 = compute_metrics_at_3('./model_results_data0_llama2_l2.csv', 'Llama2_L2')

    results_vicuna = compute_metrics_at_3('./model_results_data0_vicuna.csv', 'Vicuna')
    results_vicuna_l2 = compute_metrics_at_3('./model_results_data0_vicuna_l2.csv', 'Vicuna_l2')

    results_llama3 = compute_metrics_at_3('./model_results_data0_llam3.csv', 'llama3')
    results_llama3_l2 = compute_metrics_at_3('./model_results_data0_llam3_l2.csv', 'llama3_l2')

    print(f"=== {results_llama3['model_name']} Model Results ===")
    print(f"Precision at @3: {results_llama3['precision']:.4f}")
    print(f"Recall at @3: {results_llama3['recall']:.4f}")
    print(f"F1 Score at @3: {results_llama3['f1_score']:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {results_llama3['mrr']:.4f}")

    print(f"=== {results_llama3_l2['model_name']} Model Results ===")
    print(f"Precision at @3: {results_llama3_l2['precision']:.4f}")
    print(f"Recall at @3: {results_llama3_l2['recall']:.4f}")
    print(f"F1 Score at @3: {results_llama3_l2['f1_score']:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {results_llama3_l2['mrr']:.4f}")
 
    print(f"=== {results_vicuna['model_name']} Model Results ===")
    print(f"Precision at @3: {results_vicuna['precision']:.4f}")
    print(f"Recall at @3: {results_vicuna['recall']:.4f}")
    print(f"F1 Score at @3: {results_vicuna['f1_score']:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {results_vicuna['mrr']:.4f}")
 
    print(f"=== {results_vicuna_l2['model_name']} Model Results ===")
    print(f"Precision at @3: {results_vicuna_l2['precision']:.4f}")
    print(f"Recall at @3: {results_vicuna_l2['recall']:.4f}")
    print(f"F1 Score at @3: {results_vicuna_l2['f1_score']:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {results_vicuna_l2['mrr']:.4f}")
    
    print(f"=== {results_llama2_chat['model_name']} Model Results ===")
    print(f"Precision at @3: {results_llama2_chat['precision']:.4f}")
    print(f"Recall at @3: {results_llama2_chat['recall']:.4f}")
    print(f"F1 Score at @3: {results_llama2_chat['f1_score']:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {results_llama2_chat['mrr']:.4f}")
    
    print(f"=== {results_llama2_l2['model_name']} Model Results ===")
    print(f"Precision at @3: {results_llama2_l2['precision']:.4f}")
    print(f"Recall at @3: {results_llama2_l2['recall']:.4f}")
    print(f"F1 Score at @3: {results_llama2_l2['f1_score']:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {results_llama2_l2['mrr']:.4f}")
    

    # Print results for Llama3
    #print(f"=== {results_llama3['model_name']} Model Results ===")
    #print(f"Precision at @3: {results_llama3['precision']:.4f}")
    #print(f"Recall at @3: {results_llama3['recall']:.4f}")
    #print(f"F1 Score at @3: {results_llama3['f1_score']:.4f}")
    #print(f"Mean Reciprocal Rank (MRR): {results_llama3['mrr']:.4f}")
    #
    ## Print results for Llama3_L2
    #print(f"\n=== {results_llama3_l2['model_name']} Model Results ===")
    #print(f"Precision at @3: {results_llama3_l2['precision']:.4f}")
    #print(f"Recall at @3: {results_llama3_l2['recall']:.4f}")
    #print(f"F1 Score at @3: {results_llama3_l2['f1_score']:.4f}")
    #print(f"Mean Reciprocal Rank (MRR): {results_llama3_l2['mrr']:.4f}")
