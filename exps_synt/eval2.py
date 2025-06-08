import pandas as pd
import numpy as np

def compute_metrics_at_3(df, model_name, subset_name="entire"):
    """
    Compute precision, recall, F1 score, and MRR at @3 for model results.
    
    Args:
        df (DataFrame): Pandas DataFrame with model results
        model_name (str): Name of the model for reporting
        subset_name (str): Name of the length subset being analyzed
        
    Returns:
        dict: Dictionary with precision, recall, F1 score, and MRR
    """
    # Calculate metrics for @3 (any of the three answers is correct)
    total_rows = len(df)
    
    if total_rows == 0:
        return {
            'model_name': model_name,
            'subset_name': subset_name,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'mrr': 0,
            'true_positives': 0,
            'false_negatives': 0,
            'total_rows': 0,
            'correct_at_position': {1:0, 2:0, 3:0}
        }
    
    # For each row, check if any of the three answers is True
    true_positives = sum((df['ans1'] | df['ans2'] | df['ans3']).astype(int))
    
    # All rows where all three answers are False are false negatives
    false_negatives = total_rows - true_positives
    
    # Calculate precision and recall
    precision = true_positives / total_rows
    recall = true_positives / total_rows  # Same as precision in this context
    
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
            first_correct = min(positions)
            reciprocal_rank = 1.0 / first_correct
        else:
            reciprocal_rank = 0.0
        
        reciprocal_ranks.append(reciprocal_rank)
    
    # Calculate MRR (mean of all reciprocal ranks)
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    
    # Additional statistics
    correct_at_position = {
        1: sum(1 for _, row in df.iterrows() if row['ans1']),
        2: sum(1 for _, row in df.iterrows() if not row['ans1'] and row['ans2']),
        3: sum(1 for _, row in df.iterrows() if not row['ans1'] and not row['ans2'] and row['ans3'])
    }
    
    return {
        'model_name': model_name,
        'subset_name': subset_name,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mrr': mrr,
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'total_rows': total_rows,
        'correct_at_position': correct_at_position
    }

def process_model_file(file_path, model_name):
    """
    Process a single model file and compute metrics for all length ranges
    
    Args:
        file_path (str): Path to the model's CSV file
        model_name (str): Name of the model
        
    Returns:
        list: List of metric dictionaries for all subsets
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Clean and convert length column
    df['length'] = df['length'].astype(str).str.replace(',', '').astype(float)
    
    # Define length subsets
    subsets = [
        ('entire', (None, None)),
        ('0-2000', (0, 2000)),
        ('2000-4000', (2000, 4000)),
        ('0-4000', (0, 4000)),
        ('4000-8000', (4000, 8000))
    ]
    
    results = []
    
    for name, (low, high) in subsets:
        if name == 'entire':
            subset_df = df
        elif name == '0-2000':
            subset_df = df[df['length'] <= high]
        elif name == '0-4000':
            subset_df = df[df['length'] <= high]
        else:
            subset_df = df[(df['length'] > low) & (df['length'] <= high)]
        
        metrics = compute_metrics_at_3(subset_df, model_name, name)
        results.append(metrics)
    
    return results

if __name__ == "__main__":
    # Define model files
    model_files = {
        'Llama2': './old_results_200_e/model_results_llama2_chat.csv',
        'Llama2_L2': './old_results_200_e/model_results_llama2_chat_l2.csv',
        'Llama3': './old_results_200_e/model_results_llama3.csv',
        'Llama3_L2': './old_results_200_e/model_results_llama3_l2.csv',
    }
    
    all_results = []
    
    # Process each model file
    for model_name, file_path in model_files.items():
        model_results = process_model_file(file_path, model_name)
        all_results.extend(model_results)
    
    # Convert to DataFrame for better display and analysis
    results_df = pd.DataFrame(all_results)
    
    # Print all results
    print("All Results:")
    print(results_df[['model_name', 'subset_name', 'precision', 'recall', 'f1_score', 'mrr', 'total_rows']])
    
    # Save to CSV
    results_df.to_csv('all_metrics_results_e.csv', index=False)
    print("\nResults saved to 'all_metrics_results_e.csv'")
    
    # Print formatted results
    print("\nFormatted Results:")
    current_model = None
    for _, row in results_df.iterrows():
        if row['model_name'] != current_model:
            current_model = row['model_name']
            print(f"\n=== {current_model} ===")
        
        print(f"\nSubset: {row['subset_name']} ({row['total_rows']} queries)")
        print(f"Precision@3: {row['precision']:.4f}")
        print(f"Recall@3: {row['recall']:.4f}")
        print(f"F1 Score@3: {row['f1_score']:.4f}")
        print(f"MRR: {row['mrr']:.4f}")
        print(f"Correct by position: 1st={row['correct_at_position'][1]}, 2nd={row['correct_at_position'][2]}, 3rd={row['correct_at_position'][3]}")
