import os
import json
import re
import pandas as pd

def extract_data_from_json_files(directory):
    """
    Extract paper IDs and queries from JSON files in the given directory.
    
    Args:
        directory (str): Path to the directory containing JSON files
        
    Returns:
        pandas.DataFrame: DataFrame with paper IDs and their queries
    """
    results = []
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    for filename in json_files:
        # Extract paper ID from filename using regex
        # Handle multiple patterns:
        # - air2023_2302.10866_analysis.json
        # - tests_2310.03533_analysis.json
        # - trending2024_2311.16867v2_analysis.json
        match = re.search(r'[a-zA-Z0-9]+_(\d+\.\d+(?:v\d+)?)_analysis\.json', filename)
        if not match:
            print(f"Could not extract paper ID from filename: {filename}")
            continue
            
        paper_id = match.group(1)
        
        # Read and parse the JSON file
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # Extract queries
            if 'queries' in data and isinstance(data['queries'], list):
                for query in data['queries']:
                    results.append({
                        'paper_id': paper_id,
                        'query': query
                    })
            else:
                print(f"No queries found in {filename}")
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error processing {filename}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    return df

def main():
    # Directory containing JSON files
    directory = './paper_analysis/'  # Current directory, change as needed
    
    # Extract data
    df = extract_data_from_json_files(directory)
    
    # Display in tabular format
    if not df.empty:
        print(df)
        
        # Optionally save to CSV
        df.to_csv('paper_queries.csv', index=False)
        print("Data saved to paper_queries.csv")
    else:
        print("No data found or extracted")

if __name__ == "__main__":
    main()
