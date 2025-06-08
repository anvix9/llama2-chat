
from dotenv import load_dotenv
import services 
import utils 
import os 
import csv
import pandas

load_dotenv()
api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('PINECONE_INDEX_NAME')
pc = services.PineconeService(api_key, index_name)


# Iterate through queries, fetch results, and store them
def process_queries(input_csv_path, output_csv_path):
    with open(input_csv_path, 'r') as infile, open(output_csv_path, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(['id', 'query', 'rank', 'predicted_id'])
        
        for row in reader:
            origin_id = row['id']
            queries = [row['input']]
            
            for q_id, q in zip(origin_id, queries):
                
                try:
                    results = utils.fetch_and_query(
                        pc, 
                        query=q, 
                        primary_namespace='llama2-chat-wiki', 
                        secondary_namespace='llama2-chat-wiki'
                    )
                    top3 = results[:7] 
                    print("---")
                    print(top3)
                    print(origin_id)
                                # Write rows for each rank
                    for rank, pred_id in enumerate(top3, start=1):
                        writer.writerow([origin_id, q, rank, pred_id])

                except ValueError as e:
                    print(f"Error: {e}")

# Save results to a CSV file
output_file = "llama2-chat-result-qa-wiki_r7.csv"
#results_df.to_csv(output_file, index=False)
#print(f"Results saved to {output_file}") 
process_queries('./llama2-gold-test_custom_id.csv', output_file)
