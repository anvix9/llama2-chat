import json 
import pandas as pd
import utils 
import re

# id , query 

# id = filename split _ [1]
# query = each q in queries 


files = utils.load_json_files()
new_data = []

print(f"num of files {len(files)}")

for filename, content in files.items():
    
    id_ = filename.rsplit('_', 1)[0]

    try:
        p = content['queries']
        p2 = content['keywords']

        # Extract 'research' field
        research_text = content.get('questions', '')
        #match = re.search(r':(.*?)\?', research_text)
        #question = match.group(1).strip() if match else None
        #print(question, id_)

        new_data.append({'id': f"{id_}", 'questions': research_text, 'queries': p2})
    
    except Exception as e:
        print(f"Error: {e}")

# Save to new JSON file
with open("./queries_bank/extracted_queries.json", "w") as f:
    json.dump(new_data, f, indent=2)

