from model2vec import StaticModel
from pathlib import Path
from typing import Dict
import json
import util_llama
import re
from nltk.corpus import stopwords
from typing import List, Tuple
from nltk.tokenize import word_tokenize
import codecs

# Load a model from the HuggingFace hub (in this case the potion-base-8M model)
#model = StaticModel.from_pretrained("minishlab/potion-base-8M")

# Make sequences of token embeddings
#token_embeddings = model.encode_as_sequence(["It's dangerous to go alone!", "It's a secret to everybody."])

#print(token_embeddings[0][0])

# Known failure messages
FAILURE_MESSAGES = {
    "Query generation failed",
    "Topic extraction failed",
    "Question generation failed"
}

def is_valid_content(value):
    if not value:
        return False
    if isinstance(value, str):
        return value.strip() and value.strip() not in FAILURE_MESSAGES
    if isinstance(value, list):
        # Filter out lists with only failure messages or empties
        return any(is_valid_content(v) for v in value)
    return True  # fallback for other types

def extract_valid_queries(data):
    for key in ["keywords", "queries", "questions" ]:
        value = data.get(key)
        if is_valid_content(value):
            return value
    return ""

def clean_text(text):
    # 0. Decode Unicode escape sequences if needed
    try:
        text = codecs.decode(text, 'unicode_escape')
    except Exception:
        pass  # In case text is already decoded or has invalid escape sequences

    # 1. Remove punctuation (except maybe letters with accents)
    text_no_punct = re.sub(r'[^\w\s]', '', text)

    # 2. Convert to lowercase
    text_lower = text_no_punct.lower()

    # 3. Tokenize
    tokens = word_tokenize(text_lower)

    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.discard('i')  # Ensure 'i' is kept
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # 5. Join back
    cleaned_text = ' '.join(filtered_tokens)

    return cleaned_text

def load_analysis_json_file(directory: str = './paper_analysis/') -> Dict[str, Dict]:
        """
        Load json files containing QA data. 
        
        Args:
            directory: Directory containing json analysis files
            papers_json_path: Path to JSON file containing paper metadata
                
        Returns:
            Dictionary with paper_id as key and dict containing analysis content, questions and metadata as value
        """
        
        # Load and process analysis files
        files_content = [] 
        
        iter_ = 0

        for file_path in Path(directory).glob('*_analysis.json'):
            iter_ += 1
            try:
                # Read JSON
                with open(file_path, 'r', encoding='utf-8') as file:
                    analysis_data = json.load(file)

                # Extract ID
                paper_id = file_path.stem.replace('_analysis', '')

                # Get valid query content
                queries = extract_valid_queries(analysis_data)
                if not queries:
                    print(f"[{iter_}] Skipped {file_path.name}: No valid queries")
                    continue

                # Clean and store
                queries = [ i for i in queries if not isinstance(i, list)]
                queries = " ".join(queries)
                cleaned_q = clean_text(queries)
                files_content.append((paper_id, cleaned_q))

            except Exception as e:
                print(f"[{iter_}] Error processing {file_path.name}: {e}")
                
        return files_content   

def find_best_matches(entries: List[Tuple[str, str]], query: str, top_n: int = 10) -> List[dict]:
    query_clean = clean_text(query)
    query_words = set(query_clean.split())
    matches = []
    
    for entry_id, keywords in entries:
        c_key = clean_text(keywords)
        keyword_words = set(c_key.lower().split())
        matching = query_words.intersection(keyword_words)  # Intersection
        #print(f"{query_words} || {keyword_words}")
        #print("---")

        if matching:
            matches.append({
                "id": entry_id,
                "match_count": len(matching),
                "matching_words": matching
            })
    
    # Sort by match count (descending) and ID (ascending)
    sorted_matches = sorted(
        matches, 
        key=lambda x: (-x["match_count"], x["id"])
    )
    print(sorted_matches[:9])
    return sorted_matches[:top_n]

#t = load_analysis_json_file("./paper_analysis/")
##
#q = "Who is Hermann Ii, Count Palatine Of Lotharingia's paternal grandfather?"
##
#best_match_result = find_best_matches(t, q)
#print(len(best_match_result))
