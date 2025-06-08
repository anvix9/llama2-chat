from cgi import print_arguments, print_directory
from model2vec import StaticModel
from pathlib import Path
from typing import Dict
import json
import re
from nltk.corpus import stopwords
from typing import List, Tuple, Optional, Union
from nltk.tokenize import word_tokenize
import os 
from typing import Dict, List, Tuple
import services
from flashrank import Ranker, RerankRequest
from markdown import markdown
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import requests


# Ollama embedding generation
def get_embeddings(content):
    """
    Get embeddings from Ollama with proper normalization

    Args:
        content: Text content (string) or a dictionary of {text: id}

    Returns:
        A list of dictionaries with 'id' and normalized 'embedding',
        or None if the API call fails
    """
    url = "http://127.0.0.1:11434/api/embed"
    model_name = "saish_15/tethysai_research"

    def fetch_embedding(text: str) -> Optional[List[float]]:
        payload = {
            "model": model_name,
            "input": text,
            "options": {"temperature": 0}
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            raw_embeddings = response.json().get('embeddings')

            if not raw_embeddings:
                print("No embeddings returned from API")
                return None

            embedding_array = np.array(raw_embeddings[0])
            normalized = embedding_array / np.linalg.norm(embedding_array)
            return normalized.tolist()
        except Exception as e:
            print(f"Error extracting embedding: {str(e)}")
            return None

    results = []

    if isinstance(content, dict):
        id_ = content['id']
        text = content['text']
        embedding = fetch_embedding(text)

        if embedding is not None:
            results.append({'id': id_, 'embedding': embedding})

    elif isinstance(content, str):
        embedding = fetch_embedding(content)
        if embedding is not None:
            results.append({'id': 'query', 'embedding': embedding})
    else:
        raise TypeError("content must be either a dict or a string")

    return results if results else None


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # 1. Remove punctuation
    text_no_punct = re.sub(r'[^\w\s]', ' ', text)  
    
    # 2. Convert to lowercase
    text_lower = text_no_punct.lower()  
    
    # 3. Tokenize the text
    tokens = word_tokenize(text_lower)
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    for word in ['be', 'how', 'why', 'what', 'when', 'where', 'can', 'is', 'are']:
        stop_words.discard(word)  # keep these interrogatives and auxiliaries

    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # 6. Join back into a sentence
    cleaned_text = ' '.join(lemmatized_tokens)
    
    return cleaned_text

#def markdown_to_text(md):
#    html = markdown(md)
#    soup = BeautifulSoup(html, "html.parser")
#    return soup.get_text()
#
def load_analysis_json_files(base_directory: str = './queries_bank/') -> List[Tuple[str, str]]:
    files_content = []

    #if base_directory == './abstract/':
    ## === Load and Process Markdown Files ===

    #    markdown_folder = "./abstract/"
    #    documents = []
    #    file_map = []  # maps paper_id (stripped) to full filename

    #    for filename in os.listdir(markdown_folder):
    #        if filename.endswith(".md"):
    #            path = os.path.join(markdown_folder, filename)
    #            with open(path, "r", encoding="utf-8") as file:
    #                text = markdown_to_text(file.read())
    #                documents.append(text)
    #                paper_id = strip_prefix(filename)
    #                files_content.append((paper_id, text)) 

    #    return files_content

    for root, dirs, files in os.walk(base_directory):
        if 'extracted_queries.json' in files:
            json_path = os.path.join(root, 'extracted_queries.json')

            with open(json_path, 'r', encoding='utf-8') as file:
                try:
                    analysis_data = json.load(file)
                except json.JSONDecodeError:
                    print(f"Could not decode JSON from {json_path}")
                    continue

            for entry in analysis_data:
                id = entry.get("id", "") 
                keywords = entry.get("queries", "")
                question = entry.get("questions", "")

                if isinstance(keywords, list):
                    keywords = ' '.join(keywords)

                if isinstance(question, list):
                    question = ' '. join(question)

                cleaned_keywords = clean_text(keywords)
                clean_question = clean_text(question)
                res = cleaned_keywords + clean_question
                files_content.append((id, res))

    return files_content


def find_best_matches(
    entries: List[Tuple[str, str]],
    query: str,
    top_n: int = 20
) -> List[Dict]:

    query_clean = clean_text(query)
    query_words = set(query_clean.split())
    # your special lists

    matches = []
    for entry_id, keywords in entries:
        c_key = clean_text(keywords)
        kw_words = set(c_key.split())
        # 1) normal intersection
        matching = query_words & kw_words

        if matching:
            matches.append({
                "text":            keywords,
                "id":              entry_id,
                "match_count":     len(matching),
                "matching_words":  matching,
                #"question": _
            })

    # sort by descending match_count, then by id
    matches.sort(key=lambda x: (-x["match_count"], x["id"]))
    return matches[:top_n]

def rerank_passages(
    query: str,
    passages: List[str],
    top_n: int = 5,
    model_name: str = "ms-marco-MiniLM-L-12-v2",
    cache_dir: str = "~/cache") -> List[Dict]:
    """
    Rerank a list of plain-text passages against a single query,
    returning the top_n as a list of dicts with keys 'text' and 'score'.
    """

    # Build and run the rerank request
    rerank_request = RerankRequest(query=query, passages=passages)
    ranker = Ranker(model_name=model_name, cache_dir=cache_dir)
    raw_results = ranker.rerank(rerank_request)
    #
    ## raw_results is a list of dicts with 'text' and 'score'
    top = sorted(raw_results, key=lambda r: r['score'], reverse=True)[:top_n]
    # MMR on top
    ids = [i['id'] for i in top]

    #print("in")
    #doc_embeddings = [get_embeddings(p) for p in top]
    #query_embedding = get_embeddings(query)
    #doc_embeddings = [ i for sub in doc_embeddings for i in sub]

    #final_list = mmr(doc_embeddings, query_embedding, lambda_param=0.5, top_k=5)

    return ids 


def find_and_rerank(
    entries: List[Tuple[str, str]],
    user_query: str,
    top_k_matches: int = 20,
    top_n_rerank: int = 5
) -> None:

    # Stage 1: retrieve best-matching entries
    best = find_best_matches(entries, user_query, top_n=top_k_matches)
    if not best:
        print("No keyword matches found.")
        return ' '

    print(f"User query: {user_query!r}")
    print(f"Top keyword-match: (matched {best[0]['match_count']} words)")
    
    # Stage 2: prepare plain-text passages and rerank
    docs = []
    for i in best:
        tmp_id = i['id']
        tmp_text = i['text']
        tmp_e = {'text': tmp_text, 'id': tmp_id}
        docs.append(tmp_e)


    reranked = rerank_passages(best[0]['text'], docs, top_n=top_n_rerank)
    #print(reranked)
    #print("\nTop reranked passages:")
    #for i, r in enumerate(reranked, 1):
    #    print(f"{i}.| ID: {r['id']!r}")

    #ids = [r['id'] for r in reranked]
    ids = reranked

    return ids 

#t = load_analysis_json_files()
#### List[Tuple[id, text]]
####print(len(t))
#q = "Who was born first, Damien Hétu or Matan Cohen?"
#ids = find_and_rerank(t, q, top_k_matches=20, top_n_rerank=10)
#print(ids)

