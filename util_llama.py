from pathlib import Path 
import json 
import re 
import token_db

def load_json_files(directory='./data/'):
    """
    Load all JSON files from the specified directory.
    Args:
        directory (str): Path to the directory containing JSON files. Defaults to current directory.
    Returns:
        dict: Dictionary with filenames as keys and JSON content as values
    """
    
    # Get all JSON files in the directory
    try:
        with open("./data/2wikimqa_e.jsonl", 'r', encoding='utf-8') as file:
            # Load JSON content
            json_content = [json.loads(line) for line in file]
            # Use filename as keys
            json_files = json_content

    except json.JSONDecodeError as e:
        print(f"Error decoding ./data/2wikimqa_e.jsonl: {e}")
    except Exception as e:
        print(f"Error reading ./data/2wikimqa_e.jsonl: {e}")
    
    return json_files

def two_stage_retrieval(
    pinecone_service, 
    query: str, 
    namespace: str = "llama2-chat-wiki", 
    first_stage_k: int = 20, 
    final_k: int = 10
):
    # Stage 1: Local semantic search
    topics_dict = token_db.load_analysis_json_file()
    local_matches = token_db.find_best_matches(topics_dict, query, top_n=10)

    if not local_matches:
        return None

    paper_ids = [match["id"] for match in local_matches]
    best_match_id = paper_ids[0]
    remaining_ids = paper_ids[:] if len(paper_ids) > 1 else [paper_ids[0]]

    # Fetch vector and metadata for best match
    best_vector_data = pinecone_service.index.fetch(ids=[best_match_id], namespace=namespace)
    best_vector = best_vector_data.get("vectors", {}).get(best_match_id)

    if not best_vector:
        raise ValueError(f"Best match paper ID {best_match_id} not found in Pinecone")

    best_metadata = best_vector.get("metadata", {})

    # Optionally update the query based on metadata
    if namespace == "llama2-chat-wiki":
        query = best_metadata.get("question", query)

    # Stage 2: Fetch remaining documents for reranking
    print(remaining_ids)
    if len(remaining_ids) > 1:
        remaining_data = pinecone_service.index.fetch(ids=remaining_ids, namespace=namespace)

    elif len(remaining_ids) <= 1:
        remaining_data = pinecone_service.index.fetch(ids=remaining_ids, namespace=namespace)

    documents = [item["metadata"] for item in remaining_data.get("vectors", {}).values()]

    # Rerank and return top results
    reranked_results = pinecone_service.rerank_results(query, documents, final_k)
    return reranked_results


def fetch_and_query(pinecone_service, query, primary_namespace, secondary_namespace):
    """
    Fetch and rerank results based on an initial query and a fetched vector.

    Args:
        pinecone_service: The Pinecone service instance.
        query (str): The initial query string.
        primary_namespace (str): The namespace for the initial retrieval.
        secondary_namespace (str): The namespace for the secondary retrieval.

    Returns:
        dict: Final results from the secondary retrieval.
    """
    # Step 1: Perform initial retrieval
    results_questions = two_stage_retrieval(pinecone_service, query, namespace=primary_namespace)
    if not results_questions:
        raise ValueError("No results found in the initial retrieval.")

    return results_questions
