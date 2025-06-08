import requests
import pandas as pd
import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import defaultdict
import spacy
import time  # Added for delay between API calls

class TextProcessor:
    def __init__(self, spacy_model="en_core_web_sm"):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        self.nlp = spacy.load(spacy_model)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        punctuation = string.punctuation.replace("'", "")
        translation_table = str.maketrans("'", " ", punctuation)
        text = text.translate(translation_table).lower()
        words = nltk.word_tokenize(text)
        return ' '.join(
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words and word.isalpha()
        )

    def recursive_pos_split(self, text, split_pos_tags=['ADP', 'CCONJ'], split_punct=[':']):
        segments = [text]
        changed = True
        
        while changed:
            changed = False
            new_segments = []
            
            for segment in segments:
                doc = self.nlp(segment)
                split_points = [i for i, token in enumerate(doc) 
                               if token.pos_ in split_pos_tags or 
                               (token.is_punct and token.text in split_punct)]
                
                if not split_points:
                    new_segments.append(segment)
                    continue
                
                changed = True
                start, segments_to_add = 0, []
                
                for point in split_points:
                    end = doc[point].idx
                    if (new_segment := segment[start:end].strip()):
                        segments_to_add.append(new_segment)
                    start = end + len(doc[point].text)
                
                if (remaining := segment[start:].strip()):
                    segments_to_add.append(remaining)
                
                new_segments.extend(segments_to_add)
            
            segments = new_segments
        
        return [self.preprocess_text(seg) for seg in segments]

class PassageRanker:
    def __init__(self, text_processor=None):
        self.text_processor = text_processor or TextProcessor()
    
    def process_jsonl_file(self, file_path):
        result = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                context = item['context']
                passages = []
                
                for part in context.split('\n'):
                    if part.startswith('Passage ') and ':' in part:
                        if passages:  # Finalize current passage
                            passages[-1] = '\n'.join(passages[-1])
                        passages.append([])
                    elif passages or part.strip():
                        if not passages:  # Handle content before first Passage header
                            passages.append([])
                        passages[-1].append(part)
                
                if passages:
                    passages[-1] = '\n'.join(passages[-1])
                
                result.append({
                    'id': item['_id'],
                    'input': item['input'],
                    'passages': passages,
                    'answer': item['answers']  # Added to capture original answer
                })
        return result

def analyze_passages(query_tokens, passages):
    """
    Analyze passages based on query token co-occurrence and frequency.
    
    Args:
        query_tokens (list): List of tokens from the query
        passages (list): List of passage strings to analyze
        
    Returns:
        dict: Dictionary containing ranked passages and analysis results
    """
    results = []
    
    for i, passage in enumerate(passages):
        passage_lower = passage.lower()
        
        # Count occurrences of each token in the passage
        token_counts = {}
        for token in query_tokens:
            # Count how many times each token appears in the passage
            count = passage_lower.count(token)
            token_counts[token] = count
        
        # Count how many different tokens from the query appear in the passage
        tokens_present = sum(1 for token, count in token_counts.items() if count > 0)
        
        # Calculate total occurrences of all tokens in the passage
        total_occurrences = sum(token_counts.values())
        
        results.append({
            'passage_index': i,
            'passage_text': passage,
            'token_counts': token_counts,
            'tokens_present': tokens_present,
            'total_occurrences': total_occurrences
        })
    
    # Sort results by:
    # 1. Number of different tokens present (descending)
    # 2. Total occurrences of all tokens (descending)
    sorted_results = sorted(
        results, 
        key=lambda x: (x['tokens_present'], x['total_occurrences']), 
        reverse=True
    )
    
    return {
        'query_tokens': query_tokens,
        'ranked_passages': sorted_results
    }

def format_results(analysis_results):
    """
    Format the analysis results for display.
    
    Args:
        analysis_results (dict): Results from analyze_passages function
        
    Returns:
        str: Formatted string with analysis results
    """
    query_tokens = analysis_results['query_tokens']
    ranked_passages = analysis_results['ranked_passages']
    
    output = []
    output.append(f"Query tokens: {', '.join(query_tokens)}")
    output.append(f"Total passages analyzed: {len(ranked_passages)}")
    output.append("\nRanked Passages (by token co-occurrence and frequency):\n")
    
    for i, result in enumerate(ranked_passages):
        output.append(f"Rank #{i+1} (Passage #{result['passage_index']+1}):")
        output.append(f"  Tokens present: {result['tokens_present']} out of {len(query_tokens)}")
        output.append(f"  Total token occurrences: {result['total_occurrences']}")
        
        # Token breakdown
        output.append("  Token counts:")
        for token, count in result['token_counts'].items():
            output.append(f"    - '{token}': {count}")
        
        # Show passage excerpt (first 100 chars)
        excerpt = result['passage_text'][:100] + "..." if len(result['passage_text']) > 100 else result['passage_text']
        output.append(f"  Passage excerpt: \"{excerpt}\"")
        output.append("")
    
    return "\n".join(output)

def get_filtered_passages(analysis_results, min_tokens=1, top_n=None, rerank_by_original=False):
    """
    Get filtered list of passages based on minimum token presence.
    
    Args:
        analysis_results (dict): Results from analyze_passages function
        min_tokens (int): Minimum number of different query tokens that must be present
        top_n (int, optional): Limit results to top N passages after filtering
        rerank_by_original (bool): Whether to rerank the top passages by their original position
        
    Returns:
        list: Filtered list of passages
    """
    ranked_passages = analysis_results['ranked_passages']
    
    # Filter by minimum token presence
    filtered = [result for result in ranked_passages if result['tokens_present'] >= min_tokens]
    
    # Limit to top N if specified
    if top_n is not None and top_n > 0:
        filtered = filtered[:top_n]
    
    # Rerank by original position if requested
    if rerank_by_original and filtered:
        filtered.sort(key=lambda x: x['passage_index'])
    
    # Return just the passage text
    return [result['passage_text'] for result in filtered]

def get_llama_answer(passages, input_question):
    """Generate answer using LLaMA model"""
    url = "http://127.0.0.1:11434/api/generate"
    print(f"Passage length: {len(passages)}")
    prompt = (
        f"Answer the question based on the given passages. Pay attention to details. "
        f"Provide ONLY the final answer without any additional text.\n\n"
        f"Passages:\n{passages}\n\n"
        f"Question: {input_question}\n"
        f"Answer:"
    )
    
    payload = {
        "model": "llama3",  # Using llama2-chat model
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.5},
        "keep_alive": -1
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()['response'].strip()
    except Exception as e:
        print(f"API Error: {str(e)}")
        return "ERROR"

def check_answer_match(original_answer, model_answer):
    """Check if all words from original answer appear in model answer (case-insensitive)"""
    original_words = set(re.findall(r'\w+', original_answer[0].lower()))
    model_words = set(re.findall(r'\w+', model_answer.lower()))
    return original_words.issubset(model_words)

if __name__ == "__main__":
    ranker = PassageRanker()
    processed_data = ranker.process_jsonl_file('data_o.jsonl')
    
    results = []
    
    for row in processed_data:
        # Process query and get tokens
        query_tokens = ranker.text_processor.preprocess_text(row['input'])
        
        # Use analyze_passages and get_filtered_passages for passage filtering
        analysis_results = analyze_passages(query_tokens.split(), row['passages'])
        filtered_passages = get_filtered_passages(analysis_results, min_tokens=2, top_n=6, rerank_by_original=True)
        
        # Join filtered passages for the model input
        sorted_passages = " ".join(filtered_passages[:5])
        
        # Get 3 model answers with 2-second intervals
        model_answers = []
        for i in range(3):
            answer = get_llama_answer(sorted_passages, row['input'])
            model_answers.append(answer)
            if i < 2:  # Wait between calls (except after last)
                time.sleep(2)
        
        # Check matches for each answer
        matches = [check_answer_match(row['answer'], ans) for ans in model_answers]
        
        # Collect results for CSV
        results.append({
            'input': row['input'],
            'ans1': matches[0],
            'ans2': matches[1],
            'ans3': matches[2],
            # Additional debugging info (optional):
            'original_answer': row['answer'][0],
            'model_ans1': model_answers[0],
            'model_ans2': model_answers[1],
            'model_ans3': model_answers[2]
        })
        print(f"Processed: {row['input'][:50]}... | Matches: {matches}")

    # Create and save CSV
    result_df = pd.DataFrame(results)
    
    # Select columns for final output
    final_columns = ['input', 'ans1', 'ans2', 'ans3'] 
    result_df[final_columns].to_csv('model_results_data0_llam3_l2.csv', index=False)
    print("Results saved to model_results_vicuna_l2.csv")

