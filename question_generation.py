import os
import re
import json
import requests
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed

# Compile regex once
passage_splitter = re.compile(r'Passage \d+:')

def process_passage(passage, dataset_id, iter_):
    """Processes a single passage and returns its analysis."""
    topics = get_topics(passage)
    questions = get_llama_question(passage, f"{dataset_id}_{iter_}")
    queries = generate_queries(topics, questions) if topics and questions else []

    return {
        "keywords": topics,
        "questions": questions,
        "queries": queries
    }


def get_topics(content):
    """Extract main topics from the paper content using LLaMA."""
    _url = "http://127.0.0.1:11434/api/generate"
    print("Extracting topics...")
    
    _custom_prompt = (
        f"Based on this passage content, identify the main keywords it addresses."
        f"Format the response as a Python list of strings. Example format: '['keyword1', 'keyword2', 'keyword3', 'keyword19', etc..]'."
        f"Content: {content}"
    )
    
    _payload = {
        "model": "llama2:7b-chat",
        "prompt": _custom_prompt,
        "stream": False,
        "options": {"num_ctx": 4000, "temperature": 0.3},
        "keep_alive": -1
    }
    
    try:
        response = requests.post(_url, data=json.dumps(_payload))
        response.raise_for_status()
        response_data = response.json()
        
        # Clean the response to ensure it's a valid Python list
        topics_str = response_data['response'].strip()
        # Remove any markdown formatting if present
        topics_str = re.sub(r'```python|```', '', topics_str).strip()
        # Convert string representation of list to actual list
        print(f"Before : {topics_str}")
        result = parse_topics(topics_str)
        print("---")
        print(f"After parse: {topics_str}")
        return result 

    except Exception as e:
        print(f"Error extracting topics: {str(e)}")
        return ["Topic extraction failed"]


def parse_topics(topics_str):
    # Attempt to extract a bracketed list more robustly
    bracketed_list_match = re.search(r"\[\s*'[^]]*'\s*(?:,\s*'[^]]*'\s*)*\]", topics_str, re.DOTALL)
    
    if bracketed_list_match:
        list_str = bracketed_list_match.group(0)
        try:
            keywords = ast.literal_eval(list_str)
            if isinstance(keywords, list):
                return ", ".join(str(k) for k in keywords)
        except (SyntaxError, ValueError):
            pass

    # Fallback: parse lines
    keywords = []
    lines = topics_str.split('\n')
    
    for line in lines:
        stripped_line = line.strip()
        bullet_match = re.match(r'^\*\s+(.+)$', stripped_line)
        if bullet_match:
            keywords.append(bullet_match.group(1).strip())
            continue

        numbered_match = re.match(r'^\d+\.\s+(.+)$', stripped_line)
        if numbered_match:
            keywords.append(numbered_match.group(1).strip())
            continue

    return ", ".join(keywords) if keywords else ""


def get_llama_question(section, passage_id):
    """Generate questions based on the section content and theme using LLaMA."""
    _url = "http://127.0.0.1:11434/api/generate"
    print(f"generate questions for {passage_id}...")
    
    _custom_prompt = (
        f"Generate all simple questions that this passage can answer."
        f"Generate simple questions in terms of structures, they must not be complex questions."
        f"Generate specific and targeted questions directly, which are complete? The question should have enough context to not be too evasive.Do not generate the same questions."
        f"AVOID this kind of questions: 'What is the overall message conveyed by the text' or 'according to the passage.. .etc.."
        f"For each question, generate two (02) more which are paraphrased ones, and simpler questions that mimic the easy way human look for something through questions."
        f"Format: Q1: question? Q2: question? etc."
        f"Paragraph: {section}"
        )

    _payload = {
        "model": "llama2:7b-chat",
        "prompt": _custom_prompt,
        "stream": False,
        "options": {"num_ctx": 4000, "temperature": 0.3 },
        "keep_alive": -1
    }
    
    try:
        response = requests.post(_url, data=json.dumps(_payload))
        response.raise_for_status()
        response_data = response.json()
        return response_data['response']
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error: {err}")
        return "Error in request or code."


def generate_queries(topics, questions):
    """Generate realistic queries from topics and questions."""
    _url = "http://127.0.0.1:11434/api/generate"
    print("Generating queries...")
    
    _custom_prompt = (
        f"Based on the following topics and questions, generate realistic search queries that someone might use to find related information. "
        f"Topics: {topics}\nQuestions: {questions}\n"
        f"Return ONLY a list of search queries. Example format: ['query1', 'query2', 'query3']."
    )
    
    _payload = {
        "model": "llama2:7b-chat",
        "prompt": _custom_prompt,
        "stream": False,
        "options": {"num_ctx": 4000},
        "keep_alive": -1
    }
    
    try:
        response = requests.post(_url, data=json.dumps(_payload))
        response.raise_for_status()
        response_data = response.json()
        
        # Clean the response to ensure it's a valid Python list
        queries_str = response_data['response'].strip()
        # Remove any markdown formatting if present
        queries_str = re.sub(r'```python|```', '', queries_str).strip()
        # Convert string representation of list to actual list
        print(queries_str)

        # Step 2: Try extracting a Python-style list [....]
        list_match = re.search(r"\[.*?\]", queries_str, re.DOTALL)
        if list_match:
            try:
                keywords = ast.literal_eval(list_match.group(0))
                return ", ".join(map(str, keywords))
            except Exception as e:
                print(f"List eval failed: {e}")
        
        # Step 3: Extract numbered or marked questions (fallback)
        lines = queries_str.splitlines()
        question_patterns = [
            r'^\s*(?:Q\d+:|Q\d+\)|\d+[.)]|-)\s*(.+)$',  # Q1:, 1., 1), - item
            r'^\s*["“]?(.+?)["”]?\s*$',                 # Plain question lines
        ]
        
        questions = []
        for line in lines:
            for pattern in question_patterns:
                match = re.match(pattern, line)
                if match:
                    questions.append(match.group(1).strip())
                    break  # Stop after first matching pattern

        if questions:
            return ", ".join(questions)
        
        # Step 4: Fallback - return whole text if no parsing worked
        return queries_str

    except Exception as e:
        print("Error during extraction:", e)
        return ""

def process_jsonl_files(input_file, output_folder, max_workers=8):
    """Process JSONL entries from the input file using parallel passage analysis."""
    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, 'r', encoding="utf-8") as file:
        data_wiki = [json.loads(line) for line in file]

    for item_idx, current_item in enumerate(data_wiki):
        if item_idx <= 49:
            print("catchup...")
            continue
        
        iter_ = item_idx + 1
        print(f"Processing {current_item['_id']}... | {iter_}/{len(data_wiki)}")

        context = current_item.get('context', '')
        passages = passage_splitter.split(context)
        passages = [p.strip() for p in passages if p.strip()]

        complete = {'keywords': [], 'questions': [], 'queries': []}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_passage = {
                executor.submit(process_passage, p, current_item['dataset'], iter_): p for p in passages
            }

            for future in as_completed(future_to_passage):
                result = future.result()
                complete['keywords'].append(result['keywords'])
                complete['questions'].append(result['questions'])
                complete['queries'].append(result['queries'])

        output_filename = f"{current_item['dataset']}_{iter_}_analysis.json"
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(complete, outfile, indent=4)

        print(f"Saved analysis for {output_filename} in {output_folder}")
# Note: Ensure get_topics, get_llama_question, and generate_queries are defined as required.
# Example usage
input_file = "./data/2wikimqa_e.jsonl"
output_folder = "./paper_analysis/"
process_jsonl_files(input_file, output_folder)
