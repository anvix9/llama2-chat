import json
import os
import requests
from pathlib import Path

def get_tethy_summary(content):
    url = "http://127.0.0.1:11434/api/generate"
    prompt = (
        f"Summarize this paper content into a bulleted list of main results, Methods and contributions. Just give the answer without any polite texts before."
        f"Be short and concise no more than 700 characters."
        f"""This is a Template example you need to follow to generate the questions: 
        Q1: What are the primary challenges faced by researchers and developers in utilizing large language models for software development tasks?

        A: The major challenge lies in the performance gap between open-source models and closed-source models, with the former being inaccessible to many researchers and developers due to their proprietary nature.

        Q2: How do the authors address this challenge by developing the DeepSeek-Coder series of open-source code models?

        A: The authors introduce a range of open-source code models with sizes from 1.3B to 33B, trained from scratch on 2 trillion tokens sourced from 87 programming languages, ensuring a comprehensive understanding of coding languages and syntax.

        Q3: What specific enhancements and innovations does the DeepSeek-Coder series bring to the field of software development?

        A: The authors develop several innovative techniques, including the 'fill-in-the-blank' pre-training objective, the extension of the context window to 16K tokens, and the incorporation of the Fill-In-Middle (FIM) approach, which significantly bolster the models' code completion capabilities.

        Q4: What are the main contributions of the authors in this study?

        A: The authors make several key contributions, including:

        * Introducing DeepSeek-Coder-Base and DeepSeek-Coder-Instruct, advanced code-focused large language models.
        * Developing repository-level data construction during pre-training, which significantly boosts cross-file code generation capabilities.
        * Conducting extensive evaluations of the code LLMs against various benchmarks, demonstrating their superiority over existing open-source models.

        Contribution: The authors' work introduces a series of specialized Large Language Models (LLMs) for coding, including the DeepSeek-Coder series, which provides significant advancements in open-source code modeling.
        """
        f"Focus on key findings and avoid technical details and add the key words from the topics at the end. Content: {content}"
    )
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        return response.json()['response']
    except Exception as e:
        print(f"Error getting summary: {e}")
        return None

def generate_paper_cards():
    # Load paper titles
    with open('paper_metadata/metadata_2025_03_10_103708.json', 'r') as f:
        papers = json.load(f)
    
    # Create output directory
    Path('card_papers').mkdir(exist_ok=True)
    
    # Process each analysis file
    iter_ = 0
    analysis_files = Path('paper_analysis').glob('*_analysis.json')
    for analysis_path in analysis_files:
        paper_id = analysis_path.stem.replace('_analysis', '')
        tag1, paper_id = paper_id.split('_')

        with open(analysis_path, 'r') as f:
            analysis = json.load(f)
        
        # Get paper title
        #title = titles.get(paper_id, "Unknown Title")
        title = next((paper['title'] for paper in papers if paper['id'] == paper_id), "Unknown Title")
        
        # Combine content for summary
        content = f"{analysis['research']}\n{analysis['method']}\n{analysis['results']}"
        summary = get_tethy_summary(content)
        summary = summary.split(":")[-1]
        
        try:
            tmp_topics = ", ".join(analysis['topics'])
        except:
            tmp_topics = "AI, ML."
        # Generate markdown content
        md_content = f"""# {title}

# Research questions
{analysis['research']}

## Problem Statement, Methods and Main Results
{summary}

#### Keywords: {tmp_topics}\n

### [Link to paper](https://arxiv.org/abs/{paper_id})
        """
        
        # Save to markdown file
        print(f"Generating card for {tag1}_{paper_id}")
        output_path = f"card_papers/{tag1}_{paper_id}_card.md"
        with open(output_path, 'w') as f:
            f.write(md_content)
        iter_ +=1 

if __name__ == "__main__":
    generate_paper_cards()
