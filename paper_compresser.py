import re
import os
from pathlib import Path

def extract_sections(input_file, sections_to_extract):
    """
    Extract sections from a Markdown file based on target section headers starting with '##'.

    Args:
        input_file: Path to the input markdown file
        sections_to_extract: List of section names to extract

    Returns:
        str: Extracted content in markdown format
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Normalize section names for case-insensitive matching
    sections_to_extract = {section.lower() for section in sections_to_extract}
    # Regex to match sections starting with '##'
    section_pattern = re.compile(r"^##\s*(.+)$", re.MULTILINE)

    # Find all sections and their start positions
    sections = list(section_pattern.finditer(content))
    extracted_content = ""

    for i, match in enumerate(sections):
        section_name = re.sub(r'[^a-zA-Z0-9\s]', '', match.group(1).strip().lower()) 
        section_name = section_name.split(' ')[-1]
        if section_name in sections_to_extract:
            print(section_name)
            start = match.end()
            end = sections[i + 1].start() if i + 1 < len(sections) else len(content)
            section_content = content[start:end].strip()
            extracted_content += f"## {match.group(1).strip()}\n\n{section_content}\n\n"

    return extracted_content

def main():
    # File paths
    input_dir = './converted_markdowns/'
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, 'paper_compressed')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sections_to_extract = [
        "Abstract", "Introduction", "Method", "Conclusion","study", "studies", "discussion", "preliminaries","preliminary",
        "Summary", "Overview", "Background", "Future Work", "Motivation", "Problem Statement", "conclusions", "methodologies","methods",
        "approach", "approaches", "future directions", "architecture", "perspectives", "objectives", "aims", "motivations", "problem statement", "research problem", "goals",
        "Technical specifications", "Specifications", "State-of-the-art", "Problem-setup", "Pre-training", "Limitations", "Materials", "discussions", "limitations", "limitation",
        "experimental setup", "analysis", "approximate methods", "evaluations", "Broader impacts", "impact", "impacts", "procedure", "ablations", "ablation study", "Model", "Dataset",
        "objectives", "objective", "details", "evaluation tasks", "data construction", "inference", "main results", "field architecture", "implementation study", "setup",
        "experiment settings", "design recipes", "evaluations", "training principles", "method and data collection", "summary statistics", "conclusions, limitations, and discussion",
        "limitations and future works", "limitation and future work", "conclusion and discussion"
    ]
    md_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.md')]

    if not md_files:
        print(f"No .MD files found in {input_dir}")
        return

    else:
        # Process each Markdown file
        for md_file in md_files:
            # Construct full input path
            input_path = os.path.join(input_dir, md_file)

            try:
                # Extract sections
                extracted_content = extract_sections(input_path, sections_to_extract)

                # Save to file
                output_path = os.path.join(output_dir, f"{Path(md_file).stem}_extracted.md")
                with open(output_path, 'w', encoding='utf-8') as file:
                    file.write(extracted_content)

                print(f"Successfully extracted sections to {output_path}")

            except Exception as e:
                print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()

