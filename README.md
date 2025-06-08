# 📄 Knowledge Compression via Question Generation: Enhancing Multihop Document Retrieval without Fine-tuning

[![Under Review](https://img.shields.io/badge/Status-Under%20Review-yellow)](https://github.com/yourusername/your-repo)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains the official implementation of our paper on knowledge compression through question generation for enhanced multihop document retrieval.

**Authors:** Anvi Alex Eponon, Moein Shahiki-Tash, Ildar Batyrshin, Christian E. Maldonado-Sifuentes, Grigori Sidorov, Alexander Gelbukh

---

## 🧠 Abstract

This study presents a question-based knowledge encoding approach that enhances the performance of Large Language Models in Retrieval-Augmented Generation (RAG) systems without requiring model fine-tuning or traditional chunking strategies. We encode textual content through generated questions that span the information space lexically and semantically, creating targeted retrieval cues paired with a custom syntactic reranking method.

Our approach achieves **0.84 Recall@3** on single-hop retrieval tasks using 109 scientific papers from Natural Language Processing, outperforming traditional chunking methods by **60%**. We introduce "paper-cards"—scientific paper summaries under 300 characters that improve BM25 semantic retrieval performance, increasing Mean Reciprocal Rank from 0.56 to **0.85 at MRR@3**. For multihop tasks, our syntactic reranking method achieves **0.52 F1-score** with LLaMA2-Chat-7B on the LongBench QA v1 2WikiMultihopQA dataset.

---

## 🚀 Key Contributions

- **Novel Knowledge Compression**: A question-based encoding method for RAG systems that eliminates the need for traditional chunking
- **Superior Performance**: Achieves state-of-the-art retrieval performance without model fine-tuning
- **Paper-Cards Innovation**: Ultra-compact scientific paper summaries (≤300 characters) for enhanced semantic retrieval
- **Syntactic Reranker**: Custom reranking algorithm optimized for multihop question answering

---

## 📊 Results Summary

| Method | Dataset | Metric | Score |
|--------|---------|---------|-------|
| Our Approach | NLP Papers (109) | Recall@3 | **0.84** |
| Our Approach | 2WikiMultihopQA | F1-Score | **0.52** |
| Paper-Cards + BM25 | Query Retrieval | MRR@3 | **0.85** |

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/anvix9/llama2-chat.git
cd llama2-chat

# Install dependencies
pip install -r requirements.txt

# Install Ollama (for model evaluation)
curl -fsSL https://ollama.ai/install.sh | sh
```

## 📂 Repository Structure

```
├── main.py                 # Main workflow script and result generation
├── ra.py                   # Evaluation script for generated files
├── requirements.txt        # Python dependencies
├── data/                   # Dataset directory
├── results/               # Generated results and outputs
└── exps_synt/            # Multihop evaluation experiments
    ├── bm25_code.py      # BM25 response generation (paper-cards, abstracts, full papers)
    ├── experiments.py    # Main evaluation script with Ollama integration
    ├── eval_e.py         # Results evaluation and plot generation
    └── eval2.py          # CSV generation for evaluation metrics
```

---

## 🚀 Quick Start

Each folder has its requirements.txt that need to be installed first.

### Single-hop Retrieval
```bash
# Run main workflow
python main.py

# Evaluate results
python ra.py
```

### Multihop Evaluation
```bash
cd exps_synt

# Generate responses using different methods
python bm25_code.py --method paper-cards
python bm25_code.py --method abstract
python bm25_code.py --method full-paper

# Run experiments with Ollama models
python experiments.py

# Evaluate and generate plots
python eval2.py  # Generates CSV
python eval_e.py # Creates visualization plots
```

---

## 📈 Evaluation Metrics

- **Accuracy@K**: Proportion of relevant documents retrieved in top-K results
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of first relevant result
- **F1-Score**: Harmonic mean of precision and recall for multihop QA

---

## 🔬 Experimental Setup

- **Single-hop Dataset**: 109 randomly selected NLP research papers
- **Multihop Dataset**: LongBench QA v1 2WikiMultihopQA
- **Base Models**: LLaMA2-Chat-7B, Vicuna-7B, evaluated via Ollama
- **Baseline**: Traditional chunking methods and context compression with fine-tuning

---

## 📝 Citation

```bibtex
@article{eponon2024knowledge,
  title={Knowledge Compression via Question Generation: Enhancing Multihop Document Retrieval without Fine-tuning},
  author={Eponon, Anvi Alex and Shahiki-Tash, Moein and Batyrshin, Ildar and Maldonado-Sifuentes, Christian E. and Sidorov, Grigori and Gelbukh, Alexander},
  journal={Under Review},
  year={2024}
}
```

---

## 📧 Contact

**Anvi Alex Eponon** - [epononanvi@gmail.com](mailto:epononanvi@gmail.com)

For questions about the research or implementation, feel free to open an issue or contact the authors directly.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE]([LICENSE](https://choosealicense.com/licenses/mit/)) file for details.

---
