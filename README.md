# Training-Free Sentence-Level Context Pruning for Retrieval-Augmented Generation (TF-SCP)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset: SQuAD v2](https://img.shields.io/badge/Dataset-SQuAD%20v2-blue)](https://huggingface.co/datasets/rajpurkar/squad_v2)

Official implementation, empirical results ($n = 1,000$), visualization scripts, and IEEE conference manuscript (`main.tex`) for **Training-Free Sentence-Level Context Pruning for Retrieval-Augmented Generation using Dual Sparse-Dense Scoring (TF-SCP)**.

---

## Overview

In Retrieval-Augmented Generation (RAG), first-stage retrievers frequently inject topically related distractor passages alongside answer-bearing evidence. On compact or CPU-bound language models, extraneous tokens inflate prefill latency and degrade span extraction via distractor interference and the *"Lost in the Middle"* effect.

**TF-SCP** acts as a lightweight, zero-shot middleware filter between the retriever and the reader LLM—requiring **no training**, **no auxiliary causal LM perplexity pass**, and preserving complete sentence grammar:

1. **Sentence Segmentation:** Partitions retrieved documents into ordered sentences using `spaCy` (`en_core_web_sm`), tracking parent document boundaries.
2. **Dual Sparse–Dense Scoring:** Scores each sentence against the query using both:
   - **Sparse Lexical Relevance ($\hat{s}_{\text{lex}}$):** Query-local `BM25Okapi` over stopword-filtered tokens.
   - **Dense Semantic Similarity ($\hat{s}_{\text{sem}}$):** Cosine similarity over $\ell_2$-normalized embeddings from `sentence-transformers/all-MiniLM-L6-v2`.
   Both signals are min-max normalized within each query's candidate pool.
3. **Convex Score Fusion:**
   $$F(s_j) = \alpha \cdot \hat{s}_{\text{lex}}(s_j) + (1 - \alpha) \cdot \hat{s}_{\text{sem}}(s_j), \qquad \alpha \in [0, 1]$$
4. **Dynamic Moment-Based Thresholding:** Retains sentences satisfying $F(s_j) \ge \tau$, where
   $$\tau = \mu_F + \beta \cdot \sigma_F$$
   with a minimum floor of $k_{\min} = 1$ sentence, re-assembled in original document order.

---

## Key Empirical Findings ($n = 1,000$ SQuAD v2)

Evaluated on 1,000 answerable queries from SQuAD v2 validation paired with 2 mined corpus-wide BM25 hard-negative paragraphs per query (mean candidate pool: 16.3 sentences / 524.5 tokens) using `Qwen/Qwen2.5-0.5B-Instruct` (FP32, greedy decoding) on an Apple M3 CPU (8 GB RAM):

- **Best Accuracy & Compression ($\alpha = 0.5, \beta = +0.5$):** Achieves **74.25% token reduction** (524.5 $\rightarrow$ 133.9 tokens), reduces end-to-end CPU latency by **30.1%** (807.6 $\rightarrow$ 564.3 ms), and yields Bonferroni-significant gains ($\alpha_{\text{FWER}} = 0.05/7 \approx 0.0071$) of **+6.67 F1** ($p = 0.0001$) and **+4.80 EM** (McNemar $p = 0.0010$) over uncompressed RAG while maintaining **94.5%** answer retention.
- **Recall Safeguard ($\alpha = 0.5, \beta = 0.0$):** Eliminates **55.79%** of context tokens and cuts end-to-end latency by **21.6%** (807.6 $\rightarrow$ 633.2 ms) while retaining **97.5%** of gold answers—rescuing **73.3%** (44/60) of genuine lexical-only answer losses (McNemar $p = 2.6 \times 10^{-12}$) and improving F1 by **+2.73** ($p = 0.025$).
- **Lexical-Only Baseline ($\alpha = 1.0, \beta = 0.0$):** Compresses context by **63.71%** with **13.75 ms** pruning latency and **32.5%** end-to-end speedup, but drops answer retention to **93.2%**.

| Configuration | Token Red. (%) | Answer Retained (%) | EM | F1 ($\Delta$ vs. Full, $p$) | ROUGE-L | E2E Latency (ms) |
|---|---|---|---|---|---|---|
| Standard RAG (Full Context) | 0.00 | 99.2 | 38.20 | 50.94 (—) | 50.89 | 807.6 |
| BM25-Only ($\alpha = 1.0, \beta = 0.0$) | 63.71 | 93.2 | 39.10 | 53.54 ($+2.60$, $p = 0.046$) | 53.42 | 544.8 |
| TF-SCP ($\alpha = 0.5, \beta = -0.5$) | 31.20 | 98.8 | 40.00 | 52.70 ($+1.76$, $p = 0.096$) | 52.63 | 730.4 |
| TF-SCP ($\alpha = 0.5, \beta = 0.0$) | 55.79 | 97.5 | 40.40 | 53.67 ($+2.73$, $p = 0.025$) | 53.59 | 633.2 |
| TF-SCP ($\alpha = 0.75, \beta = 0.0$) | 60.50 | 96.2 | 40.00 | 54.87 ($+3.93$, $p = 0.002$) | 54.73 | 600.6 |
| **TF-SCP ($\alpha = 0.5, \beta = +0.5$)** | **74.25** | **94.5** | **43.00** | **57.61 ($+6.67$, $p < 0.001$)** | **57.40** | **564.3** |

---

## Repository Structure

```text
.
├── tf_scp_experiment.py              # Core TF_SCP_Pruner implementation and SQuAD v2 evaluation harness
├── analyze_results.py                # Post-hoc significance tests, answer-loss decomposition, stage timings
├── plot_results.py                   # Publication-quality IEEE vector PDF figure generator
├── generate_plots.py                 # Identical copy of the plotting script
├── prefetch_assets.py                # Offline Hugging Face dataset and model pre-fetcher
├── requirements.txt                  # Core NLP / RAG / PyTorch dependencies
├── main.tex                          # Complete IEEE conference LaTeX manuscript
├── TF-SCP_Results_and_Discussion.md  # Full n=1000 terminal logs and Sections 5–6 narrative
├── f1_vs_compression.pdf             # Plot 1 (IEEE manuscript): F1 vs. Token Reduction
├── beta_sensitivity.pdf              # Plot 2 (IEEE manuscript): Answer Retention & Compression vs. Beta
├── tf_scp_compression_vs_f1.pdf      # Default output filename for Plot 1
├── tf_scp_threshold_sensitivity.pdf  # Default output filename for Plot 2
├── results_n1000/                    # Primary n=1000 outputs (summary.json, per_sample.jsonl, analysis.txt)
├── results_n100/                     # Pilot n=100 outputs
└── results_smoke/                    # Smoke-test (n=3) outputs
```

---

## Installation & Usage

### 1. Environment Setup

**Using `uv` (recommended):**
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install matplotlib seaborn  # Required only for plot_results.py
```

**Using standard `pip`:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install matplotlib seaborn     # Required only for plot_results.py
```

### 2. Pre-fetch Models & Dataset (Optional for Offline Execution)

```bash
HF_HOME=./.hf_cache python prefetch_assets.py Qwen/Qwen2.5-0.5B-Instruct
```

### 3. Run Experiments & Statistical Analysis

```bash
# Quick smoke test (n=3)
python tf_scp_experiment.py --n 3 --out results_smoke

# Full benchmark across all 8 (alpha, beta) configurations (n=1000)
python tf_scp_experiment.py --n 1000 --seed 42 --distractors 2 --out results_n1000

# Post-hoc paired significance, answer-loss decomposition, and per-stage latency
python analyze_results.py results_n1000
```

### 4. Generate Publication Figures (PDF)

```bash
python plot_results.py
```

---

## Citation

If you find this codebase or methodology useful in your research, please cite our IEEE paper:

```bibtex
@inproceedings{tfscp2026rag,
  title     = {Training-Free Sentence-Level Context Pruning for Retrieval-Augmented Generation using Dual Sparse-Dense Scoring},
  author    = {Anonymous Author(s)},
  booktitle = {Proceedings of the IEEE Conference},
  year      = {2026},
  note      = {Implementation and experimental artifacts available at \url{https://github.com/PhantomBeast2090/sentence-level-rag-pruner-tf-scp-rag-context-pruning}}
}
```

---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
