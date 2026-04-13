# Financial NLP: SEC MD&A Sentiment & Topic Analysis

A financial text mining pipeline that extracts **Management Discussion & Analysis (MD&A)** sections from SEC 10-K/10-Q filings, classifies sentence-level sentiment with a fine-tuned FinBERT model, and models topics with LDA and BERTopic. Results are visualised in an interactive Streamlit dashboard.

---

## Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Replication Guide](#replication-guide)
  - [0. Environment Setup](#0-environment-setup)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Sentiment Analysis](#2-sentiment-analysis)
  - [3. Topic Modeling](#3-topic-modeling)
  - [4. Web App](#4-web-app)
- [Model Results](#model-results)
- [Key Design Decisions](#key-design-decisions)

---

## Project Overview

| Stage | Notebooks / Scripts | Output |
|---|---|---|
| Data extraction | `data_preparation/data_preprocessing.ipynb` | `datasets/final/mda_shared_preprocessed.csv` |
| Sentiment labeling | `sentiment_analysis/labeling.ipynb` | `webapp/labeled_sentiment.parquet` |
| FinBERT fine-tuning | `sentiment_analysis/finbert_finetuned.ipynb` | Sentiment predictions per sentence |
| Baseline models | `sentiment_analysis/rule-based.ipynb`, `naivebayes.ipynb` | Comparison metrics |
| Topic modeling (LDA) | `topic_modeling/lda.ipynb` | Topic-document distributions |
| Topic modeling (BERTopic) | `topic_modeling/bertopic.ipynb` | `webapp/topic_chunk_df.parquet` |
| Dashboard data build | `webapp/final_data.ipynb` | `webapp/final_df.parquet` |
| Dashboard | `webapp/app.py` | Streamlit app |

---

## Repository Structure

```
textmining_grp6/
├── data_preparation/
│   ├── data_preprocessing.ipynb        # Main cleaning + sentence segmentation pipeline
│   ├── sentiment_analysis_preprocessing.ipynb
│   ├── Text_Cleaning.ipynb
│   └── visu.ipynb
├── datasets/
│   ├── final/
│   │   ├── mda_shared_preprocessed.csv          # 452 k sentences, 473 companies, 2010–2025
│   │   ├── TechCompanyByMarketCap_withCIK.csv   # Company list with SEC CIK numbers
│   │   └── Loughran-McDonald_MasterDictionary_1993-2024.csv
│   ├── interim/                        # Labeling artefacts (gold samples, LLM labels)
│   └── raw_mda/                        # 20 487 plain-text MD&A files (one per filing)
├── sentiment_analysis/
│   ├── finbert_finetuned.ipynb         # FinBERT fine-tuning (Apple MPS / CUDA)
│   ├── labeling.ipynb                  # Manual 400-sample + LLM few-shot labeling
│   ├── rule-based.ipynb                # Loughran-McDonald lexicon baseline
│   └── naivebayes.ipynb                # TF-IDF + Naive Bayes baseline
├── topic_modeling/
│   ├── bertopic.ipynb                  # BERTopic (all-MiniLM-L6-v2 + HDBSCAN)
│   ├── lda.ipynb                       # Gensim LDA with coherence/perplexity grid search
│   ├── LDA_Model2_sklearn.ipynb        # sklearn LDA variant
│   └── lda_preprocess.py               # Shared tokenisation / lemmatisation helpers
├── webapp/
│   ├── app.py                          # Streamlit dashboard
│   ├── final_data.ipynb                # Merges sentiment + topic outputs → final_df.parquet
│   ├── final_df.parquet                # Pre-built dashboard data (ready to use)
│   ├── labeled_sentiment.parquet       # Sentence-level FinBERT predictions
│   ├── topic_chunk_df.parquet          # BERTopic chunk-level topic assignments
│   └── requirements.txt               # Minimal webapp dependencies
├── pyproject.toml                      # Full project dependencies (uv / pip)
└── requirements.txt
```

---

## Dataset

- **Source:** SEC EDGAR via WRDS (Wharton Research Data Services)
- **Scope:** 473 publicly listed technology companies by market cap, 10-K and 10-Q filings, 2010–2025
- **Scale:** 17 560 filings → 452 390 sentences after preprocessing
- **Company list:** `datasets/final/TechCompanyByMarketCap_withCIK.csv` — contains company name, ticker, and SEC CIK number

Raw MD&A files follow the naming convention `{Company}_{FormType}_{YYYY-MM-DD}_MDA.txt` (e.g. `NVIDIA_10-K_2023-01-26_MDA.txt`) and are stored in `datasets/raw_mda/`.

---

## Replication Guide

### 0. Environment Setup

**Python ≥ 3.11 is required.**

```bash
# Clone the repo
git clone <repo-url>
cd textmining_grp6

# Option A — uv (recommended, fast)
pip install uv
uv sync

# Option B — pip
pip install -e .

# For the web app only (lighter install)
pip install -r webapp/requirements.txt
```

> **Apple Silicon (MPS):** All notebooks run on CPU/MPS. CUDA is not required.
> The BERTopic notebook contains a CUDA assertion that must be commented out when running locally — see the note in [Step 3](#3-topic-modeling).

---

### 1. Data Preparation

**Notebook:** [data_preparation/data_preprocessing.ipynb](data_preparation/data_preprocessing.ipynb)

This notebook reads all `.txt` files from `datasets/raw_mda/`, parses the structured filename metadata, applies text cleaning, and segments each MD&A into individual sentences.

**Inputs**

| File | Description |
|---|---|
| `datasets/raw_mda/*.txt` | Raw MD&A plain-text files (already in the repo) |
| `datasets/final/TechCompanyByMarketCap_withCIK.csv` | Ticker and CIK reference |

**What it does**

1. Parses `{Company}_{FormType}_{YYYY-MM-DD}_MDA.txt` filenames into structured columns
2. Joins ticker symbols from the company reference file
3. Cleans residual artefacts: MD&A headers, encoding replacement characters (`\ufffd`), boilerplate
4. Segments text into sentences with spaCy (`en_core_web_sm`)
5. Replaces numeric tokens with `NUM` for normalisation

**Output**

`datasets/final/mda_shared_preprocessed.csv` — sentence-level dataframe with columns:

| Column | Description |
|---|---|
| `doc_id` | `{Company}_{FormType}_{Date}` — unique filing identifier |
| `company_name` | Company name as in the filename |
| `filing_type` | `10-K` or `10-Q` |
| `filing_date` | `YYYY-MM-DD` |
| `year` | Integer year |
| `quarter` | `Q1`–`Q4` |
| `sentence` | Cleaned sentence text |
| `sentiment` | Placeholder column (filled by labeling step) |

> **Skipping extraction from WRDS:** The raw MD&A files are already committed to `datasets/raw_mda/` (20 487 files). You do **not** need WRDS credentials to run the downstream steps.

---

### 2. Sentiment Analysis

#### 2a. Labeling

**Notebook:** [sentiment_analysis/labeling.ipynb](sentiment_analysis/labeling.ipynb)

Produces ground-truth labels used to fine-tune and evaluate FinBERT.

1. Loads `datasets/final/mda_shared_preprocessed.csv`
2. Deduplicates sentences and draws a 400-sentence stratified sample for **manual annotation** — output in `sentiment_analysis/manual_labeling_400_labeled.xlsx`
3. Draws a 50 k-sentence sample for **LLM few-shot labeling** (positive / negative / neutral) using the manually labeled sentences as in-context examples
4. Saves merged labels to `webapp/labeled_sentiment.parquet`

#### 2b. FinBERT Fine-Tuning

**Notebook:** [sentiment_analysis/finbert_finetuned.ipynb](sentiment_analysis/finbert_finetuned.ipynb)

Fine-tunes `ProsusAI/finbert` (BERT-base, 3-class) on the labeled MD&A sentences.

**Key hyperparameters**

| Parameter | Value |
|---|---|
| Base model | `ProsusAI/finbert` |
| Max sequence length | 128 tokens |
| Batch size | 16 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Train / test split | filings ≤ 2022 → train, 2023+ → test (time-stratified) |

**Inputs**

- `datasets/final/mda_shared_preprocessed.csv`
- `webapp/labeled_sentiment.parquet` (from labeling step)

The notebook saves per-sentence sentiment probabilities (positive / negative / neutral) back to `webapp/labeled_sentiment.parquet`.

#### 2c. Baseline Models

| Notebook | Model |
|---|---|
| [sentiment_analysis/rule-based.ipynb](sentiment_analysis/rule-based.ipynb) | Loughran-McDonald financial lexicon (word-count scoring) |
| [sentiment_analysis/naivebayes.ipynb](sentiment_analysis/naivebayes.ipynb) | TF-IDF (5 k features, 1–2gram) + Multinomial Naive Bayes |

Both baselines use the same labeled sample and the same time-stratified split for a fair comparison.

---

### 3. Topic Modeling

#### 3a. LDA

**Notebook:** [topic_modeling/lda.ipynb](topic_modeling/lda.ipynb)

1. Loads `datasets/final/mda_shared_preprocessed.csv`
2. Aggregates sentences into one document per filing
3. Runs `lda_preprocess.docs2both()` (spaCy lemmatisation → Gensim BoW + sklearn DTM)
4. Trains a Gensim LDA model and performs a grid search over topic count using coherence (C_v) and perplexity
5. Saves topic-document distributions to `topic_modeling/topic_dist_df.parquet`

#### 3b. BERTopic

**Notebook:** [topic_modeling/bertopic.ipynb](topic_modeling/bertopic.ipynb)

Neural topic modeling with sentence embeddings, UMAP dimensionality reduction, and HDBSCAN clustering.

> **Apple Silicon note:** The notebook begins with `assert torch.cuda.is_available()`. Comment this line out and set `DEVICE = "mps"` (or `"cpu"`) before running locally.

**Pipeline**

1. Loads `datasets/final/mda_shared_preprocessed.csv`
2. Groups consecutive sentences into 5-sentence chunks
3. Encodes chunks with `all-MiniLM-L6-v2` (SentenceTransformers)
4. Reduces to 5-D with UMAP (PCA warm-start on CPU for speed)
5. Clusters with HDBSCAN and fits BERTopic to extract topic labels
6. Saves chunk-level topic assignments to `webapp/topic_chunk_df.parquet`

---

### 4. Web App

#### 4a. Build the dashboard data

**Notebook:** [webapp/final_data.ipynb](webapp/final_data.ipynb)

Joins sentiment predictions (`labeled_sentiment.parquet`) with topic assignments (`topic_chunk_df.parquet`) and the sentence map (`sent_chunk_map.csv`) into a single flat file used by the Streamlit app.

```
webapp/
├── labeled_sentiment.parquet   (from Step 2b)
├── topic_chunk_df.parquet      (from Step 3b)
└── sent_chunk_map.csv          (chunk → sentence mapping, produced by bertopic.ipynb)
```

Run all cells → `webapp/final_df.parquet` is written.

> `webapp/final_df.parquet` is already committed to the repo if you want to skip directly to the dashboard.

#### 4b. Launch the dashboard

```bash
# From repo root
streamlit run webapp/app.py
```

The app reads `webapp/final_df.parquet` and requires only the lightweight dependencies in `webapp/requirements.txt`.

**Dashboard tabs**

| Tab | What it shows |
|---|---|
| Snapshot | Latest-quarter sentiment heatmap and topic breakdown per company |
| Sentiment Trends | Positive / negative / neutral score timelines, company comparison |
| Topic Trends | Topic share over time, filterable by company and topic label |
| Filing Explorer | Per-filing sentence-level sentiment browser |

---

## Model Results

| Model | Approach | Macro F1 |
|---|---|---|
| FinBERT (fine-tuned) | `ProsusAI/finbert`, 3-class, time-stratified split | **0.88** |
| Loughran-McDonald | Financial lexicon word-count scoring | — |
| TF-IDF + Naive Bayes | 5 k features, 1–2gram | — |
| TF-IDF + LinearSVC | 5 k features, 1–2gram | 0.76 |

---

## Key Design Decisions

- **MD&A boundary detection:** Regex patterns matching `Item 2` / `Item 7` headers as section start/end markers. Minimum section length of 5 000 characters filters partial extractions.
- **Time-stratified split:** Training on filings up to 2022 and evaluating on 2023+ prevents look-ahead bias — the model never sees future language drift during training.
- **Sentence chunking for BERTopic:** Grouping 5 consecutive sentences into chunks gives each unit enough context for meaningful embeddings while keeping the total document count manageable.
- **UMAP warm-start (CPU):** PCA initialisation of UMAP reduces runtime by 3–5× on Apple MPS machines compared to random initialisation.
- **Numeric normalisation:** All numeric tokens are replaced with `NUM` during preprocessing to prevent the model from learning spurious correlations with specific figures.
