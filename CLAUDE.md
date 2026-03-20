# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Financial NLP pipeline for sentiment analysis of SEC 10-K/10-Q filings. The project extracts Management Discussion & Analysis (MD&A) sections from SEC filings and applies FinBERT-based sentiment classification.

## Setup

```bash
pip install -e .
```

Dependencies: numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, torch, torchvision, transformers, datasets, accelerate, nbformat, ipykernel (Python >=3.11).

## Running the Code

All work is done in Jupyter notebooks — there are no standalone scripts.

1. **Data Extraction** — `Data Preparation/mda_extraction.ipynb`
   - Requires WRDS credentials (PostgreSQL connection to `wrds-pgdata.wharton.upenn.edu:9737`)
   - Reads company list from `Data Preparation/TechCompanyByMarketCap_withCIK.csv`
   - Downloads 10-K and 10-Q filings to `Filings/` (gitignored), extracts MD&A sections to `MDA/`

2. **Sentiment Analysis** — `SentimentAnalysis/finbert.ipynb`
   - Self-contained (no credentials needed)
   - Fine-tunes `ProsusAI/finbert` on a synthetic 200-sample dataset
   - Compares against Loughran-McDonald lexicon + TF-IDF + SVM baseline

## Architecture

### Data Flow

```
WRDS SEC Database → Filings/ (raw HTML) → MDA/ (18k+ cleaned .txt files) → FinBERT fine-tuning
```

### Key Design Decisions

- **MD&A extraction** uses regex patterns matching `Item 2`/`Item 7` headers as section boundaries. Start patterns match "Management Discussion and Analysis" variants; end patterns match subsequent items (7A, 8, 3, 4). Minimum section length is 5,000 characters to filter partial extractions.
- **Text cleaning** strips HTML/XBRL tags and normalizes whitespace after extraction.
- **MDA file naming**: `{CompanyName}_{FormType}_{Date}_MDA.txt` (e.g., `Apple_10-K_2013-10-30_MDA.txt`)
- **Synthetic training data**: 200 samples generated from templates across 10 tickers × 12 quarters (2021–2023), split time-stratified 80/20 to avoid look-ahead bias.
- **`Filings/`** directory is gitignored (raw SEC filing downloads); `MDA/_extraction_summary.csv` is also gitignored.

### Model Details

| Model | Approach | Macro F1 |
|-------|----------|----------|
| FinBERT (fine-tuned) | `ProsusAI/finbert`, 3 classes, max_len=128, batch=16, epochs=3, LR=2e-5 | 0.8804 |
| Baseline | Loughran-McDonald dict + TF-IDF (5k features, 1-2gram) + LinearSVC | 0.7593 |
