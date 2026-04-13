"""
disagreement_analysis.py

Runs FinBERT, Naive Bayes (ComplementNB + TF-IDF + LM), and the rule-based
Loughran-McDonald classifier on a stratified sample of the held-out test set
(year >= 2023), then performs a structured error analysis.

Run from the repo root:
    python sentiment_analysis/disagreement_analysis.py
"""

import re
import textwrap
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import ComplementNB
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

DATA_PATH = "datasets/final/mda_shared_preprocessed.csv"
LM_PATH   = "datasets/final/Loughran-McDonald_MasterDictionary_1993-2024.csv"

LABELS = ["positive", "negative", "neutral"]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FUNCTIONS  (unchanged from notebooks)
# ══════════════════════════════════════════════════════════════════════════════

def load_lm_words(lm_path: str) -> tuple[set, set]:
    lm = pd.read_csv(lm_path)
    pos = set(lm.loc[lm["Positive"] > 0, "Word"].str.lower().str.strip())
    neg = set(lm.loc[lm["Negative"] > 0, "Word"].str.lower().str.strip())
    print(f"  LM dictionary: {len(pos):,} positive, {len(neg):,} negative words")
    return pos, neg


def lm_tokenize(text: str) -> list[str]:
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return [t for t in text.split() if t != "num"]


def _lm_score(text: str, pos_words: set, neg_words: set) -> tuple[int, int]:
    tokens = lm_tokenize(text)
    return sum(t in pos_words for t in tokens), sum(t in neg_words for t in tokens)


def rule_based_predict(texts: list[str], pos_words: set, neg_words: set) -> list[str]:
    preds = []
    for text in texts:
        p, n = _lm_score(text, pos_words, neg_words)
        preds.append("positive" if p > n else ("negative" if n > p else "neutral"))
    return preds


def _lm_feature_matrix(texts: list[str], pos_words: set, neg_words: set) -> csr_matrix:
    rows = [list(_lm_score(t, pos_words, neg_words)) for t in texts]
    return csr_matrix(np.array(rows, dtype=float))


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def train_nb(df: pd.DataFrame, pos_words: set, neg_words: set) -> tuple[ComplementNB, TfidfVectorizer]:
    train_df = df[df["year"] <= 2022].copy()
    train_df["clean"] = train_df["sentence"].apply(_clean)
    train_df = train_df.dropna(subset=["clean", "sentiment"]).drop_duplicates(subset=["clean", "sentiment"])
    X_text = train_df["clean"].astype(str).tolist()
    y = train_df["sentiment"].tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = hstack([vectorizer.fit_transform(X_text), _lm_feature_matrix(X_text, pos_words, neg_words)])
    model = ComplementNB()
    model.fit(X, y)
    print(f"  NB trained on {len(y):,} sentences (year ≤ 2022)")
    return model, vectorizer


def nb_predict(texts: list[str], model: ComplementNB, vectorizer: TfidfVectorizer,
               pos_words: set, neg_words: set) -> list[str]:
    cleaned = [_clean(t) for t in texts]
    X = hstack([vectorizer.transform(cleaned), _lm_feature_matrix(cleaned, pos_words, neg_words)])
    return model.predict(X).tolist()


_FINBERT_ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}


def load_finbert() -> tuple:
    name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE)
    mdl.eval()
    print(f"  FinBERT loaded → {DEVICE}")
    return tok, mdl


def finbert_predict(texts: list[str], tokenizer, model, batch_size: int = 32) -> list[str]:
    preds = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i+batch_size], truncation=True, max_length=128,
                        padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = model(**enc).logits.argmax(dim=-1).cpu().tolist()
        preds.extend(_FINBERT_ID2LABEL[idx] for idx in ids)
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# ERROR TAXONOMY
# Each misclassified sentence gets one or more tags that explain the likely
# root cause.  Tags are detected from surface features alone (no oracle needed).
# ══════════════════════════════════════════════════════════════════════════════

_NEGATORS    = {"not", "no", "never", "neither", "nor", "without", "absent",
                "lack", "lacking", "unlike", "failed", "inability", "unable"}
_CONCESSIONS = ["despite", "although", "even though", "while", "notwithstanding",
                "albeit", "regardless", "nevertheless", "however"]
_HEDGES      = ["cautiously", "somewhat", "modestly", "slightly", "partially",
                "marginally", "relatively", "broadly", "generally"]
_TEMPORAL    = ["compared to prior", "compared with prior", "versus prior",
                "prior year", "year-over-year", "year over year", "last year",
                "prior period", "compared to the same period"]


def tag_error(sentence: str, pos_words: set, neg_words: set) -> str:
    lower   = sentence.lower()
    tokens  = lm_tokenize(sentence)
    pos_cnt = sum(t in pos_words for t in tokens)
    neg_cnt = sum(t in neg_words for t in tokens)

    tags = []
    if any(t in _NEGATORS for t in tokens):
        tags.append("negation")
    if any(c in lower for c in _CONCESSIONS):
        tags.append("concession")
    if any(h in lower for h in _HEDGES):
        tags.append("hedging")
    if any(t in lower for t in _TEMPORAL):
        tags.append("temporal_contrast")
    if pos_cnt > 0 and neg_cnt > 0:
        tags.append("mixed_lm_signal")
    if pos_cnt == 0 and neg_cnt == 0:
        tags.append("no_lm_signal")

    return "|".join(tags) if tags else "other"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_WRAP = 90


def _hr(char="─", n=_WRAP):
    print(char * n)


def _section(title: str):
    print(f"\n{'═' * _WRAP}")
    print(f"  {title}")
    print('═' * _WRAP)


def _wrap(text: str, indent: int = 4) -> str:
    return textwrap.fill(text, width=_WRAP, initial_indent=" " * indent,
                         subsequent_indent=" " * indent)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(pool_per_class: int = 500, seed: int = 42) -> None:

    # ── 1. Load everything ────────────────────────────────────────────────────
    print("\n[1/3] Loading models and data …")
    pos_words, neg_words = load_lm_words(LM_PATH)
    df_full = pd.read_csv(DATA_PATH)
    nb_model, nb_vec = train_nb(df_full, pos_words, neg_words)
    fb_tok, fb_mdl   = load_finbert()

    # ── 2. Score stratified test sample ───────────────────────────────────────
    print(f"\n[2/3] Scoring test sample (year ≥ 2023, up to {pool_per_class} per class) …")
    test = df_full[df_full["year"] >= 2023].copy()
    sample = pd.concat(
        [grp.sample(min(len(grp), pool_per_class), random_state=seed)
         for _, grp in test.groupby("sentiment", group_keys=False)]
    ).reset_index(drop=True)
    print(f"  Sample size: {len(sample):,}  |  class distribution: "
          + "  ".join(f"{k}={v}" for k, v in sample["sentiment"].value_counts().items()))

    texts = sample["sentence"].tolist()
    gold  = sample["sentiment"].tolist()

    sample["finbert"]     = finbert_predict(texts, fb_tok, fb_mdl)
    sample["naive_bayes"] = nb_predict(texts, nb_model, nb_vec, pos_words, neg_words)
    sample["rule_based"]  = rule_based_predict(texts, pos_words, neg_words)
    sample["gold"]        = gold   # alias for clarity

    # Boolean wrong flags + error tag on every sentence
    for col in ("finbert", "naive_bayes", "rule_based"):
        sample[f"{col}_wrong"] = sample[col] != sample["gold"]

    sample["error_tag"] = sample["sentence"].apply(
        lambda s: tag_error(s, pos_words, neg_words)
    )

    # ── 3. Disagreement DataFrame ─────────────────────────────────────────────
    print("\n[3/3] Building disagreement table and running error analysis …")

    any_wrong = (
        sample["finbert_wrong"] | sample["naive_bayes_wrong"] | sample["rule_based_wrong"]
    )
    disagree = sample[any_wrong][
        ["sentence", "gold", "finbert", "naive_bayes", "rule_based", "error_tag"]
    ].reset_index(drop=True)

    _section("DISAGREEMENT TABLE  (any model ≠ gold)")
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 0)
    print(disagree.to_string(index=True))

    # ════════════════════════════════════════════════════════════════════════
    # ERROR ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    _section("ERROR ANALYSIS")

    total = len(sample)
    n_dis = len(disagree)

    # ── A. Overall accuracy ───────────────────────────────────────────────────
    print("\n── A. Overall accuracy on test sample ──────────────────────────────────────")
    print(f"  {'Model':<16}  {'Correct':>7}  {'Wrong':>7}  {'Accuracy':>9}  {'Macro F1':>9}")
    _hr()
    for col, label in [("finbert", "FinBERT"), ("naive_bayes", "Naive Bayes"), ("rule_based", "Rule-Based")]:
        correct = (~sample[f"{col}_wrong"]).sum()
        wrong   = sample[f"{col}_wrong"].sum()
        acc     = correct / total
        f1      = float(
            classification_report(gold, sample[col].tolist(), labels=LABELS,
                                  output_dict=True, zero_division=0)
            ["macro avg"]["f1-score"]
        )
        print(f"  {label:<16}  {correct:>7,}  {wrong:>7,}  {acc:>9.1%}  {f1:>9.4f}")
    print(f"\n  Total sentences: {total:,}  |  Any-model-wrong: {n_dis:,} ({n_dis/total:.1%})")

    # ── B. Per-class error breakdown ──────────────────────────────────────────
    print("\n── B. Per-class error rate (% of that gold class misclassified) ────────────")
    header = f"  {'Gold class':<12}" + "".join(f"  {'FinBERT':>10}  {'NaiveBayes':>10}  {'RuleBased':>10}")
    print(header)
    _hr()
    for lbl in LABELS:
        subset = sample[sample["gold"] == lbl]
        n = len(subset)
        if n == 0:
            continue
        vals = []
        for col in ("finbert", "naive_bayes", "rule_based"):
            err_rate = subset[f"{col}_wrong"].sum() / n
            vals.append(f"  {err_rate:>9.1%}  {subset[f'{col}_wrong'].sum():>4}/{n:<4}")
        print(f"  {lbl:<12}" + "".join(vals))

    print("\n  Interpretation:")
    for col, label in [("finbert", "FinBERT"), ("naive_bayes", "Naive Bayes"), ("rule_based", "Rule-Based")]:
        worst_class = max(LABELS, key=lambda lbl: (
            sample[sample["gold"] == lbl][f"{col}_wrong"].mean()
            if len(sample[sample["gold"] == lbl]) > 0 else 0
        ))
        worst_rate = sample[sample["gold"] == worst_class][f"{col}_wrong"].mean()
        print(f"  {label} struggles most with '{worst_class}' ({worst_rate:.1%} error rate).")

    # ── C. Confusion matrices ─────────────────────────────────────────────────
    print("\n── C. Confusion matrices (rows = gold, columns = predicted) ────────────────")
    for col, label in [("finbert", "FinBERT"), ("naive_bayes", "Naive Bayes"), ("rule_based", "Rule-Based")]:
        cm = confusion_matrix(gold, sample[col].tolist(), labels=LABELS)
        cm_df = pd.DataFrame(cm, index=[f"gold:{lbl}" for lbl in LABELS],
                             columns=[f"pred:{lbl}" for lbl in LABELS])
        print(f"\n  {label}")
        print(cm_df.to_string())

    # ── D. Error taxonomy ─────────────────────────────────────────────────────
    print("\n── D. Error taxonomy  (% of each model's errors falling into each category)")

    # Collect all errors per model with their tags
    tag_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    model_error_counts: dict[str, int] = {}
    for col, label in [("finbert", "FinBERT"), ("naive_bayes", "NaiveBayes"), ("rule_based", "RuleBased")]:
        errors = sample[sample[f"{col}_wrong"]]
        model_error_counts[label] = len(errors)
        for _, row in errors.iterrows():
            for tag in tag_error(row["sentence"], pos_words, neg_words).split("|"):
                tag_stats[tag][label] += 1

    all_tags = sorted(tag_stats.keys())
    model_labels = ["FinBERT", "NaiveBayes", "RuleBased"]

    print(f"\n  {'Error tag':<22}" + "".join(f"  {m:>14}" for m in model_labels))
    _hr()
    for tag in all_tags:
        vals = []
        for lbl in model_labels:
            cnt  = tag_stats[tag][lbl]
            total_errors = model_error_counts.get(lbl, 1)
            vals.append(f"  {cnt:>5} ({cnt/total_errors:>5.1%})")
        print(f"  {tag:<22}" + "".join(vals))
    print(f"\n  {'Total errors':<22}" + "".join(f"  {model_error_counts.get(m,0):>14,}" for m in model_labels))

    # ── E. Cross-model disagreement pattern ───────────────────────────────────
    print("\n── E. Cross-model disagreement patterns ────────────────────────────────────")
    patterns = {
        "All three wrong         ": lambda r: r.finbert_wrong and r.naive_bayes_wrong and r.rule_based_wrong,
        "Only FinBERT wrong      ": lambda r: r.finbert_wrong and not r.naive_bayes_wrong and not r.rule_based_wrong,
        "Only NaiveBayes wrong   ": lambda r: r.naive_bayes_wrong and not r.finbert_wrong and not r.rule_based_wrong,
        "Only RuleBased wrong    ": lambda r: r.rule_based_wrong and not r.finbert_wrong and not r.naive_bayes_wrong,
        "FinBERT + NB wrong      ": lambda r: r.finbert_wrong and r.naive_bayes_wrong and not r.rule_based_wrong,
        "FinBERT + RB wrong      ": lambda r: r.finbert_wrong and r.rule_based_wrong and not r.naive_bayes_wrong,
        "NB + RB wrong           ": lambda r: r.naive_bayes_wrong and r.rule_based_wrong and not r.finbert_wrong,
    }
    print(f"  {'Pattern':<28}  {'Count':>6}  {'% of sample':>12}  {'% of errors':>12}")
    _hr()
    for desc, fn in patterns.items():
        cnt = sum(fn(r) for r in sample.itertuples())
        print(f"  {desc}  {cnt:>6,}  {cnt/total:>11.1%}  {cnt/n_dis:>11.1%}")

    # ── F. Qualitative analysis with representative examples ─────────────────
    print("\n── F. Qualitative analysis: root causes with real examples ─────────────────")

    analysis_cases = [
        {
            "title": "F1. Negation  —  Rule-Based and NB blind spot",
            "explanation": (
                "Both the LM rule-based classifier and Naive Bayes treat text as a bag "
                "of words and therefore cannot resolve negation. When a negative LM word "
                "such as 'impairment', 'loss', or 'liability' is preceded by 'no', 'not', "
                "or 'did not', the actual meaning flips to positive, but both models still "
                "count the raw word and predict negative or neutral. FinBERT attends to "
                "the full token sequence and typically recovers the correct polarity."
            ),
            "filter": lambda r: (
                "negation" in tag_error(r["sentence"], pos_words, neg_words)
                and (r["naive_bayes_wrong"] or r["rule_based_wrong"])
            ),
        },
        {
            "title": "F2. Concessive framing  —  context beyond the dictionary",
            "explanation": (
                "'Despite X, we achieved Y' structures are systematically misclassified "
                "by the rule-based model because the concessive clause (X) often contains "
                "LM negative words ('challenging', 'difficult', 'adverse') that outweigh "
                "the main-clause positive words. NB may also be misled if the concessive "
                "pattern was underrepresented in the training corpus. FinBERT's attention "
                "mechanism allows it to weight the main clause outcome more heavily."
            ),
            "filter": lambda r: (
                "concession" in tag_error(r["sentence"], pos_words, neg_words)
                and (r["rule_based_wrong"] or r["finbert_wrong"])
            ),
        },
        {
            "title": "F3. No LM signal  —  Rule-Based defaults to neutral",
            "explanation": (
                "The Loughran-McDonald dictionary contains 347 positive and 2,345 negative "
                "words, a small fraction of everyday financial vocabulary. Sentences whose "
                "sentiment is carried by unlisted words ('expanded', 'contracted', "
                "'headwinds', 'standout') produce zero hits on both lists, forcing the "
                "rule-based classifier to output neutral regardless of true polarity. "
                "NB and FinBERT, which learned from surface statistics and contextual "
                "representations respectively, are less affected."
            ),
            "filter": lambda r: (
                "no_lm_signal" in tag_error(r["sentence"], pos_words, neg_words)
                and r["rule_based_wrong"]
            ),
        },
        {
            "title": "F4. Mixed LM signal  —  all models struggle",
            "explanation": (
                "Sentences that contain both LM positive and negative words in roughly "
                "equal measure present a hard signal-averaging problem. The rule-based "
                "classifier assigns the class with the higher word count, often landing "
                "incorrectly if the true label is neutral or if the dominant word belongs "
                "to the concessive clause. NB combines TF-IDF with LM counts, so it too "
                "can be pulled in the wrong direction. Even FinBERT struggles when genuine "
                "semantic balance is present (e.g. short-term cost vs long-term gain)."
            ),
            "filter": lambda r: (
                "mixed_lm_signal" in tag_error(r["sentence"], pos_words, neg_words)
                and r["finbert_wrong"] and r["naive_bayes_wrong"] and r["rule_based_wrong"]
            ),
        },
        {
            "title": "F5. Neutral-class bias  —  all models over-predict neutral",
            "explanation": (
                "83% of sentences in the dataset are labelled neutral. Both NB (trained "
                "on this distribution) and FinBERT (pre-trained on broadly neutral SEC "
                "filings) absorb this prior, causing them to err on the side of neutral "
                "when signals are weak. This explains most false-neutral predictions on "
                "mildly positive or mildly negative sentences and is visible in the "
                "per-class error breakdown above."
            ),
            "filter": lambda r: (
                r["gold"] != "neutral"
                and r["finbert"] == "neutral"
                and r["naive_bayes"] == "neutral"
            ),
        },
        {
            "title": "F6. Temporal contrast  —  relative improvement misread as negative",
            "explanation": (
                "Sentences that compare current performance with a prior period ('revenue "
                "declined 3% but beat our internal forecast', 'losses narrowed from $1.2B "
                "to $0.4B') contain surface negative words (declined, losses) alongside "
                "positive context (beat, narrowed). Bag-of-words models fire on the "
                "negative words; FinBERT can sometimes recover context but may still "
                "predict negative when the surface form is dominated by decline language."
            ),
            "filter": lambda r: (
                "temporal_contrast" in tag_error(r["sentence"], pos_words, neg_words)
                and (r["rule_based_wrong"] or r["naive_bayes_wrong"])
            ),
        },
    ]

    for case in analysis_cases:
        print(f"\n  {case['title']}")
        _hr("-")
        print(_wrap(case["explanation"]))
        # Pull up to 3 representative examples from the real scored data
        examples = [
            row for _, row in sample.iterrows()
            if case["filter"](row)
        ][:3]
        if examples:
            print(f"\n  Representative examples ({len(examples)} shown):")
            for row in examples:
                print(f"\n    Gold: {row['gold'].upper():<9} "
                      f"FinBERT: {row['finbert']:<9} "
                      f"NaiveBayes: {row['naive_bayes']:<9} "
                      f"RuleBased: {row['rule_based']}")
                wrapped = textwrap.fill(
                    row["sentence"], width=_WRAP - 4,
                    initial_indent="    ", subsequent_indent="    "
                )
                print(wrapped)
        else:
            print("    (no matching examples found in current pool — increase pool_per_class)")

    # ── G. Summary: architectural root causes ────────────────────────────────
    print("\n── G. Summary: architectural root causes ───────────────────────────────────")
    summary = [
        ("Rule-Based (LM)",
         "Dictionary lookup with no word-order awareness. Fails on: negation, "
         "concession, out-of-dictionary vocabulary (no_lm_signal), and temporal "
         "contrast. The LM dictionary is also heavily skewed toward negative words "
         "(2,345 neg vs 347 pos), creating a systematic negative bias on mixed sentences."),
        ("Naive Bayes (CNB)",
         "TF-IDF bag-of-words + LM counts. Inherits the LM bias and cannot resolve "
         "negation or word order. Additionally susceptible to training-distribution "
         "shift: unusual phrasing or domain evolution post-2022 reduces the reliability "
         "of n-gram weights. Strong neutral prior from class imbalance (83% neutral) "
         "increases false-neutral rate on ambiguous sentences."),
        ("FinBERT",
         "Contextual transformer with financial domain pre-training. Handles negation "
         "and concession better than bag-of-words models. Primary failure modes: "
         "(1) neutral-class overconfidence on weak signals due to the heavily neutral "
         "training distribution; (2) genuine semantic ambiguity (mixed signals) where "
         "even human annotators would disagree; (3) rare or highly technical accounting "
         "language not seen in pre-training."),
    ]
    for model_name, root_cause in summary:
        print(f"\n  {model_name}")
        print(_wrap(root_cause))

    # ── Export full sample with tags to Excel ────────────────────────────────
    out_path = "sentiment_analysis/disagreement_analysis.xlsx"
    sample[[
        "sentence", "gold",
        "finbert", "naive_bayes", "rule_based",
        "finbert_wrong", "naive_bayes_wrong", "rule_based_wrong",
        "error_tag",
    ]].to_excel(out_path, index=False)
    print(f"\n  Exported → {out_path}")

    print("\n" + "═" * _WRAP)
    print("  Done.")
    print("═" * _WRAP + "\n")


if __name__ == "__main__":
    main()
