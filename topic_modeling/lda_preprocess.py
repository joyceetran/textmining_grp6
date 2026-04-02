import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from spacy.language import Language
from spacy.schemas import ConfigSchemaNlp
ConfigSchemaNlp.model_rebuild()
nlp = spacy.load('en_core_web_lg')
from sklearn.decomposition import LatentDirichletAllocation


# ── Stop words ──────────────────────────────────────────────────────
FINANCE_STOPWORDS = {
    "herein", "thereof", "thereto", "therein", "hereby", "pursuant",
    "accordance", "aforementioned", "foregoing", "whereas", "whereby",
    "hereafter", "hereinafter",
    "form", "quarterly", "annual", "report", "filing", "fiscal",
    "quarter", "period", "ended", "condensed", "consolidated",
    "statements", "notes", "refer", "also", "following", "described",
    "set", "forth", "part", "item",
    "incorporated", "reincorporated", "headquartered", "mean", "means",
    "including", "included", "related", "thereto", "million", "thousand", "billion",
    "company", "corporation", "subsidiaries", "inc", "ltd", "limited",
    "platforms", "manufacturing",
    # generic financial terms that appear everywhere
    "year", "quarter", "net", "total", "company", "income", "sale",
    "stock", "share", "value", "loss", "cost", "rate", "increase",
    "decrease", "high", "low", "new", "old", "use", "provide",
    "result", "period", "fiscal", "annual", "month", "date",
    "future", "current", "prior", "continue", "compare", "impact",
    "significant", "general", "certain", "include", "consist",
    # SEC filing boilerplate
    "report", "form", "filing", "sec", "act", "note", "item",
    "statement", "financial", "accounting", "disclosure", "exhibit",
    "registrant", "amendment", "pursuant", "herein", "thereof",
    "whereas", "whereas", "thereto",
    # numbers/time that slipped through
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    # spaCy lemmatization artifacts
    "datum",   # "data" → "datum"
    "agendum", # "agenda" → "agendum"
}

GEO_STOPWORDS = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas",
    "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming",

    "ca", "de", "tx", "ny", "wa", "nv", "fl", "ga", "il", "ma",
    "san francisco", "san jose", "santa clara", "santa", "clara",
    "palo alto", "menlo park", "mountain view", "sunnyvale", "cupertino",
    "redwood city", "fremont", "san diego", "los angeles", "seattle",
    "portland", "austin", "boston", "new york city", "new york",
    "chicago", "denver", "atlanta", "miami",
    "delaware", "nevada", "cayman islands", "bermuda", "ireland",
    "luxembourg", "singapore", "hong kong", "switzerland",
    "united states", "united kingdom", "china", "japan", "germany",
    "france", "india", "taiwan", "south korea", "israel", "canada",
    "australia", "netherlands", "sweden", "finland",
    "north america", "south america", "latin america", "europe",
    "asia pacific", "middle east", "apac", "emea", "americas",
}


ALL_STOP_WORDS = FINANCE_STOPWORDS | GEO_STOPWORDS

# Fix: use ALL STOP WORDS consistently in both places
nlp.Defaults.stop_words |= ALL_STOP_WORDS
for word in ALL_STOP_WORDS:        
    nlp.vocab[word].is_stop = True

# tokenise and lemmatise 
def preprocess_docs(texts: list[str], batch_size: int = 500) -> list[str]:
    """
    - Filters out: punctuation, whitespace, stop words, short tokens (<3 chars).
    - Returns one lemmatized string per document.
    """
    processed = []
 
    for doc in nlp.pipe(texts, disable=["parser", "ner"], batch_size=batch_size):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_space         # drop whitespace tokens
            and not token.is_stop          # drop stop words
            and len(token.lemma_) >= 3     # drop very short tokens
            and token.lemma_.isalpha()     # keep alphabetic tokens only
        ]
        processed.append(" ".join(tokens))
 
    return processed

# vectorise 
def build_count_matrix(processed_docs: list[str]):
    """
    Convert lemmatized documents into a sparse document-term count matrix.
    """
    vectorizer = CountVectorizer(
        min_df=10,                # token must appear in at least 10 documents -- remove rare words
        max_df=0.90,             # ignore tokens in >90% of documents -- remove too common words
        max_features=10_000,     # cap vocabulary size (capped at 10000 features)
        ngram_range=(1, 2),      # allow bi grams
    )
 
    dtm = vectorizer.fit_transform(processed_docs)
    return dtm, vectorizer

def docs2both(docs):
    docs['processed_text'] = preprocess_docs(docs['clean_text'].tolist())
    dtm, vectorizer = build_count_matrix(docs['processed_text'])
    vectors = [
        [(term_id, count) for term_id, count in zip(row.indices, row.data)]
        for row in dtm
    ]

    ## dtm for sk-learn model 
    ## vectors for gensim model 
    ## vectorizer is to map term id to word
    return dtm, vectors, vectorizer


