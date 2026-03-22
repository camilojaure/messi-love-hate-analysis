# Messi Love & Hate Analysis

NLP analysis (2019) of ~395,000 Instagram comments on Lionel Messi to investigate whether a societal *grieta* (polarization) existed around his public image in Argentina, following the 2018 FIFA World Cup. Commissioned by a brand sponsor evaluating reputational risk of associating with Messi's image.

## Research Question

Does a polarization exist in the public discourse about Messi in Argentina? Specifically:
1. Do two distinct topics emerge — one supportive, one critical?
2. Is there a meaningful balance between positive and negative sentiment?

## Key Finding

**No polarization detected.** Both NMF topic clusters are positive: one dominated by affectionate English-language greetings ("love", "king", "goat"), the other by Argentine national team references ("vamos argentina", "mundial"). Positive sentiment outweighs negative ~30:1 (54K vs 1.7K comments containing sentiment-coded words).

## Dataset

- ~395,000 Instagram comments scraped from Messi's official and fan accounts (2018–2019)
- Raw data is not included in this repo (too large)

## Pipeline

```
data/raw/messi_comments.csv
    ↓ scripts/clean_messi_data.py
    Language detection · Stopword removal · Stemming
data/processed/messi_comments_cleaned.csv
    ↓ notebooks/1.messi_eda_topics.ipynb
    TF-IDF · NMF (2 topics) · Word clouds
    ↓ notebooks/2.sentiment_analysis_w2v.ipynb
    Word2Vec · Cosine similarity · MDS visualization
results/
```

## Methodology

| Step | Method | Key params |
|------|--------|------------|
| Preprocessing | SnowballStemmer + langdetect | Parallel, seed=0 |
| Topic modeling | NMF | `n_components=2`, 46 TF-IDF features |
| Sentiment | Word2Vec skip-gram | 20 dims, window=7 |
| Visualization | MDS on cosine distances | 31×31 word distance matrix |

## Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
```

Run notebooks in order: `1.messi_eda_topics.ipynb` → `2.sentiment_analysis_w2v.ipynb`

Place raw data at `data/raw/messi_comments.csv` before running `scripts/clean_messi_data.py`.
