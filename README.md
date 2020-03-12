# Spotify Text Analysis
Applied sentiment analysis, topic modelling, and simple TFIDF text similarity to cluster Spotify messages together.

## 1. Setup - Download all the dependencies
```conda env create -f spotify_project.yml```

## 2. Files documentation
### A. eda.py
Complete a simple EDA on the spotify dataset. It returns the following:
- No. of missing data (per column)
- Frequencies of each message type
- No. unique authors (or at least author_id)
- Frequencies of each severity type

Also convert ```created_at``` to datetime object and created multiple time features column such as ```year```, ```month```, etc. for clustering messages through time.

**Output: Additional time columns and data sorted by dates. CSV file ```sorted_data.csv``` generated.**

### B. text_processing.py
Clean the Spotify messages by removing empty messages, duplications, and various common text processing such as lowercasing, removing stopwords, and lemmatisation. We also removed the Spotify handles and URLS.

**Output: CSV file ```processed_data.csv``` generated.**

### C. text_clustering.py
Applied out-of-the-box sentiment analysis and Gensim's topic modelling to the processed text data. Used VaderSentiment for sentiment analysis and written up a TopicModelling class for topic modelling.

**Output: CSV file ```clustered_data.csv``` generated. Also saved the dictionary ```dictionary.gensim```, bag-of-words corpus ```bow_corpus.pkl``` and lda model ```model.gensim```**

### D. topic_modelling_analysis.py
Complete a simple topic analysis to see which topic dominates the majority of Spotify text data.

**Output: Terminal output of topics analysis**

### E. text_similarity.py
Another method of clustering the Spotify messages by computing a TFIDF similarity matrix and uses KMeans to cluster sentences that are similar together.

**Output: CSV file ```complete_clustering.csv``` generated.**