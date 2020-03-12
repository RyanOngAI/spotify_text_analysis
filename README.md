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

Output: Additional time columns and data sorted by dates. CSV file ```sorted_data.csv``` generated.

### B. text_processing.py