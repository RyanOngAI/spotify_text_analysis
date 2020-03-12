import collections
import pandas as pd

from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

def cluster_sentences(sentences, nb_of_clusters=5):
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'),
                                    max_df=0.9,
                                    min_df=0.1,
                                    lowercase=True)
    # Constructing a tf-idf matrix for the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    kmeans = KMeans(n_clusters=nb_of_clusters)
    kmeans.fit(tfidf_matrix)
    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
    return dict(clusters)

if "__name__" == "__main__":
    # Read dataset
    clustered_dataset = pd.read_csv('clustered_data.csv')

    # Clustering sentences based on text similarity (measured by tfidf)
    clustered_sentences = cluster_sentences(clustered_dataset['message_body_processed'])

    # Collecting clustering results (REFACTOR - inefficient)
    clustered_data = []
    for cluster in range(5):
        for i, sentence in enumerate(clustered_sentences[cluster]):
            clustered_data.append((cluster, clustered_dataset['message_body_processed'][sentence]))
    text_similarity_clustered_df = pd.DataFrame(clustered_data, columns = ['tfidf_cluster', 'message_body_processed'])

    # Merging clustering results with original dataset
    final_dataset = pd.merge(clustered_dataset, text_similarity_clustered_df, on = 'message_body_processed')
    # final_dataset.to_csv('complete_clustering.csv', index = False)