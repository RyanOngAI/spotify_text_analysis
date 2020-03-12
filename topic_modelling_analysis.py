import pickle
import pandas as pd
from gensim import corpora, models
from text_clustering import TopicModelling

# cleaned_dataset = pd.read_csv('processed_data.csv')
# lda_model = TopicModelling(cleaned_dataset['message_body_processed'])
# saved_lda_model = models.ldamodel.LdaModel.load('model.gensim')
# corpus = pickle.load(open('bow_corpus.pkl', 'rb'))
# topic_categorisation = lda_model.topic_categorisation(saved_lda_model, corpus)

# Read dataset
clustered_dataset = pd.read_csv('clustered_data.csv')
print(clustered_dataset.head())

# Topic analysis
topic_analysis_df = clustered_dataset['dominant_topic'].value_counts().reset_index().rename(columns = {'index': 'dominant_topic', 
                                                                                                        'dominant_topic': 'topic_counts'})

topic_analysis_df['topic_contribution'] = topic_analysis_df['topic_counts'].apply(lambda x: round(x / topic_analysis_df['topic_counts'].sum(), 4))

unique_topic_keywords_df = clustered_dataset[['dominant_topic', 'topic_keywords']].drop_duplicates(subset = 'dominant_topic', keep = 'first')

topic_breakdown_df = pd.merge(topic_analysis_df, unique_topic_keywords_df, on = 'dominant_topic')
topic_breakdown_df = topic_breakdown_df[['dominant_topic', 'topic_keywords', 'topic_counts', 'topic_contribution']]
print(topic_breakdown_df)