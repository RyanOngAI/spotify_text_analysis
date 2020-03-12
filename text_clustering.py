import pickle
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import gensim
from gensim import corpora, models

# Defining TopicModelling class
class TopicModelling():

    def __init__(self, data):
        self.no_topics = 5
        self.no_below = 15
        self.no_above = 0.5 # percentage
        self.keep_top_n = 100000

        self.dictionary = None
        self.bow_corpus = None
        self.ldamodel = None

        self.data = data.apply(lambda x: x.split())

    def create_filter_dictionary(self):
        
        # Creating dictionary
        self.dictionary = corpora.Dictionary(self.data)
        
        count = 0
        for k, v in self.dictionary.iteritems():
            print(k, v)
            count += 1
            if count > 10:
                break
        
        # Filtering dictionary
        self.dictionary.filter_extremes(self.no_below, self.no_above, self.keep_top_n)
        
        # Saving dictionary
        print('Saved dictionary as dictionary.gensim')
        self.dictionary.save('dictionary.gensim')

    def create_bow(self):
        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.data]
        print('Saved bow corpus as bow_corpus.pkl')
        pickle.dump(self.bow_corpus, open('bow_corpus.pkl', 'wb'))

    def create_lda_model(self):
        self.ldamodel = models.ldamodel.LdaModel(self.bow_corpus, num_topics = self.no_topics, id2word = self.dictionary, passes = 15)
        print('Saved LDA model as model.gensim')
        self.ldamodel.save('model.gensim')

        return self.ldamodel

    def predict(self, processed_new_doc):
        new_doc_bow = self.dictionary.doc2bow(processed_new_doc)
        
        return self.ldamodel.get_document_topics(new_doc_bow)

    def topic_categorisation(self, ldamodel = None, corpus = None):
        topic_data = []
        
        if ldamodel is None:
            ldamodel = self.ldamodel
        
        if corpus is None:
            corpus = self.bow_corpus

        for i, row in enumerate(ldamodel[corpus]):
            # order by topic probabilities
            row = sorted(row, key = lambda x: (x[1]), reverse = True)
            for j, (topic_num, contribution) in enumerate(row):
                if j == 0: # Most dominant topic
                    word_list = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, word_contribution in word_list])
                    topic_data.append((int(topic_num), round(contribution, 4), topic_keywords))
                else:
                    break
        
        return topic_data


if "__name__" == "__main__":
    # Load the cleaned dataset
    cleaned_dataset = pd.read_csv('processed_data.csv')

    # Initialising sentiment analyser
    sentiment_analyser = SentimentIntensityAnalyzer()

    # Apply sentiment analysis to processed message body
    cleaned_dataset['sentiment'] = cleaned_dataset['message_body_processed'].apply(lambda x: sentiment_analyser.polarity_scores(x)['compound'])

    # Instantiating TopicModelling class and applying it to processed message body
    lda_model = TopicModelling(cleaned_dataset['message_body_processed'])
    lda_model.create_filter_dictionary()
    lda_model.create_bow()
    trained_lda_model = lda_model.create_lda_model()
    topic_categorisation = lda_model.topic_categorisation()

    cleaned_dataset[['dominant_topic', 'contribution_perc', 'topic_keywords']] = pd.DataFrame(topic_categorisation)
    # cleaned_dataset.to_csv('clustered_data.csv', index = False)