import re
import pandas as pd

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

# Read datafile & load text processing tools (stopwords, tokeniser, and lemmatiser)
sorted_data = pd.read_csv('sorted_data.csv')
english_stopwords = stopwords.words('english')
tokeniser = TweetTokenizer()
lemmatizer = WordNetLemmatizer()

# Text processing
def text_processing(dataframe, text_column):
    
    # Remove initial missing message_body (total of 14 rows)
    dataframe.dropna(subset = [text_column], axis = 0, inplace = True)
    
    cleaned_text_column = text_column + '_processed'    
    # Remove spotify handles, urls, punctuations, numbers, and special symbols. Lowercase texts.
    dataframe[cleaned_text_column] = dataframe[text_column].apply(lambda x : ' '.join(re.sub(r"(@[A-Za-z0-9]+)|(\w+:\/*\S+)|([^a-zA-Z\s])", " ", x).split()).lower())

    # Drop the duplicates in message_body_processed (should we remove duplications?)
    dataframe.drop_duplicates(subset = [cleaned_text_column], keep = 'first', inplace = True)

    # Remove stopwords
    dataframe[cleaned_text_column] = dataframe[cleaned_text_column].apply(lambda x : [word for word in tokeniser.tokenize(x) if word not in english_stopwords])

    # Lemmatisation - speed overhead
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        
        return tag_dict.get(tag, wordnet.NOUN)

    dataframe[cleaned_text_column] = dataframe[cleaned_text_column].apply(lambda x : [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in x]) 

    # Only keep messages that's greater than 5 words and less than 200 words (HARDCODED)
    # print(dataframe[cleaned_text_column].map(len).quantile([0.25, 0.5, 0.75, 0.95])) # 5, 8, 12, 18
    cleaned_data = dataframe[(dataframe[cleaned_text_column].map(len) >= 5) & (dataframe[cleaned_text_column].map(len) < 25)]
    cleaned_data.reset_index(drop = True, inplace = True)
    
    # Join the tokens into a string again
    cleaned_data[cleaned_text_column] = cleaned_data[cleaned_text_column].apply(lambda x : ' '.join(x))

    return cleaned_data

cleaned_dataset = text_processing(sorted_data, 'message_body')
print(len(cleaned_dataset))

# cleaned_dataset.to_csv('processed_data.csv', index = False)