import pandas as pd

# Read datafile
dataset = pd.read_csv('spotify-public-dataset.csv')
print(dataset.head())
print('--------------------------')

# EDA
# How many missing data?
print(dataset.isna().sum())
print('--------------------------')

# How many unique values?
print(dataset['message_type'].value_counts())
print('--------------------------')
print(dataset['author_id'].nunique())
print('--------------------------')
print(dataset['severity'].value_counts())
print('--------------------------')

# Dealing with datetime object
dataset.sort_values('created_at', inplace = True)
dataset.reset_index(drop = True, inplace = True)
dataset['created_at_datetime'] = pd.to_datetime(dataset['created_at'])
dataset['year'] = dataset['created_at_datetime'].apply(lambda x: x.year)
dataset['month'] = dataset['created_at_datetime'].apply(lambda x: x.month)
dataset['day'] = dataset['created_at_datetime'].apply(lambda x: x.day)
dataset['hour'] = dataset['created_at_datetime'].apply(lambda x: x.hour)
dataset.drop('created_at_datetime', axis = 1, inplace = True)


# dataset.to_csv('sorted_data.csv', index = False)