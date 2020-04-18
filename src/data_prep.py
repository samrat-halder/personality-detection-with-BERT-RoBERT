import pandas as pd
import os 
import pickle
from pathlib import Path

mbti_data = pd.read_csv('mbti9k_comments.csv', nrows=1000)
print(f'Total number of rows {len(mbti_data)}')
mbti_data = mbti_data.drop_duplicates(subset=['author'], keep='first') #103 authors are duplicated with different types
print(f'Total number of rows after removing duplicate authors {len(mbti_data)}')
mbti_data = mbti_data[['author', 'comment', 'type']]
mbti_data['comment'] = mbti_data.apply(lambda x: docSep(x, n=150), axis=1)
mbti_data = docSplit(mbti_data, ['author','type'])
mbti_data['comment'] = mbti_data.apply(lambda x: removeSmallSent(x), axis=1)
mbti_data = mbti_data[mbti_data['comment'] != -1]

print(f'Total number of rows after splitting douments {len(mbti_data)}')

mbti_data.to_pickle('./../data/training_data.pkl')




