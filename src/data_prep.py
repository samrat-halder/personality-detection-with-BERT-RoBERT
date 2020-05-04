import pandas as pd
import os 
import pickle
from pathlib import Path
from utils import *
import sys
seq_length = 150
n_sample = 5000

data_path = '/home/oblivion/mbti-personality-detection/data'
filename = 'mbti9k_comments.csv'
split_sent_filename = 'training_data_sample_' + str(seq_length) + '_' + str(n_sample) + '.pkl'
mbti_data = pd.read_csv(os.path.join(data_path, filename), nrows=n_sample)
print('Total number of rows ', len(mbti_data))
mbti_data = mbti_data.drop_duplicates(subset=['author'], keep='first') #103 authors are duplicated with different types
print('Total number of rows after removing duplicate authors ', len(mbti_data))
mbti_data = mbti_data[['author', 'comment', 'type']]
print('======================================================')
print('Class distribution:')
print(mbti_data.type.value_counts())
print('======================================================')
print('Combine similar classes and filtering less frequent classes...')
mbti_data['type'] = mbti_data['type'].replace(['entp','enfj','enfp','entj'], 'en')
mbti_data['type'] = mbti_data['type'].replace([ 'infp','infj'], 'inf')
#mbti_data.at[mbti_data['type'] == 'esfp','estp','estj','esfj'] = 'es'
mbti_data = mbti_data[~mbti_data['type'].isin(['esfp','estp','estj','esfj','istp','istj','isfp','isfj'])].reset_index(drop=True)
print('New Class distribution:')
print(mbti_data.type.value_counts())
print('======================================================')
mbti_data['len_comment'] = mbti_data.comment.apply(lambda x: docToSent(x))
print(mbti_data.describe())
print('Splitting comments by sequence length ', seq_length)
mbti_data['comment'] = mbti_data.apply(lambda x: docSep(x, n=seq_length), axis=1)
mbti_data.drop('len_comment', axis=1, inplace=True)
mbti_data = docSplit(mbti_data, ['author','type'])
mbti_data['comment'] = mbti_data.apply(lambda x: removeSmallDoc(x), axis=1)
mbti_data = mbti_data[mbti_data['comment'] != -1].reset_index(drop=True)
print("Done..!")
print('======================================================')
print('Class distribution after splitting:')
print(mbti_data.type.value_counts())

print('Total number of rows after splitting douments ', len(mbti_data))
mbti_data.to_pickle(os.path.join(data_path, split_sent_filename))




