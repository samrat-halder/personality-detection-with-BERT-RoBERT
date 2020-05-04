import pandas as pd
import os 
import pickle
from pathlib import Path
from utils import *
import sys
import copy


seq_length = 150
overlap_length = 25
n_sample = 1000

data_path = '/home/oblivion/mbti-personality-detection/data'
filename = 'mbti9k_comments.csv'
split_sent_filename = 'training_data_sample_' + str(seq_length) + '_' + str(n_sample) + '.pkl'
split_sent_filename_2 = 'training_data_sample_h_' + str(seq_length) + '_' + str(n_sample) + '.pkl'
mbti_data = pd.read_csv(os.path.join(data_path, filename))
print('Total number of rows ', len(mbti_data))
mbti_data = mbti_data.drop_duplicates(subset=['author'], keep='first') #103 authors are duplicated with different types
print('Total number of rows after removing duplicate authors ', len(mbti_data))

print('Sampling ', n_sample, ' rows...')
mbti_data = mbti_data.sample(n=n_sample, random_state=1)
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

#check doc length summary
#mbti_data['len_comment'] = mbti_data.comment.apply(lambda x: docToSent(x))
#print(mbti_data.describe())
#mbti_data.drop('len_comment', axis=1, inplace=True)

mbti_data_1 = copy.deepcopy(mbti_data)
print('Splitting comments for heirerchical model by sequence length ', seq_length, ' overlapping length ', overlap_length)
mbti_data_1['comment'] = mbti_data_1.apply(lambda x: overlappingSplit(x, n=seq_length, n_overlap=overlap_length), axis=1)
mbti_data_1_1 = splitDfWithIndex(mbti_data_1)
del mbti_data_1
mbti_data_1_1.to_pickle(os.path.join(data_path, split_sent_filename_2))

print("Done..!")
print('======================================================')
print('Class distribution after splitting:')
print(mbti_data_1_1.type.value_counts())
print('Total number of rows after splitting douments ', len(mbti_data_1))
mbti_data_1_1.to_pickle(os.path.join(data_path, split_sent_filename_2))


print('Splitting comments by sequence length ', seq_length)
mbti_data['comment'] = mbti_data.apply(lambda x: docSep(x, n=seq_length), axis=1)
mbti_data = docSplit(mbti_data, ['author','type'])
mbti_data['comment'] = mbti_data.apply(lambda x: removeSmallDoc(x), axis=1)
mbti_data = mbti_data[mbti_data['comment'] != -1].reset_index(drop=True)
print("Done..!")
print('======================================================')
print('Class distribution after splitting:')
print(mbti_data.type.value_counts())

print('Total number of rows after splitting douments ', len(mbti_data))
mbti_data.to_pickle(os.path.join(data_path, split_sent_filename))

