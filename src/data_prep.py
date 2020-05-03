import pandas as pd
import os 
import pickle
from pathlib import Path
from utils import *
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
mbti_data['comment'] = mbti_data.apply(lambda x: docSep(x, n=seq_length), axis=1)
mbti_data = docSplit(mbti_data, ['author','type'])
mbti_data['comment'] = mbti_data.apply(lambda x: removeSmallDoc(x), axis=1)
mbti_data = mbti_data[mbti_data['comment'] != -1]
print('Total number of rows after splitting douments ', len(mbti_data))
mbti_data.to_pickle(os.path.join(data_path, split_sent_filename))




