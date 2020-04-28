import pandas as pd
import os 
import pickle
from pathlib import Path
from utils import *

data_path = '/home/oblivion/mbti-personality-detection/data'
filename = 'mbti9k_comments.csv'
split_sent_filename = 'training_data_sample_100.pkl'
mbti_data = pd.read_csv(os.path.join(data_path, filename), nrows=1000)
print('Total number of rows ', len(mbti_data))
mbti_data = mbti_data.drop_duplicates(subset=['author'], keep='first') #103 authors are duplicated with different types
print('Total number of rows after removing duplicate authors ', len(mbti_data))
mbti_data = mbti_data[['author', 'comment', 'type']]
mbti_data['comment'] = mbti_data.apply(lambda x: docSep(x, n=100), axis=1)
mbti_data = docSplit(mbti_data, ['author','type'])
mbti_data['comment'] = mbti_data.apply(lambda x: removeSmallDoc(x), axis=1)
mbti_data = mbti_data[mbti_data['comment'] != -1]
print('Total number of rows after splitting douments ', len(mbti_data))
mbti_data.to_pickle(os.path.join(data_path, split_sent_filename))




