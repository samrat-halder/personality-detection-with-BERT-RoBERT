import pandas as pd
import os
import pickle
from pathlib import Path
from utils import *
import sys
import copy
import time
t = time.time()
n_sample = 999999
all_class = True
print('all_class set to ', str(all_class))
print('sampling is set to ', str(n_sample))
print('======================================================\n')
data_path = '/home/oblivion/mbti-personality-detection/data'
filename = 'mbti9k_comments.csv'
extended_filename = 'data_summary_'+ str(all_class)+ '.pkl'
mbti_data = pd.read_csv(os.path.join(data_path, filename))
print('Total number of rows ', len(mbti_data))
mbti_data = mbti_data.drop_duplicates(subset=['author'], keep='first') #103 authors are duplicated with different types
print('Total number of rows after removing duplicate authors ', len(mbti_data))
if n_sample != 999999:
  print('Sampling ', n_sample, ' rows...')
  mbti_data = mbti_data.sample(n=n_sample, random_state=1)
mbti_data = mbti_data[['author', 'comment', 'type']]
print('======================================================')
print('Class distribution:')
print(mbti_data.type.value_counts())
print("Removing classes with very low counts : 'esfp','estp','estj','esfj','istp','istj','isfp','isfj")
mbti_data = mbti_data[~mbti_data['type'].isin(['esfp','estp','estj','esfj','istp','istj','isfp','isfj'])].reset_index(drop=True)
print('======================================================')
if not all_class:
  print('Combine similar classes and filtering less frequent classes...')
  mbti_data['type'] = mbti_data['type'].replace(['entp','enfj','enfp','entj'], 'en')
  mbti_data['type'] = mbti_data['type'].replace([ 'infp','infj'], 'inf')
  #mbti_data.at[mbti_data['type'] == 'esfp','estp','estj','esfj'] = 'es'
  print('New Class distribution:')
  print(mbti_data.type.value_counts())
  print('======================================================')
types = mbti_data['type'].unique().tolist()
print('All types : ', types)
print('Calculating sentence length using spaCy..')
mbti_data['len_comment'] = mbti_data.comment.apply(lambda x: docToSent(x))
mbti_data.to_pickle(os.path.join(data_path, extended_filename))
print('Total time taken ', round(time.time()-t,2), ' s')
print('======================================================')
for type_ in types:
  tmp = mbti_data[mbti_data['type']==type_]
  print(tmp.describe())

