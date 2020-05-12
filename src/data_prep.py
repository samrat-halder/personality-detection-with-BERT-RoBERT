#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script does the data preparation steps for the deep learning model.
It creates two pickle files 1. For the BERT fine tuning 2. For RoBERT model


__author__ = "Samrat Halder"
__copyright__ = "Copyright 2020, ELEN6040 Research Project"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Samrat Halder"
__email__ = "sh3970@columbia.edu"
__status__ = "Production"
"""
import pandas as pd
import os 
import pickle
from pathlib import Path
from utils import *
import sys
import copy
import time
import configparser
configParser = configparser.RawConfigParser()
configFilePath = './../config.txt'
configParser.read(configFilePath)
t = time.time()

seq_length = int(configParser.get('config', 'seq_length')) #150 default sequence length. Do not set to a very high value. MAX=512
overlap_length = int(configParser.get('config', 'overlap_length')) #25 overlapping length for RoBERT
n_sample = int(configParser.get('config', 'n_sample'))     #number of rows to be picked fromt the whole dataset. default=999999. can be set to a lower value
all_class = eval(configParser.get('config', 'all_class'))  #True or False. If True does a class aggregration 
print('all_class set to ', str(all_class))
print('sampling is set to ', str(n_sample))
print('======================================================\n')


data_path = './../data'#'/home/oblivion/mbti-personality-detection/data'
filename = 'mbti9k_comments.csv'
split_sent_filename = 'training_data_sample_' + str(seq_length) + '_' + str(n_sample) + '_' + str(all_class) + '.pkl' #File name for BERT
split_sent_filename_h = 'training_data_sample_h_' + str(seq_length) + '_' + str(n_sample) + '_'+ str(all_class)+ '.pkl' #File name for RoBERT
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
if not all_class: #if all_class set to false, does a class grouping
  print('Combine similar classes and filtering less frequent classes...')
  mbti_data['type'] = mbti_data['type'].replace(['entp','enfj','enfp','entj'], 'en')
  mbti_data['type'] = mbti_data['type'].replace([ 'infp','infj'], 'inf')
  #mbti_data.at[mbti_data['type'] == 'esfp','estp','estj','esfj'] = 'es'
  print('New Class distribution:')
  print(mbti_data.type.value_counts())
  print('======================================================')

mbti_data_1 = copy.deepcopy(mbti_data)
t_h = time.time()
print('Splitting comments for heirerchical model by sequence length ', seq_length, ' overlapping length ', overlap_length)
mbti_data_1['comment'] = mbti_data_1.apply(lambda x: overlappingSplit(x, n=seq_length, n_overlap=overlap_length), axis=1)
mbti_data_1['index'] = mbti_data_1.index
mbti_data_1 = docSplit(mbti_data_1, ['author','type','index'])
mbti_data_1 = mbti_data_1[mbti_data_1['comment'] != -1].reset_index(drop=True)
print("Done..!")
print('Total time taken to prepare data for RoBERT', round((time.time()-t_h), 2),' s')
print('Class distribution after splitting:')
print(mbti_data_1.type.value_counts())
print('Total number of rows after splitting douments ', len(mbti_data_1))
mbti_data_1.to_pickle(os.path.join(data_path, split_sent_filename_h))
print('======================================================')

t_split = time.time()
print('Splitting comments by sequence length ', seq_length)
mbti_data['comment'] = mbti_data.apply(lambda x: docSep(x, n=seq_length), axis=1)
mbti_data = docSplit(mbti_data, ['author','type'])
mbti_data['comment'] = mbti_data.apply(lambda x: removeSmallDoc(x), axis=1)
mbti_data = mbti_data[mbti_data['comment'] != -1].reset_index(drop=True)
print("Done..!")
print('Total time taken to prepare split sentence', round((time.time()-t_split), 2),' s')
print('Class distribution after splitting:')
print(mbti_data.type.value_counts())

print('Total number of rows after splitting douments ', len(mbti_data))
mbti_data.to_pickle(os.path.join(data_path, split_sent_filename))
print('Total time taken to prepare data ', round((time.time()-t), 2),' s')
