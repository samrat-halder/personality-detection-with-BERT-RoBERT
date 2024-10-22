#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script does fine tuning and training the BERT classifier.


__author__ = "Samrat Halder"
__copyright__ = "Copyright 2020, ELEN6040 Research Project"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Samrat Halder"
__email__ = "sh3970@columbia.edu"
__status__ = "Production"
"""

#%tensorflow_version 1.11
import pandas as pd
import os 
import datetime
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf
import modeling
import optimization
import run_classifier
import tokenization
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from func import *
import time
import configparser
configParser = configparser.RawConfigParser()
configFilePath = './../config.txt'
configParser.read(configFilePath)
#######################
TRAIN_BATCH_SIZE = int(configParser.get('config', 'TRAIN_BATCH_SIZE'))     #training batch size=4. Do not set a high value. May arise memory error
EVAL_BATCH_SIZE = int(configParser.get('config', 'EVAL_BATCH_SIZE'))       #evaluation batch size=2
LEARNING_RATE = 1e-5     #learning rate
NUM_TRAIN_EPOCHS = 1.0   #number of training epochs
WARMUP_PROPORTION = 0.1  #warmup proportion 
MAX_SEQ_LENGTH = int(configParser.get('config', 'seq_length'))      #max sequence length for BERT input sequence=150
NUM_SAMPLE = int(configParser.get('config', 'NUM_SAMPLE'))        #number of samples being used=2500. This should be same as the data_prep file
uncased =  eval(configParser.get('config', 'uncased'))#True    #can be True or False Defaul True. We use uncased version of the BERT pretrained model
all_class = eval(configParser.get('config', 'all_class'))  #False        #can be True or False
#######################
folder = './../model_folder'
# Model configs
SAVE_CHECKPOINTS_STEPS = 100000 #if you wish to finetune a model on a larger dataset, use larger interval
# each checpoint weights about 1,5gb
ITERATIONS_PER_LOOP = 100000
NUM_TPU_CORES = 8
t = time.time()

if uncased:
	with zipfile.ZipFile(os.path.join(folder, "uncased_L-12_H-768_A-12.zip"),"r") as zip_ref:
    		zip_ref.extractall(folder)

	BERT_MODEL = 'uncased_L-12_H-768_A-12'
	BERT_PRETRAINED_DIR = f'{folder}/uncased_L-12_H-768_A-12'
	DO_LOWER_CASE = BERT_MODEL.startswith('uncased')
else:
	with zipfile.ZipFile(os.path.join(folder, "cased_L-12_H-768_A-12.zip"),"r") as zip_ref:
                zip_ref.extractall(folder)
	BERT_MODEL = 'cased_L-12_H-768_A-12'
	BERT_PRETRAINED_DIR = f'{folder}/cased_L-12_H-768_A-12'
	DO_LOWER_CASE = BERT_MODEL.startswith('cased')

print('BERT model :', BERT_MODEL)
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')

OUTPUT_DIR = f'{folder}/outputs'
print(f'Model output directory: {OUTPUT_DIR}')
print(f'BERT pretrained directory: {BERT_PRETRAINED_DIR}')

fname = './../data/training_data_sample_' + str(MAX_SEQ_LENGTH) + '_' + str(NUM_SAMPLE) + '_' + str(all_class) + '.pkl'
mbti_data = pd.read_pickle(fname)
df = pd.DataFrame()
df["Text"] = mbti_data['comment']
df["Label"] = LabelEncoder().fit_transform(mbti_data['type'])

del mbti_data

X_train, X_test, y_train, y_test = train_test_split(df["Text"].values,
                                    df["Label"].values, test_size=0.6, random_state=42, shuffle=True)
print('Length of training sample ', len(X_train))
#Preprocess data for BERT
label_list = [str(i) for i in sorted(df['Label'].unique())]
train_examples = create_examples(X_train, 'train', labels=y_train)
print("\n__________\nRow 0 - guid of training set : ", train_examples[0].guid)
print("\n__________\nRow 0 - text_a of training set : ", train_examples[0].text_a)
print("\n__________\nRow 0 - text_b of training set : ", train_examples[0].text_b)
print("\n__________\nRow 0 - label of training set : ", train_examples[0].label)

tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE) #Run end-to-end tokenization
print("\n__________\nRow 0 - tokenized version of text_a of training set :\n", ' '.join(tokenizer.tokenize(train_examples[0].text_a)))

#Create training features
print('\nCreating training features. Please wait...')
train_features = run_classifier.convert_examples_to_features(
    train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

print("\n__________\nInput IDs : ", train_features[0].input_ids)
print("\n__________\nInput Masks : ", train_features[0].input_mask)
print("\n__________\nSegment IDs : ", train_features[0].segment_ids)

num_train_steps = int(
    len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

tpu_cluster_resolver = None
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

model_fn = run_classifier.model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=len(label_list),
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=False,
    use_one_hot_embeddings=True)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE)

print('\n__________\nStarted training at {} '.format(datetime.datetime.now()))
print('\nNum examples = {}'.format(len(train_examples)))
print('\nBatch size = {}'.format(TRAIN_BATCH_SIZE))
tf.logging.info("Num steps = %d", num_train_steps)

train_input_fn = run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=True)
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print('\n__________\nFinished training at {}'.format(datetime.datetime.now()))
print('\nTotal time taken to fine tune the model ', round(time.time()-t,2), ' s')
