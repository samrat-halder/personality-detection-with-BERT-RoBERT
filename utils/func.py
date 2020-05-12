#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script includes some helper function which are used for the BERT and RoBERT 
models.
############################
__author__ = "Samrat Halder"
__copyright__ = "Copyright 2020, ELEN6040 Research Project"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Samrat Halder"
__email__ = "sh3970@columbia.edu"
__status__ = "Production"
"""

import pandas as pd
import numpy as np
import nltk
import modeling
import optimization
import run_classifier
import tensorflow as tf

def create_examples(lines, set_type, labels=None):
  """Generate data for the BERT model
  :params lines: (numpy array) input sentences
  :params set_type: (str) eg. train
  :params labels: (numpy array) labels
  :return examples: list
  """
  
  guid = f'{set_type}'
  examples = []
  if guid == 'train':
      for line, label in zip(lines, labels):
          text_a = line
          label = str(label)
          examples.append(
            run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
  else:
      for line in lines:
          text_a = line
          label = '0'
          examples.append(
            run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
  return examples

def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator.
  :params features: features object generated by BERT
  :params seq_length: (int) maximum sequence length , upper limit 512
  :params is_training: True or False
  :params drop_remainder: True or False, if True it will drop the examples in the remainder with batch size
  """

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function from Google Research repository"""
    print(params)
    batch_size = 500

    num_examples = len(features)

    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn

def get_embedding(predictions):
  """list to map the actual labels to the predictions
  This function gets the embeddings from BERT prediction object
  We have made some necessary changes in run_classifier script of Google Research
  To also include the last layer in the prediction
  :params prediction: predicted output from BERT
  :params return: pooled output representation of the node corresponding to [CLS] token
  """ 
  return [prediction['pooled_output'] for _,prediction in enumerate(predictions)]


def serving_input_receiver_fn():
  """
  This function is not being used at the moment, TODO
  """
  feature_spec = {
    "unique_ids": tf.FixedLenFeature([], tf.int64),
    "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
    "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
    "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
  }

  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[batch_size],
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def lstm_generator(df, batches_per_epoch, batch_size,
                   num_features):
  """
  Returns a genrator object for training LSTM model
  params df: (pandas dataframe) containing Embeddings and labels
  params batches_per_epoch: (int) Number of batchs per epoch
  params batch_size: (int)
  params num_features: (int)
  """
  x_list= df['emb'].to_list()
  y_list =  df['label'].to_list()
  # Generate batches
  while True:
    for b in range(batches_per_epoch):
      longest_index = (b + 1) * batch_size - 1
      timesteps = len(max(x_list[:(b + 1) * batch_size][-batch_size:], key=len))
      x_train = np.full((batch_size, timesteps, num_features), -99.)
      y_train = np.zeros((batch_size,  1))
      for i in range(batch_size):
        li = b * batch_size + i
        y_train[i] = y_list[li]
        x_train[i, 0:len(x_list[li]), :] = x_list[li]
        #y_train[i] = y_list[li]
      yield x_train, y_train
