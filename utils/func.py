import pandas as pd
import numpy as np
import nltk
import modeling
import optimization
import run_classifier
import tensorflow as tf

def create_examples(lines, set_type, labels=None):
  #Generate data for the BERT model
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
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

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
    """The actual input function."""
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
