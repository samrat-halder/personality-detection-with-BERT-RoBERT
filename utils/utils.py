#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
These are some helper function which are used in the data preparation and
summarization stage.
###########################

__author__ = "Samrat Halder"
__copyright__ = "Copyright 2020, ELEN6040 Research Project"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Samrat Halder"
__email__ = "sh3970@columbia.edu"
__status__ = "Production"
"""

from __future__ import unicode_literals, print_function
import nltk
import re
nltk.download('punkt')
from spacy.lang.en import English # updated
import pandas as pd

#raw_text = 'Hello, world. Here are two sentences.'
nlp = English()
nlp.max_length = 100000000
nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated

def docToSent(raw_text):
  """
  Splits a document into sentences using spacy
  :params raw_text: (str) full document
  :params return: length of the documents ie. how many sentences
  """
  doc = nlp(raw_text)
  sentences = [sent.string.strip() for sent in doc.sents]
  
  return len(sentences)

def docSep(row, n=150):
  """
  Splits a full document into sequences of length 150
  :params row: dataframe row which contains the "comment" column
  :params n: (int) sequence length
  :params return: (string) string of sequences joined by "|"
  """

  text = row['comment']
  text = textClean(text)
  text = nltk.word_tokenize(text)
  text_ =  [' '.join(text[i:i+n]) for i in range(0, len(text), n)]
  text_ = "|".join(text_)
  
  return text_

def removeSmallDoc(row):
  """
  Removes documents from the dataframe after some text cleaning
  which are less than length 50
  :params row: dataframe row which contains the comment columnn
  :params return: -1 or text. -1 if the document is being discarded due to shorter
  length
  """

  text = row['comment']
  text = text.split(' ')
  if len(text) < 50:
    return -1
  else: 
    text = ' '.join(text).strip(' ')
    text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)
    return text

def textClean(s):
  """
  Does some basic text cleaning operations by removing links, websites, non-alpha numeric
  characters from the text
  :params s: (strin) uncleaned text
  :params return: (strin) cleaned text
  """
  s = re.sub(r'\s*(?:https?://)?www\.\S*\.[A-Za-z]{2,5}\s*', ' ', s).strip()
  return s

def docSplit(df, cols):
  """
  Explode the dataframe with concatenated sequences
  :params df: pandas dataframe
  :params cols: (list) columns which do not contain the concatenated string
  :params return: pandas dataframe
  """
  df = (df.set_index(cols)
        .apply(lambda x: x.str.split('|').explode())
        .reset_index())
  return df

def overlappingSplit(row, n=150, n_overlap=25):
  """
  Split documents into sequences of length n with an overlapping length of n_overlap
  :params row: dataframe row which contains "comment" column
  :params n: (int) sequence length
  :params n_overlap: (int) overlapping length
  :params return: (string) sequences concatenated by "|"
  """
  text = row['comment']
  text = textClean(text)
  l_total = []
  l_partial = []
  if len(text.split())//n >0:
    n = len(text.split())//n
  else: 
    n = 1
  for w in range(n):
    if w == 0:
      l_partial = text.split()[:(n+n_overlap)]
      l_total.append(" ".join(l_partial))
    else:
      l_partial = text.split()[w*n:w*n + (n+n_overlap)]
      l_total.append(" ".join(l_partial))
  
  return '|'.join(l_total)

def splitDfWithIndex(df):
  """
  Creates a new data frame by comment comment column (list of sequences) 
  into multiple rows with assigning row index 
  :params df: pandas datafram
  :params return: pandas dataframe with index column
  """
  text_l = []
  label_l = []
  index_l =[]
  for idx,row in df.iterrows():
    for l in row['comment']:
      text_l.append(l)
      label_l.append(row['type'])
      index_l.append(idx)
  #len(train_l), len(label_l), len(index_l)
  df_new = pd.DataFrame({'comment':df_l, 'type':label_l, 'index':index_l})
  return df_new

