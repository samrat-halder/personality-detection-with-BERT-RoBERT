from __future__ import unicode_literals, print_function
import nltk
import re
nltk.download('punkt')
from spacy.lang.en import English # updated

#raw_text = 'Hello, world. Here are two sentences.'
nlp = English()
nlp.max_length = 100000000
nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated

def docToSent(raw_text):
  
  doc = nlp(raw_text)
  sentences = [sent.string.strip() for sent in doc.sents]
  
  return len(sentences)

def docSep(row, n=150):

  text = row['comment']
  text = textClean(text)
  text = nltk.word_tokenize(text)
  text_ =  [' '.join(text[i:i+n]) for i in range(0, len(text), n)]
  text_ = "|".join(text_)
  
  return text_

def removeSmallDoc(row):

  text = row['comment']
  text = text.split(' ')
  if len(text) < 50:
    return -1
  else: 
    text = ' '.join(text).strip(' ')
    text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)
    return text

def textClean(s):

  s = re.sub(r'\s*(?:https?://)?www\.\S*\.[A-Za-z]{2,5}\s*', ' ', s).strip()
  return s

def docSplit(df, cols):

    df = (df.set_index(cols)
        .apply(lambda x: x.str.split('|').explode())
        .reset_index())
    return df

def overlappingSplit(row, n=150, n_overlap=25):
  
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
  
  return l_total

def splitDfWithIndex(df):

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

