import nltk
import re

def docSep(row, n=150):

  text = row['comment']
  text = textClean(text)
  text = nltk.word_tokenize(text)
  text_ =  [' '.join(text[i:i+n]) for i in range(0, len(text), n)]
  text_ = "|".join(text_)
  
  return text_

def removeSmallSent(row):

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
