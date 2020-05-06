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
folder = './../model_folder'

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 1.0
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 150
NUM_SAMPLE = 1500
uncased = True #False
#######################
folder = './../model_folder'
OUTPUT_DIR = f'{folder}/outputs'
# Model configs
SAVE_CHECKPOINTS_STEPS = 100000 #if you wish to finetune a model on a larger dataset, use larger interval
# each checpoint weights about 1,5gb
ITERATIONS_PER_LOOP = 100000
NUM_TPU_CORES = 8

"""
#path to output the new optimized model
output_path = os.path.join(OUTPUT_DIR, 'optimized_model')

sess = tf.Session()
imported_meta = tf.train.import_meta_graph(os.path.join(OUTPUT_DIR, 'model.ckpt-0.meta')) #based on the steps of your fine-tuned model
imported_meta.restore(sess, os.path.join(OUTPUT_DIR, 'model.ckpt-0')) #based on the steps of your fine-tuned model
my_vars = []
for var in tf.all_variables():
    if 'adam_v' not in var.name and 'adam_m' not in var.name:
        my_vars.append(var)
saver = tf.train.Saver(my_vars)
saver.save(sess, os.path.join(output_path, 'model.ckpt')) #change model.ckpt to name of your preference

print('Model optimization done')
"""
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
INIT_CHECKPOINT = os.path.join(OUTPUT_DIR, 'model.ckpt-42181')

fname = './../data/training_data_sample_' + str(MAX_SEQ_LENGTH) + '_' + str(NUM_SAMPLE) + '.pkl'
mbti_data = pd.read_pickle(fname)
df = pd.DataFrame()
df["Text"] = mbti_data['comment']
df["Label"] = LabelEncoder().fit_transform(mbti_data['type'])

del mbti_data

X_train, X_test, y_train, y_test = train_test_split(df["Text"].values,
                                    df["Label"].values, test_size=0.1, random_state=42, shuffle=True)

print('Length of test set:', len(X_test),  ', ', len(y_test)) 
#Preprocess data for BERT
label_list = [str(i) for i in sorted(df['Label'].unique())]
#train_examples = create_examples(X_train, 'train', labels=y_train)
predict_examples = create_examples(X_test, 'test')

tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE) #Run end-to-end tokenization

num_train_steps = int(
    len(X_train) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
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

#Test the model 
predict_features = run_classifier.convert_examples_to_features(
    predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

predict_input_fn = input_fn_builder(
    features=predict_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=True)

result = estimator.predict(input_fn=predict_input_fn)

preds = []
for prediction in result:
  preds.append(np.argmax(prediction['probabilities']))

print("\n__________\nAccuracy of BERT is:",accuracy_score(y_test,preds))
print(classification_report(y_test,preds))
