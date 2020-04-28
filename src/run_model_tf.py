#%tensorflow_version 1.11
import pandas as pd
import os 
import datetime
import tensorflow as tf
import modeling
import optimization
import run_classifier
import tokenization
import zipfile
import modeling
import optimization
import run_classifier
import tokenization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from func import *
#######################
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_TRAIN_EPOCHS = 1.0
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 100
#######################
folder = './../model_folder'
with zipfile.ZipFile(os.path.join(folder, "uncased_L-12_H-768_A-12.zip"),"r") as zip_ref:
    zip_ref.extractall(folder)

BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_PRETRAINED_DIR = f'{folder}/uncased_L-12_H-768_A-12'
OUTPUT_DIR = f'{folder}/outputs'
print(f'Model output directory: {OUTPUT_DIR}')
print(f'BERT pretrained directory: {BERT_PRETRAINED_DIR}')

mbti_data = pd.read_pickle('./../data/training_data_sample_125_3000.pkl')
df = pd.DataFrame()
df["Text"] = mbti_data['comment']
df["Label"] = LabelEncoder().fit_transform(mbti_data['type'])

del mbti_data

X_train, X_test, y_train, y_test = train_test_split(df["Text"].values,
                                    df["Label"].values, test_size=0.2, random_state=42, shuffle=True)

# Model configs
SAVE_CHECKPOINTS_STEPS = 100000 #if you wish to finetune a model on a larger dataset, use larger interval
# each checpoint weights about 1,5gb
ITERATIONS_PER_LOOP = 100000
NUM_TPU_CORES = 8
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

label_list = [str(i) for i in sorted(df['Label'].unique())]
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
train_examples = create_examples(X_train, 'train', labels=y_train)

tpu_cluster_resolver = None #Since training will happen on GPU, we won't need a cluster resolver
#TPUEstimator also supports training on CPU and GPU. You don't need to define a separate tf.estimator.Estimator.
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

num_train_steps = int(
    len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

model_fn = run_classifier.model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=len(label_list),
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available  
    use_one_hot_embeddings=True)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available 
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE)

print('Please wait...')
train_features = run_classifier.convert_examples_to_features(
    train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
print('Started training at {} '.format(datetime.datetime.now()))
print('Num examples = {}'.format(len(train_examples)))
print('Batch size = {}'.format(TRAIN_BATCH_SIZE))
tf.logging.info("Num steps = %d", num_train_steps)
train_input_fn = run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=True)
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print('Finished training at {}'.format(datetime.datetime.now()))

#Test the model 
predict_examples = create_examples(X_test, 'test')

predict_features = run_classifier.convert_examples_to_features(
    predict_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

predict_input_fn = input_fn_builder(
    features=predict_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

result = estimator.predict(input_fn=predict_input_fn)

preds = []
for prediction in result:
  preds.append(np.argmax(prediction['probabilities']))

print("Accuracy of BERT is:",accuracy_score(y_test,preds))
print(classification_report(y_test,preds))

