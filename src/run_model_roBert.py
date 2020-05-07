import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.utils import np_utils
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dense, Input, concatenate, Layer, Lambda, Dropout, Activation
from sklearn.model_selection import train_test_split
from func import *
MAX_SEQ_LENGTH = 150
NUM_SAMPLE = 1500 

emb_data = pd.read_pickle('./../data/training_data_lstm_h_' + str(MAX_SEQ_LENGTH) + '_' + str(NUM_SAMPLE) + '.pkl')
label_list = emb_data['label'].unique().tolist()
df_train_val, df_test = train_test_split(emb_data, test_size=0.2, random_state=35)
df_train, df_val = train_test_split(df_train_val, test_size=0.2, random_state=35)
batch_size_train = 3
batches_per_epoch_train = len(df_train) // batch_size_train
num_features_train = num_features_val = num_features_test = 768 #BERT output embedding size
batch_size_val = 5
batches_per_epoch_val = len(df_val) // batch_size_train

text_input = Input(shape=(None,768,), dtype='float32', name='text')
l_mask = layers.Masking(mask_value=-99.)(text_input)
# encode in a single vector via a LSTM
encoded_text = layers.LSTM(100,)(l_mask)
out_dense = layers.Dense(30, activation='relu')(encoded_text)
# And we add a softmax classifier on top
out = layers.Dense(len(label_list), activation='softmax')(out_dense)
# At model instantiation, we specify the input and the output:
model = Model(text_input, out)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
model.summary()
call_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.95, patience=3, verbose=2,
                                mode='auto', min_delta=0.01, cooldown=0, min_lr=0)
model.fit_generator(lstm_generator(df_train, batches_per_epoch_train, batch_size_train,
                   num_features_train), steps_per_epoch=batches_per_epoch_train, epochs=10,
                    validation_data=lstm_generator(df_val, batches_per_epoch_val, batch_size_val,
                   num_features_val), validation_steps=batches_per_epoch_val, callbacks =[call_reduce] )


model.evaluate_generator(lstm_generator(df_test, batches_per_epoch_test, batch_size_test,
                   num_features_test), steps= batches_per_epoch_test)
