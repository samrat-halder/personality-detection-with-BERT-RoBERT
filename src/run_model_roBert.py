import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.utils import np_utils
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dense, Input, concatenate, Layer, Lambda, Dropout, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from func import *
import time

t = time.time()
epochs = 5
MAX_SEQ_LENGTH = 150
NUM_SAMPLE = 999999 
all_class = False
emb_data = pd.read_pickle('./../data/training_data_lstm_h_' + str(MAX_SEQ_LENGTH) + '_' + str(NUM_SAMPLE) + '_' + str(all_class) + '.pkl')
label_list = emb_data['label'].unique().tolist()
df_train_val, df_test = train_test_split(emb_data, test_size=0.1, random_state=35)
df_train, df_val = train_test_split(df_train_val, test_size=0.1, random_state=35)

print('\n___________\nSize of training-validation set ', len(df_train_val))
print('Class distribution \n', df_train_val['label'].value_counts()) 
print('\n___________\nSize of training-validation set ', len(df_test))
print('Class distribution \n', df_test['label'].value_counts())
batch_size_train = 5
batches_per_epoch_train = len(df_train) // batch_size_train
df_train = df_train[:batch_size_train*batches_per_epoch_train] #Fixing dimension to nearest batch
assert len(df_train) == batches_per_epoch_train * batch_size_train

batch_size_val = 5
batches_per_epoch_val = len(df_val) // batch_size_train
df_val = df_val[:batch_size_val*batches_per_epoch_val]

batch_size_test = 5
batches_per_epoch_test = len(df_test) // batch_size_train
df_test = df_test[:batch_size_test*batches_per_epoch_test]

num_features= 768 #BERT output embedding size

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
                   num_features), steps_per_epoch=batches_per_epoch_train, epochs=epochs,
                    validation_data=lstm_generator(df_val, batches_per_epoch_val, batch_size_val,
                   num_features), validation_steps=batches_per_epoch_val, callbacks =[call_reduce] )


test_generator = lstm_generator(df_test, batches_per_epoch_test, batch_size_test,
                   num_features)
loss, acc = model.evaluate_generator(test_generator, steps= batches_per_epoch_test)
y_pred = model.predict_generator(test_generator, steps= batches_per_epoch_test)

y_pred = np.argmax(y_pred, axis=-1)
y_test = df_test['label']#test_generator.classes[validation_generator.index_array]

print('\n__________\nloss: ', loss, 'accuracy: ', acc) # loss:  0.47286026436090467 accuracy:  0.864
print('accuracy_score: \n', classification_report(y_test, y_pred)) # accuracy_score:  0.095
print('Total time taken :', round(time.time()-t, 2), ' s')
