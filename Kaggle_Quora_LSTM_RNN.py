# Quora Question Pairs 2017 # 


############
#  packages
############
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import concatenate, Dense, Input, LSTM, SimpleRNN, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
import gensim, logging
import Tokenization_stopwords
import itertools
import pandas as pd
import numpy as np

wvdir = "/Users/ShaanPattni/Desktop/Machine Learning/Kaggle/quorarepo/"
embedding_file = "GoogleNews-vectors-negative300.bin"
embedding_file_txt = "GoogleNews-vectors-negative300.txt"

question1 = 'question1'
question2 = 'question2'
labels = 'is_duplicate'
test_id = 'test_id'

max_sequence_length = 30
max_words = 200000
embedding_dim = 300

nodes = 100
dropout_rate = .2

####################
# load word vectors
#####################
#logging.basicconfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.info)

word2vec_model = KeyedVectors.load_word2vec_format(embedding_file,binary=True)
print('Found %s word vectors of word2vec' % len(word2vec_model.vocab))

#######################################
# process data
#######################################
# Import
train = pd.read_csv('train_cleaned.csv')
test = pd.read_csv('test_cleaned.csv')

# Convert to string and iterable objects
train.question1=train.question1.astype(str)
train.question2=train.question2.astype(str)

test.question1=test.question1.astype(str)
test.question2=test.question2.astype(str)

texts_1 = train[question1]
texts_2 = train[question2]

test_texts_1 = test[question1]
test_texts_2 = test[question2]

# Tokenize
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts_1 + texts_2)
tokenizer.fit_on_texts(test_texts_1 + test_texts_2)
print("words tokenized")

# Convert into sequence inputs
sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)

test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

# Store index of data
word_index = tokenizer.word_index
print('found %s unique tokens' % len(word_index))

# Pad sequences so that they are the same length
train_1 = pad_sequences(sequences_1, maxlen=max_sequence_length)
train_2 = pad_sequences(sequences_2, maxlen=max_sequence_length)

# Convert the training labels to interger values and store in a list
train_labels = []
for values in train[labels]:
	train_labels.append(int(values))
train_labels = np.array(train_labels)

print('shape of data tensor:', train_1.shape)
print('shape of label tensor:', train_labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)

# Convert test ids into integers and store them in a list
test_ids = []
for values in test[test_id]:
	test_ids.append(int(values))
test_ids = np.array(test_ids)

print("train and test sequence completed")

#############################################
# Map words to corresponding word embeddings
#############################################
print('starting embedding matrix')

num_words = len(word_index)+1

embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
	if word in word2vec_model.vocab:
		embedding_matrix[i] = word2vec_model[word]

#######################################
# Build model
#######################################
print("Starting model creation")
# Make in sequence 1 inputs
sequence_1_input = Input(shape=(max_sequence_length,), dtype='int32',name='sequence_1')

# Embedding layer for seq1
embed_1 = Embedding(num_words,embedding_dim,weights=[embedding_matrix],trainable=False,input_length=max_sequence_length)(sequence_1_input)

# Transform into a single vector
srnn_out_1 = LSTM(nodes,dropout=dropout_rate,recurrent_dropout =dropout_rate)(embed_1)

# Make sequence 2 inputs
sequence_2_input = Input(shape=(max_sequence_length,), dtype='int32',name='sequence_2')

# Embedding layer for seq2
embed_2 = Embedding(num_words,embedding_dim,weights=[embedding_matrix],trainable=False,input_length=max_sequence_length)(sequence_2_input)

srnn_out_2 = LSTM(nodes,dropout=dropout_rate,recurrent_dropout =dropout_rate)(embed_2)

# Combine second input question with first
x = concatenate([srnn_out_1, srnn_out_2])

# Create a two layer network
x = BatchNormalization()(x)
x = Dense(nodes, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(nodes, activation='relu')(x)
x = BatchNormalization()(x)

# Output layer
output = Dense(1, activation='sigmoid', name='output')(x)

# finalize model
model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=output)
print("model finalized")

#######################################
# Compile, Fit, Predict
#######################################
print("Compile model")

# Compile 
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
print("Starting model fit")

# Fit
early_stopping_monitor = EarlyStopping(monitor = 'val_loss',patience=25)
model.fit([train_1,train_2],train_labels, batch_size=2048,shuffle=True, epochs=50, validation_split=0.1,callbacks=[early_stopping_monitor])

# Predict
preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)

# Submit
submit = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})

submit.to_csv("submission.csv",index=False)

# EOF 



