from multiprocessing.pool import ThreadPool
import numpy as np
import os

train_data, train_labels = list(), list()
test_data, test_labels = list(), list()

# Quick function to get data
def get_data(file):
    with open(file, 'r') as f:
        data = f.readlines()
        return data

# Load in data from each directory and label accordingly
for i, Dir in enumerate(['train/pos','train/neg','test/pos','test/neg']):
    label = (i+1)%2
    os.chdir("/Users/sb/Downloads/aclImdb/"+Dir)
    files = !ls   
    
    # Labels are 1 for positive and 2 for negative
    label = (i+1)%2
    if i<2:
        # Multiprocessing to speed things up
        with ThreadPool(4) as p:
            train_data.extend(p.map(get_data, files))
            train_labels += [label]*len(files) 
    else:
        with ThreadPool(4) as p:
            test_data.extend(p.map(get_data, files))
            test_labels += [label]*len(files) 


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = "post"
oov_token = "<OOV>"


# Fit tokenizer to training data, and sequnce
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_token)
tokenizer.fit_on_texts(train_data)


# Grab and reverse that word index bb
word_index = tokenizer.word_index
index_word = {val:key for key,val in word_index.items()}


with ThreadPool(4) as p:
    train_data = np.array(p.map(tokenizer.texts_to_sequences, train_data)) 
    test_data = np.array(p.map(tokenizer.texts_to_sequences, test_data))


train_data = pad_sequences(train_seq, maxlen=max_length, truncating=trunc_type)
test_data = pad_sequences(test_seq, maxlen=max_length, truncating=trunc_type)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels) 

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_length, return_sequences=True)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.summary()


model.fit(train_data, train_labels, epochs=epochs, validation_data=(test_data, test_labels))


