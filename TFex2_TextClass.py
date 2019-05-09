#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:25:32 2019

@author: Michele
"""
# script to classify movie reviews in the  IMDB dataset
# reviews have been converted to sequences of integers (each number corresponds to a word)


from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(keras.__version__)
print(np.__version__)

# load dataset and divide it into train and test sets 
#  num_words=10000 keeps the top 10,000 most frequently occurring words


(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=10000)

# labels: 1->positive review ; 0-> negative review 

# check size of data 

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# take a look at the format of observations: list of numbers associated to words (each element of the list is a review of a user)

print(train_data[0])

# convert the integers back to words

# A dictionary mapping words to an integer index

word_index = keras.datasets.imdb.get_word_index()

word_index

# first indices are reserved

word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# switch numbers with words: now numbers are keys and words the elements of the dictionary 
# dict() transform a list of tuples into a dictionary 

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# function to decode the review from the list of numbers 

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0])

# arrays of integers—must be converted to tensors before fed into NN
# plus movie reviews must be the same length to be passed as input to the same first layer of NN
# use pad function of keras

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(len(train_data[0]),len(train_data[1]))

print(train_data[0])

# notice that absent words (to match imposed length of the review) are zeros: the value we passed to pad_sequences

# now we can initialize the model: number of layers and number of hidden layers per each layer 

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())

# number of hidden layers

model.add(keras.layers.Dense(16, activation=tf.nn.relu))

# one final layer: 1 or 0

model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# A model needs a loss function and an optimizer for training: binary_crossentropy

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

# create a validation set 

 x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train data set is divided into batches (subsets) and the NN is fed with each of them for an epoch number of times

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# model evaluation 

results = model.evaluate(test_data, test_labels)

print(results)

# plot accuracy and loss over time, use history object of model.fit that contains info on training phase 

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# notice that validation set was used to avoid overfitting 

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# predict 

predictions = model.predict(test_data)

plt.hist(predictions,histtype='bar', alpha=0.5, label='predicted')
plt.hist(test_labels,histtype='bar',   alpha=0.5, label='true')
plt.legend(loc='upper right')

# transform predictions into 0 or 1 putting a threshold 

binpred = list(map(lambda x: 0 if x<0.5 else 1, predictions))

# plot histogram of residuals 
plt.figure(1)
h1 = plt.hist(binpred,histtype='bar', alpha=0.5, label='predicted')
h2 = plt.hist(test_labels,histtype='bar',   alpha=0.5, label='true')
plt.legend(loc='upper right')

plt.figure(2)
diff=plt.bar([0,1,2,3,4,5,6,7,8,9],height=(h1[0]-h2[0]), edgecolor='black', 
             linewidth=1.2, color='red',width = 1, align = 'edge') 
plt.title("tr - pred")


