# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:48:32 2017

@author: pegasus
"""
from __future__ import print_function
import numpy as np

def generate_text(model, length, VOCAB_SIZE, ix_to_char):
    ix = [np.random.randint(VOCAB_SIZE)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0,i,:][ix[-1]]=1
        print(ix_to_char[ix[-1]],end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0],1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)
        
def load_data(data_dir, seq_length):
    data = open(data_dir, 'r').read()
    chars = list(set(data))
    VOCAB_SIZE = len(chars)
    
    print('Data length: {} characters'.format(len(data)))
    print('Vocabulary size: {} characters'.format(VOCAB_SIZE))
    
    ix_to_char = {ix:char for ix, char in enumerate(chars)}
    char_to_ix = {char:ix for ix, char in enumerate(chars)}
    
    X = np.zeros((len(data)/seq_length, seq_length, VOCAB_SIZE))
    y = np.zeros ((len(data)/seq_length, seq_length, VOCAB_SIZE))
    for i in range(0, len(data)/seq_length):
        X_sequence = data[i*seq_length:(i+1)*seq_length]
        X_sequence_ix = [char_to_ix[value] for value in X_sequence]
        input_sequence = np.zeros((seq_length, VOCAB_SIZE))
        for j in range(seq_length):
            input_sequence[j][X_sequence_ix[j]]=1.
        X[i] = input_sequence
        
        y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((seq_length, VOCAB_SIZE))
        for j in range(seq_length):
            target_sequence[j][y_sequence_ix[j]]=1.
        y[i] = target_sequence
    return X, y, VOCAB_SIZE, ix_to_char
    