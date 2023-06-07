import pandas as pd
import random
import os
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

def tr_te_split(data):
    # Split data
    df_train = data.iloc[:int(len(data)*0.8), :]
    df_test = data.iloc[int(len(data)*0.8):, :]

    return df_train, df_test


def undersampling(data):
    reduced_labels = []
    reduced_text = []
    for i in range(data.shape[0]):
        text, label = data.text[i].split(' '), data.labels[i].split(' ')
        s_t = ''
        s_l = ''
        for t, l in zip(text, label):
            if l == 'O' and random.random() < 0.8:
                continue
            s_t += t+' '
            s_l += l+' '
        reduced_labels.append(s_l)
        reduced_text.append(s_t)
        
    df = pd.DataFrame({'text' : reduced_text, 'labels' : reduced_labels})
    df.to_csv('undersample.csv', index = False)
    return df
    

def massage(data, max_length):
    lst = []
    for i, j in enumerate(data.labels):
        if len(data.labels[i].split(' ')) >= max_length:
            lst.append(i)
    df = data[~data.index.isin(lst)].reset_index(drop=True) # max_length 이상의 길이를 가지는 데이터는 제외하고 학습
    
    return df


def tokenize_and_preserve_labels(tokenizer, sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def push_lst(ary):
    np.save('log/{}.npy'.format('ary'), ary)