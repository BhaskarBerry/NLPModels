# -*- coding: utf-8 -*-
#Importing the required packages
#https://www.kaggle.com/omkarsabnis/toxic-comment-classification/data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import check_output
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
#import codec
import keras
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Conv1D,MaxPool1D,Embedding
from keras.layers import LSTM,Bidirectional
from keras.layers import Dense,Input,Flatten,Dropout
#,Merge
from keras.callbacks import EarlyStopping
stopwords_ = set(stopwords.words('english'))
 
#Setting the global Variables
EMBEDDINGDIM  = 300
MAXVOCABSIZE = 175303
MAXSEQLENGTH = 200
batchsize = 256
epochs = 3

#Setting and reading the train data set
traincomments = pd.read_csv(r"C:\Users\bberry\Documents\ReddyNotes\github\NLPModels\textclassification\Data\toxicdata\train.csv",sep = ',',header = 0)
traincomments.columns=['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("Number of train records:", traincomments.shape)

labelnames = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
ytrain = traincomments[labelnames].values

#Reading and setting up the test data
testcomments =  pd.read_csv(r"C:\Users\bberry\Documents\ReddyNotes\github\NLPModels\textclassification\Data\toxicdata\test.csv",sep = ',',header = 0)
print("Number of test records:", testcomments.shape)

#Cleaning up the text
"""
Function to clean up the text
which need to 2 parameters to be passed
"""
def standardizetext(df,textfield):
    df[textfield] = df[textfield].str.replace(r"http\S+", "")
    df[textfield] = df[textfield].str.replace(r"http+", "")
    df[textfield] = df[textfield].str.replace(r"@\S+", "")
    df[textfield] = df[textfield].str.replace(r"[^A-Za-z0-9,!?@\'\`\"\_\n]", " ")
    df[textfield] = df[textfield].str.replace(r"@", "at")
    df[textfield] = df[textfield].str.lower()
    return df

#Cleaning the train _data and making the new csv file --> train_clean_data.csv
traincomments.fillna('_NA_')
traincomments = standardizetext(traincomments, "comment_text")
traincomments.to_csv(r"C:\Users\bberry\Documents\ReddyNotes\github\NLPModels\textclassification\Data\toxicdata\traincleandata.csv")

#Cleaning the test _data and making the new csv file --> test_clean_data.csv
testcomments.fillna('_NA_')
testcomments = standardizetext(testcomments,"comment_text")
testcomments.to_csv(r"C:\Users\bberry\Documents\ReddyNotes\github\NLPModels\textclassification\Data\toxicdata\testcleandata.csv")

#Tokenizing the text
tokenizer = RegexpTokenizer(r'\w+')
cleantraincomments = pd.read_csv(r"C:\Users\bberry\Documents\ReddyNotes\github\NLPModels\textclassification\Data\toxicdata\traincleandata.csv")
cleantraincomments['comment_text'] = cleantraincomments['comment_text'].astype(str)
cleantraincomments.dtypes
cleantraincomments["tokens"] = cleantraincomments["comment_text"].apply(tokenizer.tokenize)

#delete the stop words
cleantraincomments["tokens"] = cleantraincomments["tokens"].apply(lambda vec: [word for word in vec if word not in stopwords_])
cleantraincomments.head()

cleantestcomments = pd.read_csv(r"C:\Users\bberry\Documents\ReddyNotes\github\NLPModels\textclassification\Data\toxicdata\testcleandata.csv")
cleantestcomments['comment_text'] = cleantestcomments['comment_text'].astype(str)
cleantestcomments.dtypes
cleantestcomments["tokens"] = cleantestcomments["comment_text"].apply(tokenizer.tokenize)
cleantestcomments["tokens"] = cleantestcomments["tokens"].apply(lambda vec: [word for word in vec if word not in stopwords_]) 
cleantestcomments.head()


   
    



















