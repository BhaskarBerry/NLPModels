# -*- coding: utf-8 -*-
#Import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv(r"C:\ml_repo\textclassification\Data\Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

#cleaning the texts
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus_data = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus_data.append(review)

#Create the bag of words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer( max_features= 1500)
X = cv.fit_transform(corpus_data).toarray()
y = dataset.iloc[:,-1].values

#splitting the dataset into the Training and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)    

#fitting Naive Bayes on the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#predicting the test result
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


# Accuracy of .73 not bad for the 1000 data if the data is more then the accuracy might be improved
