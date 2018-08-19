# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:25:55 2018

@author: rehab
"""

import nltk 
import random 
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB ,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC

documents=[(list(movie_reviews.words(fileid)),category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]


random.shuffle(documents)
all_words=[w.lower() for w in movie_reviews.words()]
all_words=nltk.FreqDist(all_words)

word_feature=list(all_words)[:3000]
def find_feature(document):
    words=set(document)
    feature={}
    for w in word_feature:
        feature[w]=(w in words)
        
    return feature

feature_set=[(find_feature(rev),category)for (rev,category) in documents]  
training_set=feature_set[:1900]
testing_set=feature_set[1900:]
  
classifier=nltk.NaiveBayesClassifier.train(training_set)
print('NaiveBayesClassifier Accuracy',nltk.classify.accuracy(classifier,testing_set))

classifier_list=[MultinomialNB(),BernoulliNB(),LogisticRegression(),SGDClassifier(),SVC()]
for i in classifier_list:
    classifier=SklearnClassifier(i)
    classifier.train(training_set)
    print(str(i).split('(')[0],'Accuracy',nltk.classify.accuracy(classifier,testing_set))
