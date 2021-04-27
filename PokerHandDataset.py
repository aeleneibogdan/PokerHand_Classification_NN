# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 00:56:09 2021

@author: aelen
"""

import numpy as np

import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

datatest=pd.read_csv('poker-hand-testing.data',header=None)
datatrain=pd.read_csv('poker-hand-training-true.data', header=None)

pd.set_option('display.max_columns', None)

columns= ['SuitCard1','RankCard1','SuitCard2','RankCard2','SuitCard3','RankCard3','SuitCard4','RankCard4',
                  'SuitCard5','RankCard5','Player Hand']

datatest.columns=columns
datatrain.columns=columns


classes=['Nothing in Hand','One pair', 'Two pairs','Three of a kind','Straight','Flush','Full house',
                           'Four of a kind','Straight flush','Royal flush']
# [0,1,2,3,4,5,6,7,8,9]
# ['Nothing in Hand','One pair', 'Two pairs','Three of a kind','Straight','Flush','Full house',
                           # 'Four of a kind','Straight flush','Royal flush']
                           
print("Testing DATASET")

print('\n')
print(datatest.head(5))

print('\n')
print(datatest.describe())

print('============================================================================')

print("Training DATASET")

print('\n')
print(datatrain.head(5))

print('\n')
print(datatrain.describe())

# 1 FOR TEST data                 # 2 FOR TRAIN data


X1=datatest.iloc[:,:11]
X2=datatrain.iloc[:,:11]

T1=datatest['Player Hand']
T2=datatrain['Player Hand']

#Testing data
xTrain1, xTest1, tTrain1, tTest1 = train_test_split(X1,T1, test_size=0.2)

#Training data
xTrain2, xTest2, tTrain2, tTest2 = train_test_split(X2,T2, test_size=0.5)

mlp = MLPClassifier(alpha=1e-5,verbose=1,max_iter=500,hidden_layer_sizes=(22,22),random_state=42)

#Training the Testing data
mlp.fit(xTrain1, tTrain1)
#Training the Training data
mlp.fit(xTrain2, tTrain2)

yTrain1 = mlp.predict(xTrain1)
print('Train accuracy for TESTING dataset=', accuracy_score(tTrain1,yTrain1))

yTest1=mlp.predict(xTest1)
print('Test Accuracy for TESTING dataset=',accuracy_score(tTest1,yTest1))

print('Confusion matrix for TESTING dataset')
print(confusion_matrix(tTest1,yTest1))

print("===========================")


yTrain2=mlp.predict(xTrain2)
print('Train accuracy for TRAINING dataset=',accuracy_score(tTrain2,yTrain2))

yTest2=mlp.predict(xTest2)
print('Test accuracy for TRAINING dataset=',accuracy_score(tTest2,yTest2))
print('Confusion matrix for TRAINING dataset=')
print(confusion_matrix(tTest2,yTest2) )
