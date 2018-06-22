# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:23:08 2018

@author: verma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('demonetisation.csv' , parse_dates=['created'])

data_sort_retweet = data.sort_values(by='retweetCount' , ascending = False)
data_sort_retweet.head()

data_sort_fav = data.sort_values(by='favoriteCount' , ascending = False)
data_sort_fav.head()

features = data.iloc[:,[3,10,15]].values
labels = data.iloc[:,[6]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
features[:,2] = labelencoder.fit_transform(features[:,2])
labels[:,0] = labelencoder.fit_transform(labels[:,0])

onehotencoder_label = OneHotEncoder(categorical_features = [0])
labels = onehotencoder_label.fit_transform(labels).toarray()
labels = labels[:,1,] #dummy trap

from sklearn.model_selection import train_test_split 
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.fit_transform(features_test)


#-----------------Logistic Regression------------------------------------------
from sklearn.linear_model import LogisticRegression
classifier_logist = LogisticRegression()
classifier_logist.fit(features_train,labels_train)

labels_pred_logist=classifier_logist.predict(features_test)

from sklearn.metrics import confusion_matrix
cm_logist = confusion_matrix(labels_test, labels_pred_logist)
Score_logist = classifier_logist.score(features_test, labels_test)

#------------------------------------------------------------------------------

#-------------------------KNN Algorithm----------------------------------------
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=5,p =2)
classifier_knn.fit(features_train, labels_train)

pred_knn = classifier_knn.predict(features_test)

from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(labels_test, pred_knn)

Score_knn = classifier_knn.score(features_test, labels_test)

#------------------------------------------------------------------------------

#-----------------Decision Tree Classifier ------------------------------------

from sklearn.tree import DecisionTreeClassifier
classifier_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_tree.fit(features_train,labels_train)

pred_tree = classifier_tree.predict(features_test)


from sklearn.metrics import confusion_matrix
cm_tree = confusion_matrix(labels_test,pred_tree)
Score_tree = classifier_tree.score(features_test, labels_test)

#------------------------------------------------------------------------------

#------------------------Random Forest Classifier------------------------------

from sklearn.ensemble import RandomForestClassifier
classifier_forest = RandomForestClassifier(n_estimators =10, criterion = 'entropy', random_state = 0)
classifier_forest.fit(features_train,labels_train)

pred_forest = classifier_forest.predict(features_test)

from sklearn.metrics import confusion_matrix
cm_forest = confusion_matrix(labels_test, pred_forest)
Score_forest = classifier_tree.score(features_test, labels_test)

#------------------------------------------------------------------------------