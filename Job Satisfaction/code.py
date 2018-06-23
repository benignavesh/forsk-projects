# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 22:37:41 2018

@author: verma
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('job_satisfaction.csv')

#Visualizing why people left? -------------------------------------------------
 
df_left = df.loc[df['left'] == 1]
for i in df:
    ax = df_left[[i]].plot(kind='bar', title ="Satisfaction level of employees", legend=True, fontsize=12)
    ax.axes.get_xaxis().set_ticks([]) # to remove the x axis ticks
    plt.show()
#------------------------------------------------------------------------------
    
#Predicting which valuable employee will leave next
features = df.iloc[:,[0,1,2,3,4,5,7,8,9]].values
labels = df.iloc[:,[6]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
features[:,7] = labelencoder.fit_transform(features[:,7])
features[:,8] = labelencoder.fit_transform(features[:,8])

# Performing OneHotEncoding on the sales
onehotencoder = OneHotEncoder(categorical_features = [7])
features = onehotencoder.fit_transform(features).toarray()
features = features[:,1:]
# Performing OneHotEncoding on the salary
onehotencoder1 = OneHotEncoder(categorical_features = [16])
features = onehotencoder1.fit_transform(features).toarray()
features = features[:,1:]

#Splitting into test and train
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test =train_test_split(features,labels,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

#Using Logistic regression
from sklearn.linear_model import LogisticRegression
classifier3 = LogisticRegression()
classifier3.fit(features_train, labels_train)

logist_pred = classifier3.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, logist_pred)
Score_logist = classifier3.score(features_test, labels_test)
'''
78.78% using Logistic Regression
'''
#Knn algo
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,p=2)
classifier.fit(features_train, labels_train)

pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, pred)

Score = classifier.score(features_test, labels_test)
'''
94% using K-NN algorithm
'''
#Using decision tree classifier
from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier2.fit(features_train,labels_train)

tree_pred = classifier2.predict(features_test)


from sklearn.metrics import confusion_matrix
cm_tree = confusion_matrix(labels_test,tree_pred)
Score_tree = classifier2.score(features_test, labels_test)
'''
98.13% using Decision Tree Classifier
'''

#using random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators =10, criterion = 'entropy', random_state = 0)
classifier1.fit(features_train,labels_train)

forest_pred = classifier1.predict(features_test)
from sklearn.metrics import confusion_matrix
cm_forest = confusion_matrix(labels_test, forest_pred)

Score_forest = classifier1.score(features_test, labels_test)
'''
99.04% using Random Forest Classifier
'''
#using SVM
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(features_train, labels_train)
svm_predictions = svm_model_linear.predict(features_test)
 
# creating a confusion matrix
cm = confusion_matrix(labels_test, svm_predictions)
# model accuracy for X_test  
Score_svm = svm_model_linear.score(featues_test, labels_test)
 
