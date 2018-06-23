# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 00:10:52 2018

@author: verma
"""
import pandas as pd
from sklearn.metrics import accuracy_score

#from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

features_train = train.iloc[:,:-2].values
labels_train = train.iloc[:,[562]].values

features_test = test.iloc[:,:-2].values
labels_test = test.iloc[:,[562]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
labels_train[:,0] = labelencoder.fit_transform(labels_train[:,0])
labels_test[:,0] = labelencoder.fit_transform(labels_test[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])
labels_train = onehotencoder.fit_transform(labels_train).toarray()
labels_test = onehotencoder.fit_transform(labels_test).toarray()
labels_train = labels_train[:,1:]
labels_test = labels_test[:,1:]

# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(features_train, labels_train)
dtree_predictions = dtree_model.predict(features_test)
Score_tree = dtree_model.score(features_test, labels_test)
#------------------------------------------------------------------------------
# Using naive Bayes

from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier_nb = LabelPowerset(GaussianNB())

# train
classifier_nb.fit(features_train, labels_train)

# predict
predictions_nb = classifier_nb.predict(features_test)
score_nb= accuracy_score(labels_test,predictions_nb)
#------------------------------------------------------------------------------
#using random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators =10, criterion = 'entropy', random_state = 0)
classifier1.fit(features_train,labels_train)

forest_pred = classifier1.predict(features_test)

Score_forest = classifier1.score(features_test, labels_test)
#------------------------------------------------------------------------------
# Using base Classifier with single-class SVM
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC

# initialize Binary Relevance multi-label classifier
# with an SVM classifier
# SVM in scikit only supports the X matrix in sparse representation

classifier_br = BinaryRelevance(classifier = SVC(), require_dense = [False, True])

# train
classifier_br.fit(features_train, labels_train)

# predict
predictions_br = classifier_br.predict(features_test)
score_br= accuracy_score(labels_test,predictions_br)
#------------------------------------------------------------------------------
#Knn algo
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors=5,p=2)
classifier_knn.fit(features_train, labels_train)

pred_knn = classifier_knn.predict(features_test)
Score_knn = classifier_knn.score(features_test, labels_test)

#------------------------------------------------------------------------------

#using Multilabel KNN Algorithm
from skmultilearn.adapt import MLkNN
mlknn_model = MLkNN(k=20)
# train
mlknn_model.fit(features_train, labels_train)
# predict
predictions = mlknn_model.predict(features_test)

score_mlknn= accuracy_score(labels_test,predictions)

'''
PERFORMING FEATURE SELECTION
'''


from sklearn.decomposition import PCA
pca = PCA(n_components=388)
fit_train = pca.fit(features_train)
fit_test = pca.fit(features_test)
features_train_new = pca.transform(features_train)
features_test_new = pca.transform(features_test)

#using Multilabel KNN Algorithm
from skmultilearn.adapt import MLkNN
mlknn_model_new = MLkNN(k=20)
# train
mlknn_model_new.fit(features_train_new, labels_train)
# predict
predictions_new = mlknn_model_new.predict(features_test_new)

score_mlknn_new= accuracy_score(labels_test,predictions_new)

# After performing PCA, we get that reducing no of features to 388 doesn't make
# difference in the score.


