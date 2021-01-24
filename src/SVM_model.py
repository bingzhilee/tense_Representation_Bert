# -*- coding: utf-8 -*-
# Copyright(c) 2009 - present CNRS
# All rights reserved.

import argparse
import bloscpack as bp
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV as gsc
from sklearn.model_selection import RandomizedSearchCV as rsc
import scipy.sparse as sp
import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

parser = argparse.ArgumentParser(
    description='SVM classifier using pre-trained BERT sentence representations  ')

parser.add_argument('--input', type=str, required=True, help='Path to the dataset')
parser.add_argument('--output', type=str, required=True,help="directory to save the output files")
parser.add_argument('--lang', type=str, required=True,help="fr or zh")
args = parser.parse_args()

X_train = bp.unpack_ndarray_from_file(args.input+'/train_X.blp')
Y_train = bp.unpack_ndarray_from_file(args.input+'/train_Y.blp')

X_test = bp.unpack_ndarray_from_file(args.input+'/test_X.blp')
Y_test = bp.unpack_ndarray_from_file(args.input+'/test_Y.blp')

# train the model on training set
model = LinearSVC(class_weight="balanced",random_state=42)
# defining parameter range
params = {"C": [0.0001,0.001,0.01,0.02,0.03,0.04,0.05,0.1]}
tense_clf = gsc(model,params,n_jobs=5,refit = True)

#fitting the model for grid search
tense_clf.fit(X_train,Y_train)

# print best parameter after tuning
print(tense_clf.best_params_)
# print how our model looks after hyper-parameter tuning
print(tense_clf.best_estimator_)


y_pred= tense_clf.predict(X_test)
print(classification_report(Y_test,y_pred))
#plot_confusion_matrix(model,X_test,Y_test,display_labels=['Past','Fut','Pres'],cmap=plt.cm.Blues,normalize='true')


