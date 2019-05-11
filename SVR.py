# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:23:52 2019

@author: Abdullah kabir
"""

#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Reading the csv
dataset = pd.read_csv('4000_merge_dataset_Final.csv')
#Deviding X-axis And Y-axis
X = dataset.iloc[:, :-1].values
Y= dataset.iloc[:,6].values
#Data Preprocessing Starts
#Data Encoding
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X= LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features= [0])
X=onehotencoder.fit_transform(X).toarray()
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y=Y.reshape(-1,1)
Y= sc_Y.fit_transform(Y)
#Spliting the data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Applying SVR
from sklearn.svm import SVR
Support_regressor= SVR(kernel='rbf', degree=3, coef0=1, tol=0.1, C=1000, gamma=12,
                       epsilon=0.01,cache_size=50, max_iter=-1)
Support_regressor.fit(X_train,y_train)
y_pred=Support_regressor.predict(X_test)
#Reversing the scaled value
y_pred= sc_Y.inverse_transform(y_pred)
y_test= sc_Y.inverse_transform(y_test)
#K fold cross validation
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator = Support_regressor,X= X_train,y=y_train,cv=4)
acuuracy= accuracies.mean()
#Grid Search CV for best parameters and accuracy
from sklearn.model_selection import GridSearchCV
parameters= [{ 'C': [1,10,20,0.5,0.2],'kernel':['linear']},
              {'C': [1,10,20,100,1000],'kernel':['rbf'],
               'epsilon':[1,0.0,0.5,0.1,0.01,0.001,0.0001],
               'gamma':[1000,100,10,1,0.5,0.1,0.01,0.001,0.0001],
               'degree':[3,4]}]
grid_search = GridSearchCV(estimator = Support_regressor,
                           param_grid =parameters,
                           cv=6,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters= grid_search.best_params_

#Calculating r2
from sklearn.metrics import r2_score
R2= r2_score(y_test,y_pred)

#Calculating Rootmean
from sklearn import metrics
rootMean= np.sqrt(metrics.mean_squared_error(y_test, y_pred))

