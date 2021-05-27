# -*- coding: utf-8 -*-
"""
Author: Spencer Baker
Date: 5/24/2021
Notes: Different classification algorithms for Milestone 2B
    Inputs - change of impedance
    Output - Exercise classification
    Algorithms - 
        K-Nearest Neighbor
        Logistic Regression
        Linear Discriminate Analysis
        Quadratic Discriminate Analysis
        Bagging
        Random Forests
        Boosting
        Suport Vector Machines
"""

# --- INITIALIZE --- #
# clear console


# import tools
import pandas as pd
# import numpy as np
from tqdm import tqdm

# import models
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# import evaluations
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# import scipy
# import matplotlib.pyplot as plt
# import urllib
# import joblib
# from pylab import rcParams



# --- LOAD DATA --- #
# data = pd.read_csv('2B_Range_Data.csv')
data = pd.read_csv('2B_Peak_Valley_Data.csv')
X = data.drop(columns = 'Ex')
y = data['Ex']


# --- ALGORITHMS AND EVALUATION --- #
# number of iterations
N = 1000  #1000

# scores
k_neighbor_score = 0
LR_ovo_score = 0
LDA_score = 0
QDA_score = 0
tree_score = 0
Bag_score = 0
RF_score = 0
Boost_score = 0
SV_score = 0


for i in tqdm(range(N)): # for i in range(0,N):
    
    # Randomly split training, testing data
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.3)
    
    
    # K-Nearest Neighbor
    k = 1
    KNN_algorithm = KNeighborsClassifier(n_neighbors=k)
    KNN_algorithm.fit(X_train, y_train)
    KNN_yhat = KNN_algorithm.predict(X_test)
    k_neighbor_score += accuracy_score(y_test, KNN_yhat)
    
    
    # Logistic Regression (one versus one or one versus all?)
    LR_ovo_algorithm = LogisticRegression(max_iter = 100000)    # 'multi_class',‘auto’, ‘ovr’, ‘multinomial’})
    LR_ovo_algorithm.fit(X_train, y_train)
    LR_ovo_yhat = LR_ovo_algorithm.predict(X_test)
    LR_ovo_score += accuracy_score(y_test, LR_ovo_yhat)
    
    
    # Linear Discriminate Analysis
    LDA_algorithm = LinearDiscriminantAnalysis()
    LDA_algorithm.fit(X_train, y_train)
    LDA_yhat = LDA_algorithm.predict(X_test)
    LDA_score += accuracy_score(y_test, LDA_yhat)
    
    
    """
    # Quadratic Discriminate Analysis
    QDA_algorithm = QuadraticDiscriminantAnalysis()
    QDA_algorithm.fit(X_train, y_train)
    QDA_yhat = QDA_algorithm.predict(X_test)
    QDA_score += accuracy_score(y_test, QDA_yhat)
    """
    
    
    # Trees
    tree_algorithm = DecisionTreeClassifier()
    tree_algorithm.fit(X_train, y_train)
    tree_yhat = tree_algorithm.predict(X_test)
    tree_score += accuracy_score(y_test, tree_yhat)
    

    # Bagging
    Bag_algorithm = BaggingClassifier()
    Bag_algorithm.fit(X_train, y_train)
    Bag_yhat = Bag_algorithm.predict(X_test)
    Bag_score += accuracy_score(y_test, Bag_yhat)
    
    
    # Random Forests
    RF_algorithm = RandomForestClassifier()
    RF_algorithm.fit(X_train, y_train)
    RF_yhat = RF_algorithm.predict(X_test)
    RF_score += accuracy_score(y_test, RF_yhat)
    
    
    # Boosting
    # B = 1
    min_samples = 10
    # d = 1
    L = 1e-1
    Boost_algorithm = GradientBoostingClassifier(learning_rate=L, min_samples_split=min_samples)
    Boost_algorithm.fit(X_train, y_train)
    Boost_yhat = Boost_algorithm.predict(X_test)
    Boost_score += accuracy_score(y_test, Boost_yhat)
    
    
    # Support Vector Machines
    SV_algorithm = SVC(C=250)
    SV_algorithm.fit(X_train, y_train)
    SV_yhat = SV_algorithm.predict(X_test)
    SV_score += accuracy_score(y_test, SV_yhat)
    



# --- EVALUATION --- #
# scores
print('\n')
print('k-neighbors', '\t', '\t', '\t', 100*k_neighbor_score/N)
print('Logistic Regression', '\t', '\t', 100*LR_ovo_score/N)
print('Linear Discriminate Analysis', '\t', 100*LDA_score/N)
# print('Quadratic Discriminate Analysis', '\t', 100*QDA_score/N)
print('Decision Tree Classifier', '\t', 100*tree_score/N)
print('Bagging Classifier', '\t', '\t', 100*Bag_score/N)
print('Random Forest', '\t', '\t', '\t', 100*RF_score/N)
print('Boosting', '\t', '\t', '\t', 100*Boost_score/N)
print('Support Vector Machine', '\t', '\t', 100*SV_score/N)

# confusion matrix

