# -*- coding: utf-8 -*-
"""
In this python script we provided an example of how to use our 
implementation of GB-SMOTE methods to perform classification.

Usage:
```
python demorun.py -data ./dataset/moon_1000_100_2.csv -n 5 
or
python demorun.py -data ./dataset/moon_1000_200_4.csv -n 5
```

run arguments:
    -data : string
    |   Specify a dataset.
    -ker: string
    |   Specify the type of kernel of SVM.
    -n : integer
    |   Specify the number of n-fold cross-validation

"""
import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import preprocessing
import tools.dataprocess as dp
from tools.imbalancedmetrics import ImBinaryMetric
from GB_SMOTE import GB_SMOTE
RANDOM_STATE = 42

def parse():
    '''Parse system arguments.'''
    parse=argparse.ArgumentParser(
        description='General excuting GB-SMOTE', 
        usage='demorun.py -data <datasetpath> -n <n-fold cross-validation>'
        )
    parse.add_argument("-data",dest="dataset",help="the path of a dataset")
    parse.add_argument("-ker",dest="kernel", default="rbf", help="type of the kernel")
    parse.add_argument("-n",dest="n",type=int,default=5,help="n-fold cross-validation")
    return parse.parse_args()

def metric(y,y_pre):
        return ImBinaryMetric(y,y_pre).AP() 

def searchParameter_SVM(X,y,kernel):
    '''
    '''
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2,random_state=RANDOM_STATE)
    #calculating SVM's gamma: gamma=1/theta^2  theta^2=1/N^2*sigma_i=1 ^ N sigma_j=1 ^N (||xi-xj||^2)
    # d=pairwise_distances(X=X,metric="euclidean")
    # gamma=(1.0*d.size)/d.sum()
    # searching SVM's gamma 
    if kernel=='rbf':
        tuned_params = {"gamma": [2**i for i in range(-10,3)],
                        "C" : [2**i for i in range(-10,10)]}
    else:
        tuned_params = {"C" : [2**i for i in range(-10,10)]}
    model = GridSearchCV(SVC(probability=True,kernel=kernel),
                         tuned_params,cv=sss,
                         scoring=make_scorer(metric), n_jobs=-1)
    model.fit(X,y)
    return model.best_params_

def main():
    para = parse()
    kernel=para.kernel
    dataset=para.dataset
    scores = []
    X,y=dp.readDateSet(dataset)
    X=preprocessing.scale(X)
    print(f"Dataset:%s,#attribute:%s [neg pos]:%s\n "%(dataset,X.shape[1],str(np.bincount(y))))
    sss = StratifiedShuffleSplit(n_splits=para.n, test_size=0.2,random_state=RANDOM_STATE)
    fcnt=0
    for train_index, test_index in sss.split(X, y):
        fcnt+=1
        print('{} fold'.format(fcnt))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #search best parameters of SVM classifier to calculate the slack variables by CV
        print("Searching best parameters (i.e., C and gamma) of SVM classifier...")
        best_param=searchParameter_SVM(X_train,y_train,kernel)
        if kernel=="rbf":
            svm=SVC(C=best_param['C'], gamma=best_param['gamma'], probability=True)#best_param['gamma']
        else:
            svm=SVC(C=best_param['C'], kernel="linear", probability=True)
        #search the parameter C of GB-SMOTE by CV
        print("Searching the parameter (i.e., C) of GB-SMOTE...")
        sss2 = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
        tuned_params = {"C" : [2**i for i in range(-10,10)]}
        model = GridSearchCV(GB_SMOTE(clf=svm),tuned_params,cv=sss2, n_jobs=-1)
        model.fit(X_train,y_train)
        best_clf=model.best_estimator_
        #retrain SVM classifier
        best_clf.fit(X_train,y_train)
        #predict
        y_pre=best_clf.predict(X_test)
        y_pred = best_clf.predict_proba(X_test)[:, 1]
        metrics=ImBinaryMetric(y_test,y_pre)
        scores.append([metrics.f1(),metrics.MCC(),metrics.aucprc(y_pred)])
        print('F1:{:.3f}\tMCC:{:.3f}\tAUC-PR:{:.3f}'.format(metrics.f1(),metrics.MCC(),metrics.aucprc(y_pred)))
        print('------------------------------')
    # Print results to console
    print('Metrics:')
    df_scores = pd.DataFrame(scores, columns=['F1','MCC','AUC-PR'])
    for metric in df_scores.columns.tolist():
        print ('{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(), df_scores[metric].std()))
    
if __name__ == '__main__':
    main()
            