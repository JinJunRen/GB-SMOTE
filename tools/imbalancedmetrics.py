# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:05:39 2020

@author: Jinjun Ren
"""
import numpy as np
from sklearn.metrics import (
    f1_score, 
    precision_recall_curve, 
    auc,
    recall_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score
    )
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class ImBinaryMetric:
    def __init__(self,y_true,y_pre):
        self.y_true=np.array(y_true).astype(int)
        self.y_pred=np.array(y_pre).astype(int)
        self.conf_m=confusion_matrix(self.y_true,self.y_pred)
        self.TN=self.conf_m[0,0]
        self.TP=self.conf_m[1,1]
        self.FP=self.conf_m[0,1]
        self.FN=self.conf_m[1,0]
        
    
    def recall(self):
        return recall_score(self.y_true,self.y_pred)
    
    def precision(self):
        return precision_score(self.y_true,self.y_pred)
    
    def f1(self):
        return f1_score(self.y_true,self.y_pred)
    
    def MCC(self):
        '''Compute optimal MCC score.'''
        mccs = []
        for t in range(100):
            y_pred_b = self.y_pred.copy()
            y_pred_b[y_pred_b < 0+t*0.01] = 0
            y_pred_b[y_pred_b >= 0+t*0.01] = 1
            mcc = matthews_corrcoef(self.y_true, y_pred_b)
            mccs.append(mcc)
        return max(mccs)
    
    def AP(self):
        return average_precision_score(self.y_true,self.y_pred)

    def aucprc(self,y_preprob,pic=False):
        y=self.y_true.copy()
        precision, recall, thresholds = precision_recall_curve(y, np.array(y_preprob))
        if pic:
            plt.plot(precision, recall)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision Recall')
            plt.show()
        return auc(recall, precision)
    
        
    def displaymetric(self):
        return str(self.gmean())+"\t"+str(self.recall())+"\t"+str(self.precision())+"\t"+str(self.f1())+"\t"+str(self.MCC())
        
    
if __name__ == '__main__':
    data=np.array([[0,0.356,0.468],
                   [0,0.374,0.498],
                   [0,0.382,0.476],
                   [1,0.342,0.48],
                   [0,0.366,0.51],
                   [0,0.426,0.454],
                   [1,0.432,0.48],
                   [0,0.416,0.464],
                   [1,0.438,0.44],
                   [1,0.444,0.464]
                   ])
    test=np.array([[0,0.356,0.494],
                   [0,0.336,0.508],
                   [1,0.336,0.464],
                   [1,0.444,0.464]
                   ])
    X=data[:,1:]
    y=data[:,0]
    test_X=test[:,1:]
    test_y=test[:,0]
    clf = LinearDiscriminantAnalysis()
    clf.fit(X,y)
    pre_y=clf.predict(test_X)
    prepro_y=clf.predict_proba(test_X)[:,1]
    m=ImBinaryMetric(test_y,pre_y)
    print(m.displaymetric())
    print(m.aucprc(prepro_y))
    
    
    
        
        
    