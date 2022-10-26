# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 14:21:43 2020
@author: Jinjun Ren
"""
import numpy as np
import pandas as pd

def getClassinfo(labels):
    '''
    Count the number of each class
    '''
    ele,cnt=np.unique(labels,return_counts=True)
    if(np.min(cnt)==np.max(cnt)):#Both classes have the same number of samples.
        pcnt=cnt[0]
        ncnt=cnt[0]
        pos_label=ele[0]
        neg_label=ele[1]
    else:
        pcnt=np.min(cnt)
        ncnt=np.max(cnt)
        pos_label=ele[np.argmin(cnt)]#the label of the positive class
        neg_label=ele[np.argmax(cnt)]#he label of the negative class
    return pcnt,pos_label,ncnt,neg_label

def reSetlabel(label):
    """
    Reset the labels of both classes, i.e., positive class is '1' and the negative class is '0'
    """
    p_cnt,p_lab,n_cnt,n_lab=getClassinfo(label)
    label[label==n_lab]=0
    label[label==p_lab]=1
    return label

def readDateSet(filename):
    '''
    Read the file "filename" and return a dataset by means of X and y, where X denotes the samples and y denotes their labels.
    Note:the last column of "filename" is labels.
    '''
    data=pd.read_csv(filename,delim_whitespace=False ,sep=',',encoding='gbk',header=None)
    X=np.array(data.iloc[:,0:-1])
    y=data.iloc[:,-1]
    y=np.array(y,dtype=np.int32)
    if {0,1}!=set(y):#if the labels donot belong to the set {0，1}, then reset label using function reSetlabel.
        y=reSetlabel(y.reshape(len(y),1))
    return X,y.ravel()
