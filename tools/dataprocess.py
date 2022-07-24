# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 14:21:43 2020
对数据集进行读取、统计和加工
@author: wlkx
"""
import numpy as np
import pandas as pd

def getClassinfo(label):
    '''
    参数：标签y列表(默认两类)
    功能：计算各类别的个数和相应的标签
    返回：正类数，负类数，正类标签，负类标签
    '''
    ele,cnt=np.unique(label,return_counts=True)
    if(np.min(cnt)==np.max(cnt)):#两个样本一样多
        pcnt=cnt[0]
        ncnt=cnt[0]
        pos_label=ele[0]
        neg_label=ele[1]
    else:
        pcnt=np.min(cnt)
        ncnt=np.max(cnt)
        pos_label=ele[np.argmin(cnt)]#正类，样本数小的类
        neg_label=ele[np.argmax(cnt)]#负类
    return pcnt,pos_label,ncnt,neg_label

def reSetlabel(label):
    """
    功能：重新设置类别标签，大类为0，小类为1
    """
    p_cnt,p_lab,n_cnt,n_lab=getClassinfo(label)
    label[label==n_lab]=0
    label[label==p_lab]=1
    return label

#读取数据集以X,y的形式返回
def readDateSet(filename):
    '''
    参数：filename指文件名
    功能：读取指定文件中的数据，其中样本标签默认为最后一列。
    返回：格式X(数据),y(标签)
    '''
    data=pd.read_csv(filename,delim_whitespace=False ,sep=',',encoding='gbk',header=None)
    X=np.array(data.iloc[:,0:-1])
    y=data.iloc[:,-1]
    y=np.array(y,dtype=np.int32)
    if {0,1}!=set(y):#类标签不是{0，1}时，重新设置大类的标签为0，小类的标签为1
        y=reSetlabel(y.reshape(len(y),1))
    return X,y.ravel()

#读取数据集以X,y的形式返回
def readDateSet_Nochangelabel(filename):
    '''
    参数：filename指文件名
    功能：读取指定文件中的数据，其中样本标签默认为最后一列。
    返回：格式X(数据),y(标签)
    '''
    data=pd.read_csv(filename,delim_whitespace=False ,sep=',',encoding='gbk',header=None)
    X=np.array(data.iloc[:,0:-1])
    y=data.iloc[:,-1]
    y=np.array(y,dtype=np.int32)
    return X,y.ravel()
