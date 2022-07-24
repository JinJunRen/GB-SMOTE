# -*- coding: utf-8 -*-
"""
Created on Jun. 01  2022
@author: Jinunren
mailto: jinjunren@lzufe.edu.cn
"""

import numpy as np
import sys
sys.path.append("..")
from tools.imbalancedmetrics import ImBinaryMetric
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

class GB_SMOTE(BaseEstimator, ClassifierMixin): 
    def __init__(self,clf=None, C=100):
        if clf == None:
            self.clf=SVC(probability=True)
        else :
            self.clf=clf
        self.runtime=0
        self.C=C
        
    @classmethod
    def metric(self,y,y_pre):
        return ImBinaryMetric(y,y_pre).AP() 
    
    def replaceLable(self, y, neg_label=0, pos_label=1):
        '''
		if y_i is the positive class then we assigin to +1 to it, and
		if y_i is the negative class then we assigin to +1 to it.
        '''
        newy=np.copy(y)
        newy[newy==neg_label]=-1.0 
        newy[newy==pos_label]=1.0    
        return newy
    
    def selectInstances(self,X,in_margin,pos_safe,n):
        '''
		Select n sample-pairs from #M^+ and S^+, respectively, and create a synthesized sample set 
		Note that: the elements in in_margin and pos_safe are the indexs of samples in X.
        '''
        left_flag=right_flag=False
        # print(f"in_margin:{in_margin},pos_safe:{pos_safe}")
        if len(in_margin)!=0:
            fun_p_dist=self.clf.decision_function(X[in_margin,:])
            fun_p=np.power(np.e,np.abs(fun_p_dist))
            pro_p=fun_p/np.sum(fun_p) #The probability of the selection in S^+
        else:
            left_flag=True
        if len(pos_safe)!=0:
            fun_q_dist=self.clf.decision_function(X[pos_safe,:])
            fun_q=np.power(np.e,-1*np.abs(fun_q_dist))
            pro_q=fun_q/np.sum(fun_q)
        else:
            right_flag=True
        if left_flag:#if E^+ is empty，copying S^+ to E^+ 
            in_margin=np.copy(pos_safe)
            pro_p=np.copy(pro_q)
        if right_flag: 
            pos_safe=np.copy(in_margin)
            pro_q=np.copy(pro_p)
        p=list()
        q=list()
        delta=np.zeros(n)
        for i in range(n):
            pro_total=0
            m=np.random.rand()
            selectid=0
            for index,temp in enumerate(pro_p):
                pro_total=pro_total+temp
                if pro_total>m:
                    selectid=index
                    break
            p.append(in_margin[selectid])
            pro_total=0
            m=np.random.rand()
            selectid=0
            for index,temp in enumerate(pro_q):
                pro_total=pro_total+temp
                if pro_total>m:
                    selectid=index
                    break
            q.append(pos_safe[selectid])
        link_pq=np.array([str(a)+"+"+str(b) for a,b in zip(p,q)])
        element,cnt=np.unique(link_pq,return_counts=True)

        for i,ele in enumerate(element):
            t_delta=np.zeros(cnt[i])
            t_delta=t_delta+1.0/(cnt[i]*2)
            step= 1.0/cnt[i]         
            for j in range(1,cnt[i]):
                t_delta[j]= t_delta[j]+step*j
            delta[link_pq==ele]=t_delta        
        return np.array(p),np.array(q),delta
    
    def fit(self,X,y):
        self.X=X.copy()
        self.y=y.copy()
        classnum=np.bincount(y)
        self.newsamplenum=np.abs(classnum[0]-classnum[1])
        self.clf.fit(X,y)
        pe,pm,ps,ne,nm,ns=self.partitionInstance(X,self.replaceLable(y))#groupying all samples
        if len(pm) <= 0 and len(ps) <= 0:
            print("ps and pm are all empty!")
            self.p = np.random.choice(pe, self.newsamplenum, replace =True)
            self.q = np.random.choice(pe, self.newsamplenum, replace =True)
            self.delta = np.random.rand(self.newsamplenum)
        else:
            self.p,self.q,self.delta=self.selectInstances(X,pm,ps,self.newsamplenum)
        self.new_clf=None
        return self  
     
    def _predict(self,test_X):
        X_len=len(self.X)
        testX_len=len(test_X)
        dim=X_len+testX_len
        all_X=np.vstack((self.X,test_X))
        if self.clf.kernel=='rbf':
            kernelmatrix=rbf_kernel(all_X,gamma=self.clf.gamma)
        else:
            kernelmatrix=all_X.dot(all_X.T)
        if len(self.p)>0:
            new_KM,new_testX=self.augmentKernelMatrix(X_len,dim,kernelmatrix,self.newsamplenum)
        else:
            new_KM=kernelmatrix
            new_testX=kernelmatrix[X_len:,:]
        new_clf=SVC(C=self.C,probability=True)
        new_clf.kernel="precomputed"
        new_train_y=np.vstack((self.y.reshape(X_len,1),np.ones((self.newsamplenum,1))))
        new_clf.fit(new_KM,new_train_y.ravel())
        y_pre=new_clf.predict(new_testX)
        y_preb=new_clf.predict_proba(new_testX)
        return y_pre,y_preb
        
    def predict(self,test_X):
        y_pre,_=self._predict(test_X)
        return y_pre
    
    def predict_proba(self,test_X):
        _,y_prepro=self._predict(test_X)
        return y_prepro 
    def decision_function(self,test_X):
        X_len=len(self.X)
        testX_len=len(test_X)
        dim=X_len+testX_len
        all_X=np.vstack((self.X,test_X))
        if self.clf.kernel=='rbf':
            kernelmatrix=rbf_kernel(all_X,gamma=self.clf.gamma)
        else:
            kernelmatrix=all_X.dot(all_X.T)
        if len(self.p)>0:
            new_KM,new_testX=self.augmentKernelMatrix(X_len,dim,kernelmatrix,self.newsamplenum)
        else:
            new_KM=kernelmatrix
            new_testX=kernelmatrix[X_len:,:]
        new_clf=SVC(C=self.C,probability=True)
        new_clf.kernel="precomputed"
        new_train_y=np.vstack((self.y.reshape(X_len,1),np.ones((self.newsamplenum,1))))
        new_clf.fit(new_KM,new_train_y.ravel())
        return new_clf.decision_function(new_testX)
        
    def calcKxi(self,X,y):
        '''
		calcuate the slack variables of samples in D 
        '''
        Kxi=1-y*self.clf.decision_function(X)
        Kxi[Kxi<0]=0
        pos_kxi=np.intersect1d(np.where(Kxi>0)[0],np.where(y>0)[0])
        neg_kxi=np.intersect1d(np.where(Kxi>0)[0],np.where(y<0)[0])
        return Kxi,pos_kxi,neg_kxi

    def partitionInstance(self,X,y):
        '''
		partition each class into three sets, e.g., error set, margin set and safe set.
        '''
        kxi,pos_kxi_index,neg_kxi_index=self.calcKxi(X,y)
        pos_class=np.intersect1d(np.where(y>0)[0],np.where(y>0)[0])
        pos_err=pos_kxi_index[kxi[pos_kxi_index].reshape(-1)>=1]
        pos_within_margin=pos_kxi_index[kxi[pos_kxi_index].reshape(-1)<1]
        pos_safe=np.setdiff1d(pos_class,pos_kxi_index)
        neg_class=np.intersect1d(np.where(y<0)[0],np.where(y<0)[0])
        neg_err=neg_kxi_index[kxi[neg_kxi_index].reshape(-1)>=1]
        neg_within_margin=neg_kxi_index[kxi[neg_kxi_index].reshape(-1)<1]
        neg_safe=np.setdiff1d(neg_class,neg_kxi_index)
        return pos_err,pos_within_margin,pos_safe,neg_err,neg_within_margin,neg_safe

    def augmentKernelMatrix(self,X_len,dim,kernelmatrix,n):
        p=self.p
        q=self.q
        delta=self.delta
        deltamatrix=delta*np.ones((dim,n))
        all_K2=(1-deltamatrix)*kernelmatrix[:,p]+deltamatrix*kernelmatrix[:,q]
        #obtain the K2
        K2=all_K2[0:X_len,:]
        K3=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                de1=delta[i]
                de2=delta[j]
                if i==j and self.clf.kernel=='rbf':
                    K3[i,j]=1
                else:
                    K3[i,j]=(1-de1)*(1-de2)*kernelmatrix[p[i],p[j]]+(1-de1)*(de2)*kernelmatrix[p[i],q[j]]+(de1)*(1-de2)*kernelmatrix[q[i],p[j]]+(de1)*(de2)*kernelmatrix[q[i],q[j]]
        new_kernel_matrix=np.vstack((np.hstack((kernelmatrix[0:X_len,0:X_len],K2)),np.hstack((K2.T,K3)))) 
        testX_km=np.hstack((kernelmatrix[X_len:,0:X_len], all_K2[X_len:,:]))
        return new_kernel_matrix,testX_km
    
    def score(self, X, y=None):
        # counts number of values bigger than mean
        y_pre=self.predict(X)
        metric=ImBinaryMetric(y,y_pre)
        return metric.AP()
