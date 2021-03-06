#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:35:11 2020

@author: krishna
"""



#----------Here I had applied the algorithis which needs scaling with 81 and 20 features-------------------

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('All.csv')
column_names=list(data.columns)
data['URL_Type_obf_Type'].value_counts()

#creating a category of malicious and non-malicious
data['category']='malicious'
data['category'][7930:15711]='non-malicious'
data['category'].value_counts()

#Data cleaning
column_names=list(data.columns)
column_names.pop(80)
column_names.pop(79)

#shuffling the dataframe
shuffled_dataset=data.sample(frac=1).reset_index(drop=True)

#dropping the categorical value
# categorical_data=shuffled_dataset[['URL_Type_obf_Type','category']]
# data1=shuffled_dataset.drop(['URL_Type_obf_Type','category'],axis=1)

#checking for na and inf values
shuffled_dataset.replace([np.inf,-np.inf],np.nan,inplace=True)                  #handling the infinite value
shuffled_dataset.fillna(shuffled_dataset.mean(),inplace=True)                   #handling the na value

#checking if any value in data1 now contains infinite and null value or not
null_result=shuffled_dataset.isnull().any(axis=0)
inf_result=shuffled_dataset is np.inf

#scaling the dataset with standard scaler
shuffled_x=shuffled_dataset.drop(['URL_Type_obf_Type','category'],axis=1)
shuffled_y=shuffled_dataset[['URL_Type_obf_Type','category']]
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
shuffled_dataset_scaled=sc_x.fit_transform(shuffled_x)
shuffled_dataset_scaled=pd.DataFrame(shuffled_dataset_scaled)
shuffled_dataset_scaled.columns=shuffled_x.columns
dataset_final=pd.concat([shuffled_dataset_scaled,shuffled_y],axis=1)
dataset_final.drop(['ISIpAddressInDomainName'],inplace=True,axis=1)   #dropping this column since it always contain zero

#Preparing the dataset with the reduced features of K-Best
# reduced_features=['SymbolCount_Domain','domain_token_count','tld','Entropy_Afterpath','NumberRate_AfterPath','ArgUrlRatio','domainUrlRatio','URLQueries_variable','SymbolCount_FileName','delimeter_Count','argPathRatio','delimeter_path','pathurlRatio','SymbolCount_Extension','SymbolCount_URL','NumberofDotsinURL','Arguments_LongestWordLength','SymbolCount_Afterpath','CharacterContinuityRate','domainlength']
# reduced_features.append('URL_Type_obf_Type')
# reduced_features.append('category')
# shuffled_dataset1=shuffled_dataset[reduced_features]

#Applying the top 30 features
rfe_score_top_30_copy=rfe_score_top_30.copy()
rfe_score_top_30_copy=rfe_score_top_30_copy.index
list=rfe_score_top_30_copy.to_list()
list.append('URL_Type_obf_Type')
list.append('category')
columns=dataset_final.columns
dataset_final=dataset_final[list]


#splitting the dataset into train set and test set
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(dataset_final,test_size=0.2,random_state=42)  
    #sorting the train_set and test set
pd.DataFrame.sort_index(train_set,axis=0,ascending=True,inplace=True) 
pd.DataFrame.sort_index(test_set,axis=0,ascending=True,inplace=True) 
    #splitting further ito train_x,train_y,test_x,test_x                        ----Multiclass classification-----
train_y=train_set['URL_Type_obf_Type']                                          #train data for binary classification
train_y_binary=train_set['category']        
train_x=train_set.drop(['URL_Type_obf_Type','category'],axis=1,inplace=True)
train_x=train_set
test_y=test_set['URL_Type_obf_Type']
test_y_binary=test_set['category']                                              #test data for binary classsification
test_x=test_set.drop(['URL_Type_obf_Type','category'],axis=1,inplace=True)
test_x=test_set

#Encoding the categorical variables
    #for SVM classification
train_y_svm=train_y  
test_y_svm=test_y
    #for other types of classification
train_y=pd.get_dummies(train_y)
train_y_binary=pd.get_dummies(train_y_binary)
test_y=pd.get_dummies(test_y)
test_y_binary_num=pd.get_dummies(test_y_binary)

#Applying the Logistic Regresssion algorithim on the dataset  malicious 1 and non_malicious zero
from sklearn.linear_model import LogisticRegression
lg_clf=LogisticRegression()
train_y_lg=train_y_binary['malicious']
test_y_lg=test_y_binary_num['malicious']
lg_clf.fit(train_x,train_y_lg)
lg_predicted=lg_clf.predict(test_x)

from sklearn.metrics import confusion_matrix,precision_score,f1_score
lg_confusion_matrix=confusion_matrix(test_y_lg.values,lg_predicted)
precision_lg=precision_score(test_y_lg,lg_predicted,average='micro')  #gives 93.16 percent of accuracy 
precision_lg_all=precision_score(test_y_lg,lg_predicted,average=None)      
f1_lg=f1_score(test_y_lg,lg_predicted,average='micro')                 #gives 93.16 percent of accuracy


#Using softmax regression for multiclass classification
    #preparing train_y_svm for softmax regresiion using labelencoder
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
train_y_label=encoder.fit_transform(train_y_svm)
test_y_label=encoder.fit_transform(test_y_svm)
    #Now applying Softmax regression
softmax_reg=LogisticRegression(multi_class='multinomial',solver='lbfgs',C=10)
softmax_reg.fit(train_x,train_y_label)
sg_predicted=softmax_reg.predict(test_x)

from sklearn.metrics import confusion_matrix,precision_score,f1_score
sg_confusion_matrix=confusion_matrix(test_y_label,sg_predicted)
precision_sg=precision_score(test_y_label,sg_predicted,average='micro')   #gives 85 percent of accuracy
precision_sg_all=precision_score(test_y_label,sg_predicted,average=None)    #gives 86,81,83,82,90
f1_sgd=f1_score(test_y_label,sg_predicted,average='micro')                  #gives 85 percent of score

#Applying Logistic regression for multiclass classification
from sklearn.multiclass import OneVsRestClassifier
ovr_clf=OneVsRestClassifier(LogisticRegression())
ovr_clf.fit(train_x,train_y)
ovr_predicted=ovr_clf.predict(test_x)    

from sklearn.metrics import confusion_matrix,precision_score,f1_score
ovr_confusion_matrix=confusion_matrix(test_y.values.argmax(axis=1),ovr_predicted.argmax(axis=1))
precision_ovr=precision_score(test_y,ovr_predicted,average='micro')   #gives 84 percent of accuracy with onehot and 82 with labelencoder
precision_ovr_all=precision_score(test_y,ovr_predicted,average=None)    #gives 86,80,85,78,91
f1_sgd=f1_score(test_y_label,sg_predicted,average='micro')              

#Applying SVM for multiclass classification
from sklearn.svm import LinearSVC
svc=OneVsRestClassifier(LinearSVC(C=10,loss='hinge'))
svc.fit(train_x,train_y)
svc_predicted=svc.predict(test_x)

from sklearn.metrics import confusion_matrix,precision_score,f1_score
svc_confusion_matrix=confusion_matrix(test_y.values.argmax(axis=1),svc_predicted.argmax(axis=1))
precision_svc=precision_score(test_y,svc_predicted,average='micro')   #gives 84.53 percent of accuracy without C and loss='hinge' with bothr paraemeters precision increased to 86
precision_svc_all=precision_score(test_y,svc_predicted,average=None)    #gives 86,81,85,78,91
f1_sgd=f1_score(test_y_label,sg_predicted,average='micro')   

#Appying SVM  for multiclass classification but with different kernels
from sklearn.svm import SVC
svm_poly=OneVsRestClassifier(SVC(kernel='poly',degree=3,C=10))
svm_poly.fit(train_x,train_y)
svm_poly_predicted=svm_poly.predict(test_x)

from sklearn.metrics import confusion_matrix,precision_score,f1_score
svm_poly_confusion_matrix=confusion_matrix(test_y.values.argmax(axis=1),svm_poly_predicted.argmax(axis=1))
precision_svm_poly=precision_score(test_y,svm_poly_predicted,average='micro')   #gives 94.32 percent of accuracy 
precision_svm_poly_all=precision_score(test_y,svm_poly_predicted,average=None)    #gives 95,93,93,90,98
f1_svm_poly=f1_score(test_y,svm_poly_predicted,average='micro')                 #gives 91 percent of accuracy

#Performing grid search on SVM polynomial parameters
from sklearn.model_selection import GridSearchCV
degree=np.arange(1,11)
C=np.arange(1,1001)

from sklearn.svm import SVC
svm_poly=OneVsRestClassifier(SVC(kernel='poly'))

    
param_grid={'estimator__degree':[1,2,3,4,5],'estimator__C':[1,100]}
grid_search=GridSearchCV(svm_poly,param_grid,cv=2,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(train_x,train_y)

#Using SVM for binary classification
from sklearn.svm import SVC
svm_poly=OneVsRestClassifier(SVC(kernel='poly',degree=5,C=1000))
svm_poly.fit(train_x,train_y_binary)
svm_poly_predicted_binary=svm_poly.predict(test_x)

from sklearn.metrics import confusion_matrix,precision_score,f1_score
svm_poly_binary_confusion_matrix=confusion_matrix(test_y_binary_num.values.argmax(axis=1),svm_poly_predicted_binary.argmax(axis=1))
precision_svm_poly_binary=precision_score(test_y_binary_num,svm_poly_predicted_binary,average='micro')   #gives 98 percent of accuracy and 98.66 with degree=5,c=1000
precision_svm_poly_binary_all=precision_score(test_y_binary_num,svm_poly_predicted_binary,average=None)    #gives 98,95
f1_svm_poly_binary=f1_score(test_y_binary_num,svm_poly_predicted_binary,average='micro')                 #gives 98 percent of accuracy

#Performing grid search on SVM polynomial parameters
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
svm_poly=OneVsRestClassifier(SVC(kernel='poly'))
   
param_grid={'estimator__degree':[4,5,6],'estimator__C':[100,1000]}
grid_search=GridSearchCV(svm_poly,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(train_x,train_y_binary)
grid_search.best_params_
cvres=grid_search.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
    
#Performing feature reduction with PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
model=pca.fit(train_x)
feature_importance=pca.explained_variance_ratio_
feature_importance_df=pd.DataFrame(pca.components_,columns=train_x.columns)
temp=pca.components_
    

#Using gaussian SVM kernel for classification
from sklearn.svm import SVC
svm_rbf=OneVsRestClassifier(SVC(kernel='rbf',gamma=5,C=1000))
svm_rbf.fit(train_x,train_y)
svm_rbf_predicted=svm_poly.predict(test_x)

from sklearn.metrics import confusion_matrix,precision_score,f1_score
svm_rbf_confusion_matrix=confusion_matrix(test_y.values.argmax(axis=1),svm_rbf_predicted.argmax(axis=1))
precision_svm_rbf=precision_score(test_y,svm_rbf_predicted,average='micro')   #gives 94.32 percent of accuracy 
precision_svm_rbf_all=precision_score(test_y,svm_rbf_predicted,average=None)    #gives 95,93,93,90,98
f1_svm_poly=f1_score(test_y,svm_poly_predicted,average='micro')                 #gives 91 percent of accuracy







