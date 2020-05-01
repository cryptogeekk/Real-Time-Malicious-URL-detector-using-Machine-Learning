#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:29:46 2020

@author: krishna
"""

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

#Feature reduction using PCA
# from sklearn.decomposition import PCA
# pca=PCA(n_components=79)
# train_x_reduced=pca.fit_transform(train_x)
# feature_importance=pca.explained_variance_ratio_
# feature_importance_df=pd.DataFrame(pca.components_,columns=train_x.columns)


# temp=train_x.isnull().sum()
# np.where(np.isnan(train_x))
# np.nan_to_num(train_x)

# np.where(train_x.values>=np.finfo(np.float64).max)
# np.isnan(train_x.values.any())




#finding feature importance using Select K best
from scipy.stats import chisquare
from sklearn.feature_selection import SelectKBest,chi2,f_classif
k_best=SelectKBest(score_func=f_classif,k=10)
k_best.fit(train_x,train_y_svm)
# train_new=k_best.fit_transform(train_x,train_y)

score1=k_best.scores_
    #plotting the graph of scores
score1_df=pd.DataFrame(score1)  
score1_df.index=train_x.columns         #gives the feature scores of each attribute
score1_df.plot.bar(figsize=(20,10))
sorted_score=score1_df.sort_values(ascending=False,by=0)
    #top 20 feature
kboost_score_top_20=list(sorted_score[:20].index)


#finding feature importance using Random forest classifier
from sklearn.ensemble import RandomForestClassifier
rnd_clf=RandomForestClassifier(n_estimators=500,n_jobs=-1)
start_time=time.time()
rnd_clf.fit(train_x,train_y)
end_time=time.time()
time_required=start_time-end_time
print('Time required for training the model is ' + str(time_required))
feature_names=list(train_x.columns)
for name,score in zip(train_x.columns,rnd_clf.feature_importances_):
    print(name,score)
rnd_score=rnd_clf.feature_importances_
rnd_score=pd.DataFrame(rnd_score)
rnd_score.index=train_x.columns
rnd_score.plot.bar(figsize=(20,10))
sorted_rnd_score=rnd_score.sort_values(ascending=False,by=0)

#comparing top 40 features of both classifier
rnd_score_top_20=list(sorted_rnd_score[:50].index)
kboost_score_top_20=list(sorted_score[:50].index)
count=0
for feature in rnd_score_top_20:
    if feature in kboost_score_top_20:
        count=count+1
        #for 50 we get count 38,for 40 we got 27

#feature reduction with backward feature elimination technique
from sklearn.feature_selection import RFE
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC
estimator=OneVsRestClassifier(SVC(kernel='rbf',gamma=5,C=1000))
# estimator=LinearSVC()
selector=RFE(estimator,5)
selector=selector.fit(train_x,train_y_svm)
RFE_score=selector.ranking_
RFE_score_df=pd.DataFrame(RFE_score)
RFE_score_df.index=train_x.columns
RFE_score_sorted=RFE_score_df.sort_values(ascending=True,by=0)

#comparing the features between all three feature reduction algorithims
RFE_score_50=list(RFE_score_sorted[:50].index)
rnd_score_top_50=list(sorted_rnd_score[:50].index)
kboost_score_top_50=list(sorted_score[:50].index)

count=0
final_features=[]       #These are the features after 3 reduction algorithims are applied
for columns in train_x.columns:
        if columns in rnd_score_top_50:
            if columns in kboost_score_top_50:
                if columns in RFE_score_50:
                    final_features.append(columns)
                    
                    
