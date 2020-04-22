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

#creating a category of amlicious and non-malicious
data['category']='malicious'
data['category'][7930:15711]='non-malicious'
data['category'].value_counts()

#Data cleaning
column_names=list(data.columns)
column_names.pop(80)
column_names.pop(79)

#checking if any dataframe columns contain null values or not
na_values_columns=[]
for column_names in data:
    result=data[column_names].isnull().values.any()
    if result==True:
        na_values_columns.append(column_names)
        
#filling the na column value data with a median value    
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')
# for na_values_columns in data:
#     imputer.fit_transform(data[[na_values_columns]])
transformed_data=imputer.fit_transform(data[na_values_columns])
    
#checking again if the imputer corrected to the na values or not
transformed_data_df=pd.DataFrame(transformed_data)
for column in range(0,8):
    print(transformed_data_df[column].isnull().values.any())
    #resulted all false value,now we know that simple imputer worked
    
#putting back the transofrmed data to the original data
transformed_data_df.columns=na_values_columns
data1=data.copy()
for column in transformed_data_df.columns:
    data1[column]=transformed_data_df[column]
   
#checking again if data1 contains any na values or not
# for column in column_names:
#     print(data1[column].isnull().values.any())  #Resulted all false value,Now we can move ahead.
    
#saving the preprocessed data
data1[data1==np.inf]==np.nan                #Handling the infinite value with mean value
data1.fillna(data1.mean(),inplace=True)
# data1.to_csv('preprocessed_data.csv')

#shuffling the dataset
shuffled_dataset=data1.sample(frac=1).reset_index(drop=True)

#splitting the dataset into train set and test set
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(shuffled_dataset,test_size=0.2,random_state=42)    
# train_set['URL_Type_obf_Type'].value_counts()
# test_set['URL_Type_obf_Type'].value_counts()
    #splitting further
train_y=train_set['URL_Type_obf_Type']
train_x=train_set.drop(['URL_Type_obf_Type','category'],axis=1,inplace=True)
train_x=train_set
test_y=test_set['URL_Type_obf_Type']
test_x=test_set.drop(['URL_Type_obf_Type','category'],axis=1,inplace=True)
test_x=test_set

#handling infinite value
train_x[train_x==np.inf]=np.nan
train_x.fillna(train_x.mean(),inplace=True)

#sorting on the basis of index
pd.DataFrame.sort_index(train_x,axis=0,ascending=True,inplace=True) 
pd.DataFrame.sort_index(train_y,ascending=True,inplace=True) 
train_x.drop(['URL_Type_obf_Type'],axis=1,inplace=True)
pd.DataFrame.sort_index(test_x,axis=0,ascending=True,inplace=True) 
pd.DataFrame.sort_index(test_y,ascending=True,inplace=True) 
test_x.drop(['URL_Type_obf_Type'],axis=1,inplace=True)

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

#Encoding the train_y
train_y_svm=train_y         #creating tarin_y without encoding for SVM since svm donot support multiclass classification
train_y=pd.get_dummies(train_y)


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
from sklearn.svm import LinearSVC
estimator=LinearSVC()
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
                    
                    
