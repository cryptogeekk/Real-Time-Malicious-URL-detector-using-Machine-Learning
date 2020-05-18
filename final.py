#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 02:35:05 2020

@author: krishna
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:20:59 2020

@author: krishna
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:09:00 2020

@author: krishna
"""

#----------Here I had taken only 5 features obtained from my dataset and applied Decision tree and Random FOrest--------------------

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('dataset_final1')
data.drop('Unnamed: 0',axis=1,inplace=True)  #only done for this dataset since it contains one extra unnamed column
data.drop('domainUrlRatio',axis=1,inplace=True)   #only done for experiment purpose, in main code remove it.
column_names=list(data.columns)
data['URL_Type_obf_Type'].value_counts()

# rnd_score_top_5.append('URL_Type_obf_Type')
# kboost_score_top_6.append('URL_Type_obf_Type')

#experimenting with the reduced faetures
# data=data[rnd_score_top_5]  
# data=data[kboost_score_top_6]   


#creating a category of malicious and non-malicious
# data['category']='malicious'
# data['category'][7930:15711]='non-malicious'
# data['category'].value_counts()

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
shuffled_x=shuffled_dataset.drop(['URL_Type_obf_Type'],axis=1)
shuffled_y=shuffled_dataset[['URL_Type_obf_Type']]
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
shuffled_dataset_scaled=sc_x.fit_transform(shuffled_x)
shuffled_dataset_scaled=pd.DataFrame(shuffled_dataset_scaled)
shuffled_dataset_scaled.columns=shuffled_x.columns
dataset_final=pd.concat([shuffled_dataset_scaled,shuffled_y],axis=1)
# dataset_final=pd.concat([shuffled_x,shuffled_y],axis=1)                 #for non-feature scaling algorithims
#dataset_final.drop(['ISIpAddressInDomainName'],inplace=True,axis=1)   #dropping this column since it always contain zero



#splitting the dataset into train set and test set
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(dataset_final,test_size=0.2,random_state=42)  
    #sorting the train_set and test set
pd.DataFrame.sort_index(train_set,axis=0,ascending=True,inplace=True) 
pd.DataFrame.sort_index(test_set,axis=0,ascending=True,inplace=True) 
    #splitting further ito train_x,train_y,test_x,test_x                        ----Multiclass classification-----
train_y=train_set['URL_Type_obf_Type']                                          #train data for binary classification
# train_y_binary=train_set['category']        
train_x=train_set.drop(['URL_Type_obf_Type'],axis=1,inplace=True)
train_x=train_set
test_y=test_set['URL_Type_obf_Type']
# test_y_binary=test_set['category']                                              #test data for binary classsification
test_x=test_set.drop(['URL_Type_obf_Type'],axis=1,inplace=True)
test_x=test_set

#Encoding the categorical variables
    #for SVM classification
train_y_svm=train_y  
test_y_svm=test_y
    #for other types of classification
train_y=pd.get_dummies(train_y)
# train_y_binary=pd.get_dummies(train_y_binary)
# train_y_binary=train_y_svm['benign']
test_y=pd.get_dummies(test_y)
# test_y_binary_num=pd.get_dummies(test_y_binary)

#Applying Logistic regression for multiclass classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
ovr_clf=OneVsRestClassifier(LogisticRegression())
ovr_clf.fit(train_x,train_y)
ovr_predicted=ovr_clf.predict(test_x)
 

from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score
ovr_confusion_matrix=confusion_matrix(test_y.values.argmax(axis=1),ovr_predicted.argmax(axis=1))
precision_ovr=precision_score(test_y,ovr_predicted,average='micro')   #gives 95.76 percent of accuracy
precision_ovr_all=precision_score(test_y,ovr_predicted,average=None)    #gives 94.98,96.52
f1_lgd=f1_score(test_y,ovr_predicted,average='micro')                   #gives 95.76 percent of accuracy
recall_lgd=recall_score(test_y,ovr_predicted,average='micro')
#plotting precision recall curve
ovr_predicted=pd.DataFrame(ovr_predicted)   
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(test_y['benign'],ovr_predicted[0])

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b-", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")

    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

#plotting precision recall curve
ovr_predicted=pd.DataFrame(ovr_predicted)   
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_y['benign'],ovr_predicted[0])

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, linewidth=2, label='True positive rate')
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    
plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(test_y['benign'],ovr_predicted[0])

    
#--------------------------------------Applying SVM for multiclass classification------------------------
from sklearn.svm import LinearSVC
svc=OneVsRestClassifier(LinearSVC(C=10,loss='hinge'))
svc.fit(train_x,train_y)
svc_predicted=svc.predict(test_x)

from sklearn.metrics import confusion_matrix,precision_score,f1_score
svc_confusion_matrix=confusion_matrix(test_y.values.argmax(axis=1),svc_predicted.argmax(axis=1))
precision_svc=precision_score(test_y,svc_predicted,average='micro')
precision_svc_all=precision_score(test_y,svc_predicted,average=None)    
f1_sgd=f1_score(test_y_label,sg_predicted,average='micro')   

#-------------------------------------------Appying SVM  for multiclass classification but with different kernels
from sklearn.svm import SVC
svm_poly=OneVsRestClassifier(SVC(kernel='poly',degree=4,C=10000))
# svm_poly=OneVsRestClassifier(SVC(kernel='poly'))
svm_poly.fit(train_x,train_y)
svm_poly_predicted=svm_poly.predict(test_x)

from sklearn.metrics import confusion_matrix,precision_score,f1_score
svm_poly_binary_confusion_matrix=confusion_matrix(test_y.values.argmax(axis=1),svm_poly_predicted.argmax(axis=1))
precision_svm_poly=precision_score(test_y,svm_poly_predicted,average='micro')   #gives 98.94 percent of accuracy 
precision_svm_poly_all=precision_score(test_y,svm_poly_predicted,average=None)    #gives 98.62 for benign and 99.26 percent
f1_svm_poly=f1_score(test_y,svm_poly_predicted,average='micro')                 #gives 98.948 percent of accuracy

#plotting precision recall curve
svm_poly_predicted=pd.DataFrame(svm_poly_predicted)   
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(test_y['benign'],svm_poly_predicted[0])

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b-", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabelz('Recall score')
    plt.ylabel('Precision score')

    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

#plotting precision recall curve
svm_poly_predicted=pd.DataFrame(svm_poly_predicted)   
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_y['benign'],svm_poly_predicted[0])

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, linewidth=2, label='True positive rate')
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    
plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(test_y['benign'],svm_poly_predicted[0])


#-------------------------Performing grid search on SVM polynomial parameters
from sklearn.model_selection import GridSearchCV
degree=np.arange(1,11)
C=np.arange(1,1001)

from sklearn.svm import SVC
svm_poly=OneVsRestClassifier(SVC(kernel='poly'))

    
param_grid={'estimator__degree':[1,2,3,4,5,6],'estimator__C':[1,100,1000]}
grid_search=GridSearchCV(svm_poly,param_grid,cv=2,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(train_x,train_y)

#Using SVM for binary classification
from sklearn.svm import SVC
svm_poly=OneVsRestClassifier(SVC(kernel='poly',degree=4,C=10000))
svm_poly.fit(train_x,train_y)
svm_poly_predicted_binary=svm_poly.predict(test_x)

from sklearn.metrics import confusion_matrix,precision_score,f1_score
svm_poly_binary_confusion_matrix=confusion_matrix(test_y.values.argmax(axis=1),svm_poly_predicted_binary.argmax(axis=1))
precision_svm_poly_binary=precision_score(test_y,svm_poly_predicted_binary,average='micro')   
precision_svm_poly_binary_all=precision_score(test_y,svm_poly_predicted_binary,average=None)
f1_svm_poly_binary=f1_score(test_y,svm_poly_predicted_binary,average='micro')                 

#Performing grid search on SVM polynomial parameters
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
svm_poly=OneVsRestClassifier(SVC(kernel='poly'))
   
param_grid={'estimator__degree':[4],'estimator__C':[10000]}
grid_search=GridSearchCV(svm_poly,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(train_x,train_y)
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
precision_svm_rbf=precision_score(test_y,svm_rbf_predicted,average='micro')   
precision_svm_rbf_all=precision_score(test_y,svm_rbf_predicted,average=None)    
f1_svm_poly=f1_score(test_y,svm_poly_predicted,average='micro')                 


#Using K nearest Neighbours for classification
from sklearn.neighbors import KNeighborsClassifier
knc_clf=KNeighborsClassifier(leaf_size=10,n_neighbors=5,p=2)
# knc_clf=KNeighborsClassifier()
knc_clf.fit(train_x,train_y)
knc_clf_predicted=knc_clf.predict(test_x)

from sklearn.metrics import confusion_matrix,precision_score,f1_score
knc_clf_confusion_matrix=confusion_matrix(test_y.values.argmax(axis=1),knc_clf_predicted.argmax(axis=1))
precision_knn_clf=precision_score(test_y,knc_clf_predicted,average='micro')     #gives 99.04 percent of accuracy
precision_knn_clf_all=precision_score(test_y,knc_clf_predicted,average=None)   #gives 98.37 and 99.70 percent of accuracy

#plotting precision recall curve
knc_clf_predicted=pd.DataFrame(knc_clf_predicted)   
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(test_y['benign'],knc_clf_predicted[0])

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b-", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabelz('Recall score')
    plt.ylabel('Precision score')

    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

#Roc curve
knc_clf_predicted=pd.DataFrame(knc_clf_predicted)   
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_y['benign'],knc_clf_predicted[0])

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, linewidth=2, label='True positive rate')
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    
plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(test_y['benign'],knc_clf_predicted[0])


#Hyperparameter tuning for KNeighbour classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
knc_clf=KNeighborsClassifier(n_jobs=-1)

   
param_grid={'n_neighbors':[5,10,15,20],'leaf_size':[10,20,30,40,50,70,90],'p':[1,2]}
grid_search=GridSearchCV(knc_clf,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(train_x,train_y)
grid_search.best_params_
cvres=grid_search.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
    
knc_clf.get_params().keys()


#Applying random forest on it
from sklearn.ensemble import RandomForestClassifier
# rnd_clf=RandomForestClassifier(n_estimators=500,max_leaf_nodes=60,n_jobs=-1)
rnd_clf=RandomForestClassifier()
rnd_clf.fit(train_x,train_y)
rnd_clf_predicted=rnd_clf.predict(test_x)

from sklearn.metrics import confusion_matrix,precision_score,f1_score
rnd_clf_confusion_matrix=confusion_matrix(test_y.values.argmax(axis=1),rnd_clf_predicted.argmax(axis=1))
precision_rnd_clf=precision_score(test_y,rnd_clf_predicted,average='micro')     #gives 99.34 percent of accuracy with 500 estimators and 30 leaf nodes and 99.57 percent of accuracy with hyperparamater tuning
precision_rnd_clf_all=precision_score(test_y,rnd_clf_predicted,average=None)   #gives 99.25 and 99.44 percent of accuracy

#plotting precision recall curve
rnd_clf_predicted=pd.DataFrame(rnd_clf_predicted)   
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(test_y['benign'],rnd_clf_predicted[0])

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b-", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabelz('Recall score')
    plt.ylabel('Precision score')
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

#Roc curve
rnd_clf_predicted=pd.DataFrame(rnd_clf_predicted)   
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_y['benign'],rnd_clf_predicted[0])

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, linewidth=2, label='True positive rate')
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    
plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(test_y['benign'],rnd_clf_predicted[0])

#--------------------------------------------

#for visualizing result
# Visualising the clusters




#Hyperparameter tuning in Random forest classifier
from sklearn.model_selection import GridSearchCV

param_grid={'n_estimators':[500],'max_leaf_nodes':[30,40,50,60],'n_jobs':[-1]}
grid_search=GridSearchCV(rnd_clf,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(train_x,train_y)
grid_search.best_params_
cvres=grid_search.cv_results_
for mean_score,params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
    
    
    
    
knc_clf.get_params().keys()


#-------------------------------------Here I had done feature for research paper work------------------------------------


#finding feature importance using Select K best
from scipy.stats import chisquare
from sklearn.feature_selection import SelectKBest,chi2,f_classif
k_best=SelectKBest(score_func=f_classif,k=9)
k_best.fit(train_x,train_y_svm)
# train_new=k_best.fit_transform(train_x,train_y)

score1=k_best.scores_
    #plotting the graph of scores but first doing standard scaling
from sklearn.preprocessing import MinMaxScaler
sc_x=MinMaxScaler()
score1_df=pd.DataFrame(score1)  
score1_df=sc_x.fit_transform(score1_df)
score1_df=pd.DataFrame(score1_df)  


score1_df.index=train_x.columns         #gives the feature scores of each attribute
score1_df.plot.bar(figsize=(5,5))
sorted_score=score1_df.sort_values(ascending=False,by=0)
    #top 20 feature
kboost_score_top_6=list(sorted_score[:6].index)    #we took six becausue from graph we can see, six features were contributing the most


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
#standardizing the score usning min max scaler
rnd_score=sc_x.fit_transform(rnd_score)
rnd_score=pd.DataFrame(rnd_score)

rnd_score.index=train_x.columns
rnd_score.plot.bar(figsize=(5,5))
sorted_rnd_score=rnd_score.sort_values(ascending=False,by=0)

#comparing top 40 features of both classifier
rnd_score_top_5=list(sorted_rnd_score[:5].index)  # we took only 5 features since 5 features were contributing the most


#feature reduction with backward feature elimination technique
from sklearn.feature_selection import RFE
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import LabelEncoder
train_y_lg=train_y['phishing']
from sklearn.svm import SVR
estimator=SVR(kernel='linear')
# from sklearn.linear_model import LogisticRegression
# estimator=OneVsRestClassifier(LogisticRegression())
# estimator=KNeighborsClassifier(leaf_size=10,n_neighbors=5,p=2)
# estimator=SVR(kernel='')
selector=RFE(estimator,9,step=1)
selector=selector.fit(train_x,train_y_lg)      #we should give here train_y in text form

RFE_score=selector.ranking_
RFE_score_df=pd.DataFrame(RFE_score)
RFE_score_df.index=train_x.columns
RFE_score_sorted=RFE_score_df.sort_values(ascending=True,by=0)
rfe_score_top_30=RFE_score_sorted[:30]

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
                    
# from sklearn.datasets import make_friedman1
# x,y=make_friedman1(n_samples=50,n_features=10,random_state=0)


#Finding feature importance using Correlation matrix
train_y_corr=train_y['benign']
data_corr=train_x.join(train_y_corr)
corr_matrix=data_corr.corr()
corr_matrix['benign'].sort_values(ascending=False)


#calculatig the pearson and spearmar correlation coefficient
from scipy.stats import pearsonr
from scipy.stats import spearmanr
for columns in train_x.columns:
    print(pearsonr(train_x[columns].values, train_y.values))
    print(spearmanr(train_x[columns].values, train_y_corr.values))

    





#for support vector machine
nothing=np.arange(0,101)
initial=[3.22,96.87,98.50,97.68]
final=[1.671,98.38,99.65,99.01]
initial=pd.DataFrame(initial)
final=pd.DataFrame(final)

index=['FPR','Precision','Recall','F1-score']

initial.index=index
final.index=index

initial.plot.bar()
final.plot.bar()
final.se



fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Men')
rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

#----------------------------------------------------------
labels=['FPR','Precision','Recall','F1 score']
Before_hyperparamter_tuning=[3.22,96.87,98.50,97.68]
after_hyperparameter_tuning=[1.671,98.38,99.65,99.01]

x=np.arange(len(labels))
width=0.35

fig,ax=plt.subplots()
rects1=ax.bar(x-width/2, Before_hyperparamter_tuning,width,label='Before ')
rects2=ax.bar(x+width/2, after_hyperparameter_tuning,width,label='After')

ax.set_ylabel('Percentage')
ax.set_title('Results')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

