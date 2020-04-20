#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 01:29:46 2020

@author: krishna
"""

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
for column in column_names:
    print(data1[column].isnull().values.any())  #Resulted all false value,Now we can move ahead.
    
#saving the preprocessed data
data1.to_csv('preprocessed_data.csv')
    
