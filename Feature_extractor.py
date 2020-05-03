#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 00:40:56 2020

@author: krishna
"""

import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize,word_tokenize
dataset_phishing_url_original=pd.read_csv('phishing_dataset.csv') 
dataset_phishing_url_modified=dataset_phishing_url_original['URL']
phising_columns=['domain_token_count','tld','urlLen','domainlength','fileNameLen','domainUrlRatio','NumberofDotsinURL','Query_DigitCount','LongestPathTokenLength','delimeter_Domain','delimeter_path','SymbolCount_Domain','Entropy_Domain','URL_Type_obf_Type']

# domain_token_count(data1.iloc[1])
# sent_tokenize(data1.iloc[1])    

dataset_phising_all=pd.read_csv('Phishing.csv')
column_names=dataset_phising_all.columns
# data3=data2[phising_columns]

#Creating an empty dataset with 13 faetures
dataset_13=pd.DataFrame(0,index=np.arange(len(dataset_phishing_url_modified)),columns=phising_columns)

#function for  domain count
index=0
def domain_token_count(url):
    global index
    token_count=0
    for char in url:
        if char=='/':
            token_count=token_count+1
    token_count=token_count-2
    dataset_13['domain_token_count'].iloc[index]=token_count
    index=index+1

# creating a dataset for domain_token_count
for url in dataset_phishing_url_modified:
    domain_token_count(url)

#function for tld
