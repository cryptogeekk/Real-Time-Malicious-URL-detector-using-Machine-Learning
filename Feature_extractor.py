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
url=dataset_phishing_url_modified[0]   #Remove this code while compiling all code
url='http://clubeamigosdopedrosegundo.com.br/last/'
url2='http://blogger.com.buses-forsale.co.za/sq/index.php?bshowgif=0&amp'
url3='http://updatepaypal.c0m.uk.d4ps1s5u3xo5t2c6v2kn3fz7c9y5u2v5u2rf3o5x1x0.zfc6v3dx2s5j9uk1xble4efc1m3dxk21k3v5ch95den4m39d1sv2h.balihoo.gr/account/webxscr.html?cmd=2615d80a13c0db1f22d2300ef60a67593b79a4d03747447e6b625t28d36121s1cd82730257d4ffad785277a59c2209'



dataset_phising_all=pd.read_csv('Phishing.csv')
data_initial_13=dataset_phising_all[phising_columns]
column_names=data_initial_13.columns


#Creating an empty dataset with 13 faetures
dataset_13=pd.DataFrame(0,index=np.arange(len(dataset_phishing_url_modified)),columns=phising_columns)

#---------------------------------------function for domain token count----------------------------------------------
index1=0
def domain_token_count(url):
    global index1
    token_count=0
    for char in url:
        if char=='/':
            token_count=token_count+1
    token_count=token_count-2
    dataset_13['domain_token_count'].iloc[index1]=token_count
    index1=index1+1

# creating a dataset for domain_token_count
for url in dataset_phishing_url_modified:
    domain_token_count(url)

#---------------------------------------function for tld count------------------------------------------------
index2=0
top_level_domains=pd.read_csv('top-level-domain-names.csv')
domain_list=top_level_domains['Domain'].tolist()
domain_list1=[]
    #function to remove . from domain names
for domain in domain_list:
    splitted_domain=domain.split('.')
    domain_list1.append(splitted_domain[1])
    

def tld__count(url):
    global index2
    tld_count=0
    splitted_text1=url.split('/')
    splitted_text2=splitted_text1[2].split('.')
    
    for domain in splitted_text2:
        if domain in domain_list1:
            tld_count=tld_count+1  
    # print(tld_count)
    dataset_13['tld'].iloc[index2]=tld_count
    index2=index2+1
            
#creating a dataset for tld_count
for url in dataset_phishing_url_modified:
    tld__count(url)
    
#-----------------------------------function for URL len---------------------------------------------------------
index3=0
def url_length(url):
    global index3
    count_length=0
    for character in url:
        count_length=count_length+1
        
    dataset_13['urlLen'].iloc[index3]=count_length
    index3=index3+1

#creating a dataset for url_length
for url in dataset_phishing_url_modified:
    url_length(url)
    
#-------------------------------function to create domain length--------------------------------------------------
index4=0
def domain_length_count(url):
    global index4
    domain_length=0
    splitted_text3=url.split('/')
    splitted_text4=splitted_text3[2].split('.')
    
    for domain in splitted_text4:
        if domain in domain_list1:
            domain_length=domain_length+len(domain)
    # print(domain_length)
    dataset_13['domainlength'].iloc[index4]=domain_length
    index4=index4+1 

#creating a dataset
for url in dataset_phishing_url_modified:
    domain_length_count(url)
    
#------------------------------function to create domain to URL ratio--------------------------------------------
index5=0
def domain_url_ratio(url):
    global index5
    ratio=0

    for domain,url in zip(dataset_13['domainlength'],data_initial_13['urlLen']):
        ratio=domain/url
        dataset_13['domainUrlRatio'].iloc[index5]=ratio
        index5=index5+1
    
#creating a dataset
for url in dataset_phishing_url_modified:
    domain_url_ratio(url)

#-------------------------------Numberof dots in URL-------------------------------------------------
index6=0
def number_of_dots(url):
    global index6
    
    count=0
    for character in url:
        if character=='.':
            count=count+1
    dataset_13['NumberofDotsinURL'].iloc[index6]=count
    index6=index6+1
    
#creating a dataset
for url in dataset_phishing_url_modified:
    number_of_dots(url)
    
         


