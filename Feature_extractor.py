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
url1='http://clubeamigosdopedrosegundo.com.br/last/'
url2='http://blogger.com.buses-forsale.co.za/sq/index.php?bshowgif=0&amp'
url3='http://updatepaypal.c0m.uk.d4ps1s5u3xo5t2c6v2kn3fz7c9y5u2v5u2rf3o5x1x0.zfc6v3dx2s5j9uk1xble4efc1m3dxk21k3v5ch95den4m39d1sv2h.balihoo.gr/account/webxscr.html?cmd=2615d80a13c0db1f22d2300ef60a67593b79a4d03747447e6b625t28d36121s1cd82730257d4ffad785277a59c2209'
# url4=url_list_2[0]


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
    
#----------------------------digits in query--------------------------------------------------------
# index7=0
temp_list=[]
url_list_2=[]
def number_of_dots(url):
    # global index7
    count=0
    for character in url:
        if character=='?':
            count=count+1
    temp_list.append(count)
    if count==2:
        url_list_2.append(url)
        
    # index6=index7+1
#creating a dataset
for url in dataset_phishing_url_modified:
    number_of_dots(url)
    
    
temp_list_df=pd.DataFrame(temp_list)
temp_list_df[0].value_counts()
data_initial_13['Query_DigitCount'].value_counts()

#------------------------digits in query #I may have mistake in this-------------------------------------------------------------
def search(content_first):
    count=0
    for char in content_first:
        if char.isdigit():
            count=count+1
    return count
                                                    

index7=0
def digits_in_query(url):
    global index7
    count_question_mark=0
    count=0
    for character in url:
        if character=='?':
            count_question_mark=count_question_mark+1
    
    if count_question_mark==0:
        dataset_13['Query_DigitCount'].iloc[index7]=0
            
    elif count_question_mark==1:
        content=url.split('?')
        if '&' in content[1]:
            content_first=content[1].split('&')
            count=search(content_first[0])
            if count>0:
                dataset_13['Query_DigitCount'].iloc[index7]=count
            else:
                dataset_13['Query_DigitCount'].iloc[index7]=-1
        elif ';' in content[1]:
            content_first=content[1].split(';')
            count=search(content_first[0])
            if count>0:
                dataset_13['Query_DigitCount'].iloc[index7]=count
            else:
                dataset_13['Query_DigitCount'].iloc[index7]=-1
        else:
            count=search(content[1])
            if count>0:
                dataset_13['Query_DigitCount'].iloc[index7]=count
            else:
                dataset_13['Query_DigitCount'].iloc[index7]=-1
            
    elif count_question_mark==2:
        content=url.split('?')
        if '&' in content[1]:
            content_first=content[1].split('&')
            count=search(content_first[0])
            if count>0:
                dataset_13['Query_DigitCount'].iloc[index7]=count
            else:
                dataset_13['Query_DigitCount'].iloc[index7]=-1
        elif ';' in content[1]:
            content_first=content[1].split(';')
            count=search(content_first[0])
            if count>0:
                dataset_13['Query_DigitCount'].iloc[index7]=count
            else:
                dataset_13['Query_DigitCount'].iloc[index7]=-1
        else:
            count=search(content[1])
            if count>0:
                dataset_13['Query_DigitCount'].iloc[index7]=count
            else:
                dataset_13['Query_DigitCount'].iloc[index7]=-1

        if '&' in content[2]:
            content_first=content[2].split('&')
            count=search(content_first[0])
            if count>0:
                dataset_13['Query_DigitCount'].iloc[index7]=count
            else:
                dataset_13['Query_DigitCount'].iloc[index7]=-1
        elif ';' in content[2]:
            content_first=content[2].split(';')
            count=search(content_first[0])
            if count>0:
                dataset_13['Query_DigitCount'].iloc[index7]=count
            else:
                dataset_13['Query_DigitCount'].iloc[index7]=-1
        else:
            count=search(content[2])
            if count>0:
                dataset_13['Query_DigitCount'].iloc[index7]=count
            else:
                dataset_13['Query_DigitCount'].iloc[index7]=-1
    
    index7=index7+1

#creating a dataset
for url in dataset_phishing_url_modified:
    digits_in_query(url)
    
dataset_13['Query_DigitCount'].value_counts()
            
#-------------------Function for longest path token length------------------------------------
index8=0
temp_list=[]
def slash__count(url):
    count=0
    for char in url:
        if char=='/':
            count=count+1
    count=count-2
    # temp_list.append(count)
    return count
    

def find_length(url):
    count=0
    for char in url:
        count=count+1
    return count

def path_length_attach(url,slash_count):
    i=0
    length_list=[]
    content=url.split('/')
    for i in range(0,slash_count):
        length_list.append(find_length(content[i+3]))

    if len(length_list)==0:
        return 0
    else:
        maximum_length=max(length_list)
        return maximum_length
                       

def path_length(url):
    global index8
    slash_count=slash__count(url)
    length=path_length_attach(url,slash_count)
    dataset_13['LongestPathTokenLength'].iloc[index8]=length
    index8=index8+1
    
#creating a dataset 
for url in dataset_phishing_url_modified:
    path_length(url)
    

#----------------------Delimeter Domain------------------------------------------------
data_initial_13['delimeter_Domain'].value_counts()
delimiter_list=['.',',' , ';' , '{' , '}' , '|' , '/' , '+','#','%','<','>','~','(',')' , '[' , ']' , '<' , '>' , '"','<?' , '?>', '/*' , '*/' , '<%', '%>' ]


index9=0
def delimiter_count(url):
    count=0
    global index9
    content=url.split('/')
    
    for x in range(0,3):
        for char in content[x]:
            if char in delimiter_list:
                count=count+1
    
    dataset_13['delimeter_Domain'].iloc[index9]=count
    index9=index9+1
    
for url in dataset_phishing_url_modified:
    delimiter_count(url)

dataset_13['delimeter_Domain'].value_counts()       #I may have done mistake here.
data_initial_13['delimeter_Domain'].value_counts()



#----------------------Delimeter Path--------------------------------
#The symbol in a delimiter list is called a bag of words. For this two delimiter propertie look for paer no 15
delimiter_list=['.',',' , ';' , '{' , '}' , '|' , '/' , '+','#','%','<','>','~','(',')' , '[' , ']' , '<' , '>' , '"','<?' , '?>', '/*' , '*/' , '<%', '%>','?','=','-','_' ]
index10=0
def delimiter_path_count(url):
    count=0
    global index10
    content=url.split('/')
    length_of_content=len(content)
    
    for x in range(3,length_of_content):
        for char in content[x]:
            if char in delimiter_list:
                count=count+1
    
    dataset_13['delimeter_path'].iloc[index10]=count
    index10=index10+1
    
for url in dataset_phishing_url_modified:
    delimiter_path_count(url)
    

#---------------------------------symbol count domain-------------------------
symbol_list=['@',':','//','/','?',',','=',';','(',')','+','[',']']

index11=0
def domain_symbol_count(url):
    count=0
    global index11
    content=url.split('/')
    domain_name=content[2]
    
    for char in domain_name:
        if char in symbol_list:
                count=count+1
    
    dataset_13['SymbolCount_Domain'].iloc[index11]=count
    index11=index11+1
    
for url in dataset_phishing_url_modified:
    domain_symbol_count(url)

dataset_13['SymbolCount_Domain'].value_counts()       #I may have done mistake here.
data_initial_13['SymbolCount_Domain'].value_counts()




 
