#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:29:34 2020

@author: krishna
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#------------------------Preparing the dataset-----------------------------
data_phishing=pd.read_csv('phishing_dataset.csv')
data_phishing=data_phishing['URL']
data_phishing=pd.DataFrame(data_phishing)
data_phishing['url_type']='phishing'
data_phishing['category']='malicious'

data_benign=pd.read_csv('Benign_list_big_final.csv')
data_benign=data_benign['URL']
data_benign=pd.DataFrame(data_benign)
data_benign['url_type']='benign'
data_benign['category']='non-malicious'

data_malware=pd.read_csv('Malware_dataset.csv')
data_malware=data_malware['URL']
data_malware=pd.DataFrame(data_malware)
data_malware['url_type']='malware'
data_malware['category']='malicious'

data_defacement=pd.read_csv('DefacementSitesURLFiltered.csv')
data_defacement=data_defacement['URL']
data_defacement=pd.DataFrame(data_defacement)
data_defacement['url_type']='defacement'
data_defacement['category']='malicious'

data_spam=pd.read_csv('spam_dataset.csv')
data_spam=data_spam['URL']
data_spam=pd.DataFrame(data_spam)
data_spam['url_type']='spam'
data_spam['category']='malicious'

concatenating_dataframes=[data_benign,data_malware,data_phishing,data_spam,data_defacement]
data_final=pd.concat(concatenating_dataframes)  
data_final['url_type'].value_counts()

#--------------------Lexical features----------------
from urllib.parse import urlparse
import re
import urllib
from xm.dom import minidom
import pygeoip
import ipaddress

def check_ip_address(data_final):
    ipv4 = "192.168.2.10"
    count=0
        
    for element in data_final['URL']:
        try:
            ipaddress.ip_address(element)
            count=count+1
        except ValueError:
            pass
    if count==0:
        print('The dataset contains olny URL not any IP address')
          

check_ip_address(data_final)





        
