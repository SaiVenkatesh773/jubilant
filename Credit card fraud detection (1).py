# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 07:21:04 2022

@author: 91701
"""

import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
      
from google.colab import files
        
dataset = (r'C:\Users\91701\Downloads\CreditCardFraudDetection\CreditCardFraudDetection\creditcard.csv')

print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
        
dataset.isna().any()
      
dataset['Class'] = 0 
dataset['Class'] = 1 
        
nfcount=0
notFraud=dataset['Class']
        
for i in range(len(notFraud)):
    if notFraud[i]==0:
        
            nfcount=nfcount+1
nfcount
per_nf=(nfcount/len(notFraud))*100
print('percentage of total not fraud transaction in the dataset: ',per_nf)
      
fcount=0
Fraud=dataset['Class']
for i in range(len(Fraud)):
    if Fraud[i]==1:
           fcount=fcount+1
fcount
per_f=(fcount/len(Fraud))*100
print('percentage of total fraud transaction in the dataset: ',per_f)
      
x=dataset['Time']
y=dataset['Amount']
plt.plot(x, y)
plt.title('Time Vs amount')
plt.figure(figsize=(10,8)
           
           
plt.title("Amount Distribution")
           
sns.distplot(dataset['Amount'],color='red')
     
correlation_metrics = dataset.corr()
fig = plt.figure(figsize = (14, 9))
sns.heatmap(correlation_metrics, vmax = .9, square = True)
plt.show()
      
x=dataset.drop(['Class'], axis = 1)
y=dataset['Class']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)
    
from sklearn.linear_model import LinearRegression
linear =LinearRegression()
linear.fit(xtrain, ytrain)
      
y_pred = linear.predict(xtest)
table= pd.DataFrame({\"Actual\":ytest,\"Predicted\":y_pred})
        
print(table)
