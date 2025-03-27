#!/usr/bin/env python
# coding: utf-8

# In[26]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[27]:


data = pd.read_csv('/content/creditcard.csv')
data.head()


# # New Section

# In[28]:


data.isnull().sum()


# In[29]:


data.describe()


# In[30]:


print('Valid transaction',len(data[data['Class']==0]))
print('fraud transaction',len(data[data['Class']==1]))


# In[31]:


y= data['Class']
x= data.drop(columns=['Class'],axis=1)


# In[32]:


#splitting the data into train test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.3,random_state=0)


# In[33]:


# fitting randomforest model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()


# In[34]:


#model_1
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20,criterion='entropy', random_state=0,max_depth=10)
classifier.fit(x_train,y_train)


# In[35]:


y_pred = classifier.predict(x_test)


# In[36]:


from sklearn.metrics import  classification_report, confusion_matrix
print('Classifcation report:\n', classification_report(y_test, y_pred))
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Confusion matrix:\n', conf_mat)


# In[37]:


#model_2
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=30,criterion='entropy', random_state=0,max_depth=10)
classifier.fit(x_train,y_train)


# In[38]:


y_pred_2 = classifier.predict(x_test)


# In[39]:


from sklearn.metrics import  classification_report, confusion_matrix
print('Classifcation report:\n', classification_report(y_test, y_pred_2))
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred_2)
print('Confusion matrix:\n', conf_mat)


# In[43]:


from sklearn.tree import DecisionTreeClassifier
# Train a CART classifier
cart_classifier = DecisionTreeClassifier(random_state=42)
cart_classifier.fit(x_train, y_train)


# In[46]:


rf_predictions = classifier.predict(x_test)
cart_predictions = cart_classifier.predict(x_test)


# In[49]:


print("Random Forest Classifier:")
print(classification_report(y_test, rf_predictions))

print("CART Classifier:")
print(classification_report(y_test, cart_predictions))


# In[47]:


#trying with undersmapling technique
# This is the pipeline module we need from imblearn for Undersampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# Define which resampling method and which ML model to use in the pipeline
resampling = RandomUnderSampler()
model = RandomForestClassifier(n_estimators=30,criterion='entropy', random_state=0,max_depth=10)


# Define the pipeline and combine sampling method with the RF model
pipeline = Pipeline([('RandomUnderSampler', resampling), ('RF', model)])
pipeline.fit(x_train, y_train)
predicted = pipeline.predict(x_test)


# Obtain the results from the classification report and confusion matrix
print('Classifcation report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)


# In[51]:


# This is the pipeline module we need from imblearn for Oversampling
from imblearn.over_sampling import RandomOverSampler
# Define which resampling method and which ML model to use in the pipeline
resampling = RandomOverSampler()
model = RandomForestClassifier(n_estimators=30,criterion='entropy', random_state=0,max_depth=10)

# Define the pipeline and combine sampling method with the model
pipeline = Pipeline([('RandomOverSampler', resampling), ('RF', model)])
pipeline.fit(x_train, y_train)
predicted = pipeline.predict(x_test)


# Obtain the results from the classification report and confusion matrix
print('Classifcation report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)


# In[52]:


# This is the pipeline module we need from imblearn for SMOTE
from imblearn.over_sampling import SMOTE
# Define which resampling method and which ML model to use in the pipeline
resampling = SMOTE(sampling_strategy='auto',random_state=0)
model = RandomForestClassifier(n_estimators=30,criterion='entropy', random_state=0,max_depth=10)

# Define the pipeline and combine sampling method with the model
pipeline = Pipeline([('SMOTE', resampling), ('RF', model)])
pipeline.fit(x_train, y_train)
predicted = pipeline.predict(x_test)


# Obtain the results from the classification report and confusion matrix
print('Classifcation report:\n', classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)


# In[53]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[54]:


#visulalizing the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[55]:


#visulalizing the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred_2)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[59]:


# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("Random Forest Confusion Matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')

sns.heatmap(confusion_matrix(y_test, cart_predictions), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title("CART Confusion Matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')

plt.show()

