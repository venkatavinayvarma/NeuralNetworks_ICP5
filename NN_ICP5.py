#!/usr/bin/env python
# coding: utf-8

# 1. Implement Naïve Bayes method using scikit-learn library
# 2. Use dataset available with name glass
# 3. Use train_test_split to create training and testing part Evaluate the model on test part using score and  classification_report(y_true, y_pred) 

# In[13]:


import pandas as pd    
df = pd.read_csv('glass.csv')   # Read the CSV data


# In[3]:


df.info  # Check data quality


# In[14]:


df.describe # Explore data descriptions


# In[15]:


df.columns.values # Print column names


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


# In[17]:


# Divide data into features and target variable
X = df.drop("Type", axis=1)
Y = df["Type"]


# In[18]:


# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=47)


# In[19]:


gnb = GaussianNB() # Initialize the Gaussian Naive Bayes classifier

gnb.fit(X_train, Y_train) # Training the model with the training set

Y_predi = gnb.predict(X_test)  # Using the trained model on testing the data

accur_knn = round(gnb.score(X_train, Y_train) * 50, 2) # Evaluating the model using accuracy_score and predicted output
print('Accuracy: ', accur_knn)


# In[26]:


print('\nClassification Report: \n', classification_report(Y_test, Y_predi)) # Classification report of the data set


# 2. Implement linear SVM method using scikit library
# 3. Use the same dataset above Use train_test_split to create training and testing part
# 4.  Evaluate the model on test part using score and  classification_report(y_true, y_pred) 

# In[21]:


from sklearn.svm import SVC
svm = SVC() # Initializing the SVM classifier with linear kernel


# In[22]:


svm.fit(X_train, Y_train) # Training the model with the training set

Y_pred = svm.predict(X_test)  # Predicting the target variable for the test set

acc_svm = round(svm.score(X_train, Y_train) * 50, 2)  # Evaluating the model accuracy using score
print('Accuracy: ', acc_svm,'\n')


# In[27]:


print('Classification Report: \n', classification_report(Y_test, Y_pred,zero_division=1)) # Accuracy report from classification_report


# In[ ]:




