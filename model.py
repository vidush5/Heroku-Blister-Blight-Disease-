#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Required modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#To load the graph in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Load the data set using pandas 
data = pd.read_csv('C:/Users/Vidush/Desktop/Blister Blight Disease Prediction/dataset.csv')

#creating dataframe 
df = pd.DataFrame(data)


# In[4]:


#Print the first five datas in the dataset
data.head()


# In[5]:


#Print the shape of the dataset
np.shape(data)


# In[6]:


df['labels'].unique()


# In[7]:


sns.countplot(x='labels', data=data)


# In[8]:


sns.pairplot(data)


# In[10]:


#Create train-test split
X = data[['soil_moisture','humidity','temperature']]
y = data['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[11]:


#Create classifier object
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)


# In[12]:


#Train the classifier (fit the estimator) using the training data
knn.fit(X_train, y_train)


# In[13]:


#Estimate the accuracy of the classifier on future data, using the test data
knn.score(X_test,y_test)


# ## Use the trained k-NN classifier model to classify new, previously unseen objects

# In[21]:


disease_prediction = knn.predict([[66, 70, 33.2]])


# In[23]:


print(disease_prediction)


# In[24]:


# Saving model to disk
import pickle
pickle.dump(knn, open('E:/Flask Webapp/model.pkl','wb'))


# In[25]:


# Loading model to compare the results
model = pickle.load(open('E:/Flask Webapp/model.pkl','rb'))


# In[26]:


print(model.predict([[66, 70, 33.2]]))


# In[27]:


print(model.predict([[68, 81, 23]]))


# In[28]:


print(model.predict([[66, 61, 33.1]]))


# In[ ]:




