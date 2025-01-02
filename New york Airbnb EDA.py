#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data = pd.read_csv('newyork_dataset.csv', encoding_errors='ignore')


# In[6]:


data.head()


# In[7]:


data.shape


# In[8]:


data.info()


# In[9]:


# Statistical Summary
data.describe()


# Task 4: Data Cleaning

# In[10]:


data.isnull().sum()


# In[11]:


# dropping all missing values rows
data.dropna(inplace=True)

# data.fillna()
data.isnull().sum()


# In[12]:


# dealing with duplicates rows
data.duplicated().sum()


# In[13]:


data.drop_duplicates(inplace=True)
data.duplicated().sum()


# In[14]:


# type casting 
# changing data types 

data.dtypes

data['id'] = data['id'].astype(object)
data.dtypes

data['host_id'] = data['host_id'].astype(object)
data.dtypes


# EDA
# Task 5: Data Analysis
# 
# Univariate Analysis

# In[15]:


# idenfying outliers in price

df = data[data['price'] < 1500]

sns.boxplot(data=df, x='price')


# In[16]:


#Price distribuion

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='price', bins=50)
plt.ylabel("Frequency")


# In[17]:


df.dtypes


# In[18]:


#Price distribuion
plt.figure(figsize=(6, 3))
sns.histplot(data=df, x='availability_365')
plt.ylabel("Frequency")


# In[19]:


data.dtypes


# In[24]:


df.groupby(by='neighbourhood_group')['price per head'].mean()


# In[20]:


df.groupby(by='neighbourhood_group')['price'].mean()


# # Feature Engineering

# In[21]:


df.head()


# In[29]:


#['price per bed']
df['price per bed']=df['price']/df['beds']
df.head()


# In[30]:


df=df.drop('price per head', axis=1)


# In[31]:


df.head()


# In[33]:


sns.barplot(data=df,x='neighbourhood_group',y='price',hue='room_type')


# In[34]:


#number of reviews and price relation
sns.scatterplot(data=df,x='number_of_reviews',y='price',hue='neighbourhood_group')


# In[35]:


sns.pairplot(data=df,vars=['price', 'minimum_nights', 'number_of_reviews', 'availability_365'], hue='room_type')


# In[36]:


#geopolitical distribution
sns.scatterplot(data=df, x='longitude', y='latitude', hue='room_type')
plt.title("Geographical Distribution of AirBnb Listing")
plt.show()


# In[37]:


#heat map- correlation of one variable with others for numerical column
corr = df[['latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365', 'beds']].corr()
corr


# In[38]:


sns.heatmap(data=corr, annot=True)


# In[ ]:




