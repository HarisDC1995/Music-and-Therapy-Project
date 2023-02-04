#!/usr/bin/env python
# coding: utf-8

# # Music and Therapy 
Importing the basic Python Libraries
# In[1]:


#importing the basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as mano
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read the CSV file
df = pd.read_csv("Music-and-Therapy.csv")
df


# In[3]:


# Convert the data into Pandas dataframe
data = pd.DataFrame(df)


# # Checked the Data Types of the columns

# In[4]:


#see data types
data.dtypes


# In[5]:


#to find rows and columns
data.shape


# # To Find Out Missing Values

# In[6]:


data.isnull().sum()

Using this code to find about the dublicate rows in the data set
# In[7]:


duplicateRows = data[data.duplicated()]
duplicateRows

It shows that there were no dublicate data in the dataset
# In[8]:


data.describe()   


# # Filling the Missing Values
Filling BPMThe BPM can be filled by the linear interpolation. It is more logical to fill the missing values through linear interpolation methold
# In[9]:


data['BPM'] = data['BPM'].interpolate(method ='linear', limit_direction ='forward')
data['BPM'].hist()


# In[10]:


data['BPM'].plot()

Filling the missing values Age  
# In[11]:


data['Age'] = data['Age'].interpolate(method ='linear', limit_direction ='forward')
data['Age'].hist()


# # Ask you can see that BPM and Age values are Filled with linear method

# In[12]:


data.isnull().sum()


# In[13]:


def detect_outliers(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    outliers = []
    for x in data:
        z_score = (x - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(x)
    return outliers

There is no outliers in the dataset since print outliers command doesn't return any result
# In[14]:


#ploted the heatmap to check positive correlations between the variables
mano.heatmap(data, figsize=(20,6))


# Plot the Scatter Chart to show the usage of music hours per day based on age

# In[15]:


#In this Chart i show the usage of Hours per day according to age
data.plot(x='Age',y='Hours per day',kind='scatter',alpha=0.5,cmap='rainbow')


# Plot the Scatter Chart to show the level of depression based on age

# In[16]:


#In this Chart is show the depression in people according to the Age (values filled in by mean)
data.plot(x='Age',y='Depression',kind='scatter',alpha=0.5,cmap='rainbow')

Plot the Box plot to show the Age of people who listen to music while working and also who don't
# In[17]:


plt.figure(figsize=(10, 5))
sns.boxplot(x = data['Age'], y = data['While working'])
plt.title("Age vs While Working")


# # Now Dropping the missing values by dropping the cells that doesn't contain the value
Filling the Missing Values
# In[18]:


df2=data.copy()


# In[19]:


df2.dropna(axis=0,inplace=True)


# In[20]:


df2.isnull().sum()


# In[21]:


df2


# # Generate Profiling Report Code:

# In[22]:


import pandas_profiling as pp
pp.ProfileReport(df2)

