#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd 
dataset = pd.read_csv('./AccessLogDataset.csv')
dataset


# In[ ]:





# In[25]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(dataset, test_size=0.4, shuffle=True, random_state=100)


# In[23]:


dataset.isna().sum()


# In[31]:


dataset['gmt'].unique()


# In[34]:


dataset1 = dataset.drop(['gmt','browser'], axis=1)
dataset1


# In[39]:


data = dataset1.dropna()
data


# In[41]:


data.to_csv('test_access_tokens.csv')


# ## 오토 인코더 

# In[2]:


import pandas as pd 
dataset = pd.read_csv('test_access_tokens.csv')
dataset.columns


# In[3]:


df = dataset[['ip', 'datetime', 'request', 'status', 'size', 'referer',
       'country', 'detected']]
df

