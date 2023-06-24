#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("WineQT.csv")


# In[3]:


data.head(10)


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


data.groupby('quality').mean()


# In[8]:


corr = data.corr()


# In[9]:


plt.figure(figsize=(12,7))
sns.heatmap(corr,cmap="coolwarm",annot=True)
plt.show()


# In[10]:


sns.countplot(data['quality'])
plt.show()


# In[11]:


data.hist(figsize=(15,13),bins=50)
plt.show()


# In[12]:


sns.pairplot(data)


# In[13]:


data['goodquality'] = [1 if x >= 7 else 0 for x in data['quality']]
X = data.drop(['quality','goodquality'], axis = 1)
y = data['goodquality']


# In[15]:


data.head(17)


# In[16]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
X_train


# In[17]:


y_test


# In[18]:


from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()

model1.fit(X_train,y_train)

prediction1 = model1.predict(X_test)

prediction1


# In[19]:


from sklearn.metrics import accuracy_score,confusion_matrix

print(accuracy_score(prediction1,y_test))


# In[20]:


from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier()

model2.fit(X_train,y_train)

prediction2 = model2.predict(X_test)

prediction2


# In[21]:


print(accuracy_score(prediction2,y_test))


# In[ ]:




