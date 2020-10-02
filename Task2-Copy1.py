#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


Data=pd.read_csv(r"C:\Users\hp\Desktop\Task1.csv")


# In[3]:


Data


# In[4]:


Data.shape


# In[5]:


Data.head()


# In[6]:


Data.describe()


# In[8]:


Data.plot(x="Hours",y="Scores",style="o")
plt.title("Hours Vs Percentage")
plt.xlabel("Hours")
plt.ylabel("Percentage")
plt.show()


# In[10]:


X = Data.iloc[:, :-1].values
y = Data.iloc[:, 1].values


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[12]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[13]:


print(regressor.intercept_)


# In[14]:


print(regressor.coef_)


# In[15]:


y_pred = regressor.predict(X_test)


# In[17]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[18]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




