
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


data=pd.read_csv(r"D:\DataSets-master\DataSets-master/Data_MVLR.csv")


# In[4]:


data


# In[5]:


data.tail(5)


# In[6]:


x_input=data.drop(["Petrol_Consumption"],axis=1)


# In[7]:


y_target=data["Petrol_Consumption"]


# In[8]:


x_pinput=x_input.values.reshape(len(x_input),4)


# In[9]:


y_ptarget=y_target.values.reshape(len(y_target),1)


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x_input,y_target,test_size=0.3)


# In[12]:


from sklearn.linear_model import LinearRegression


# In[13]:


myobj=LinearRegression()


# In[14]:


mymodel=myobj.fit(x_train,y_train)


# In[15]:


ya=y_test


# In[16]:


yp=mymodel.predict(x_test)


# In[17]:


ya


# In[18]:


from sklearn import metrics


# In[19]:


import numpy as np


# In[20]:


error=metrics.mean_squared_error(y_test,yp)


# In[21]:


np.sqrt(error)


# In[24]:


mymodel.intercept_


# In[25]:


mymodel.coef_


# In[26]:


from matplotlib import pyplot as plt


# In[27]:


plt.plot(x_test,y_test,'r*')


# In[29]:


plt.plot(x_test,yp)

