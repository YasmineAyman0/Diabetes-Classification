#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
dataset=pd.read_csv('diabetes_data_upload.csv')


# In[2]:


dataset.head()


# In[3]:


res=dataset[dataset['Age']>84]
len(res)


# In[4]:


dataset.shape


# In[5]:


dataset.describe()


# In[6]:


map_class={'Positive':1,'Negative':0}
dataset['class']=dataset['class'].map(map_class)


# In[7]:


plt.scatter(dataset['class'],dataset['Age'])
plt.show()


# In[8]:


dataset.info()


# In[9]:


dict_map={'No':0,'Yes':1}
dict_male={'Male':0,'Female':1}

dataset['Gender']=dataset['Gender'].map(dict_male)
dataset['Polyuria']=dataset['Polyuria'].map(dict_map)

dataset['Polydipsia']=dataset['Polydipsia'].map(dict_map)

dataset['sudden weight loss']=dataset['sudden weight loss'].map(dict_map)

dataset['weakness']=dataset['weakness'].map(dict_map)

dataset['Polyphagia']=dataset['Polyphagia'].map(dict_map)

dataset['Genital thrush']=dataset['Genital thrush'].map(dict_map)

dataset['visual blurring']=dataset['visual blurring'].map(dict_map)

dataset['Itching']=dataset['Itching'].map(dict_map)

dataset['Irritability']=dataset['Irritability'].map(dict_map)

dataset['delayed healing']=dataset['delayed healing'].map(dict_map)

dataset['partial paresis']=dataset['partial paresis'].map(dict_map)

dataset['muscle stiffness']=dataset['muscle stiffness'].map(dict_map)

dataset['Alopecia']=dataset['Alopecia'].map(dict_map)

dataset['Obesity']=dataset['Obesity'].map(dict_map)

dataset.head()


# In[10]:


dataset.isnull().sum()


# In[11]:


dataset.info()


# In[12]:


import seaborn as sns 
import matplotlib.pyplot as plt
x,y=plt.subplots(figsize=(12,12))
sns.heatmap(dataset.corr(),cmap='YlGnBu',linewidth=.5,square=True,annot=True)
plt.show()


# In[13]:


plt.boxplot(dataset['Age'])


# In[14]:


num_out_lier=dataset[(dataset['Age']>84)]
len(num_out_lier)


# In[15]:


Q1=dataset.quantile(0.25)
Q3=dataset.quantile(0.75)
IQR=Q3-Q1

above_outlier=int(Q3['Age']+IQR['Age']*1.5)

minium_outlier=int(Q1['Age']-IQR['Age']*1.5)
min_ten_percent=dataset['Age'].quantile(0.10)
max_ninty_percent=dataset['Age'].quantile(0.90)
print("above_outlier:",above_outlier)
print("minium_outlier:",minium_outlier)
print(dataset['Age'].skew())
dataset['Age']=np.where(dataset['Age']<min_ten_percent,min_ten_percent,dataset['Age'])
dataset['Age']=np.where(dataset['Age']>max_ninty_percent,max_ninty_percent,dataset['Age'])


# In[16]:


# see the outlier by using box plot
plt.boxplot(dataset['Age'])
plt.show


# In[17]:


# handle outlier by removing the above_outlier data
#index=dataset[dataset['Age']>above_outlier].index
#dataset.drop(index,inplace=True)
#print(dataset['Age'].skew())
#dataset.describe()


# In[18]:


#box plot Age to see if the outlier removed or not
#plt.boxplot(dataset['Age'])
#plt.show


# In[19]:


dataset.hist(bins=50,figsize=(20,15))
plt.show()


# In[20]:


dataset.head()


# In[21]:


feature_cols=['Age','Gender','Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity']
X=dataset[feature_cols]
y=dataset['class']
print(X)
print(y)


# In[22]:


# Split dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7) # 80% training and 20% test


# In[23]:


#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=1)
#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)


# In[24]:


#from sklearn.linear_model import LinearRegression
#from sklearn import metrics
#lin_reg=LinearRegression()
#lin_reg.fit(X_train,y_train)
#y_pred = lin_reg.predict(X_test)
# Model Accuracy, how often is the classifier correct?
#print("Error:",metrics.mean_squared_error(y_test,y_pred)*100)
#print("Accuracy:",100-metrics.mean_squared_error(y_test,y_pred)*100)


# In[25]:


#prediction for error
from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(y_test, y_pred)*100

print("Error:",lin_mse)

