#!/usr/bin/env python
# coding: utf-8

# # Loan Prediction

# <font size="4"><I>
# Loans are financial aid given to one person or organization by another person or organization. However, it has advantages and disadvantages of its own. One of these is the trust that a money lender ought to have in a load borrower. In terms of collateral and monthly income, this trust must meet certain conditions. This notebook serves as merely a straightforward illustration of such machine learning validation.
# </I> </font>

# 
# ![loan%20.jpeg](attachment:loan%20.jpeg)

# In[1]:


# Processing Libraries
import pandas as pd
import numpy as np
# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
# ML Libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# ML Models
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# warnig
import warnings
warnings.filterwarnings("ignore")


# In[2]:


train = pd.read_csv("/Users/harshsanjayshah/Downloads/train_u6lujuX_CVtuZ9i.csv")


# In[3]:


test = pd.read_csv("/Users/harshsanjayshah/Downloads/test_Y3wMUE5_7gLdaTN.csv")


# ## Data Exploration 

# ### Train Dataset

# In[4]:


train.head()


# In[5]:


train.shape


# In[6]:


train.info()


# In[7]:


train.describe()


# In[8]:


train.isna().sum()


# ### Test Dataset

# In[9]:


test.head()


# In[10]:


test.shape


# In[11]:


test.info()


# In[12]:


test.isna().sum()


# ### Filling the null values

# In[13]:


null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount','Dependents', 'Loan_Amount_Term', 'Gender', 'Married']


# In[14]:


for col in null_cols:
    print(f"{col}:\n{train[col].value_counts()}\n")
    train[col] = train[col].fillna(
    train[col].dropna().mode().values[0] )


# ### Encoding categorical values to numeric values

# In[16]:


train.dtypes


# In[17]:


train.drop(['Loan_ID'],axis =1,inplace =True)
test.drop(['Loan_ID'],axis =1,inplace =True)


# In[18]:


to_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2,'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}


# In[19]:


train= train.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)


# In[20]:


test = test.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)


# In[21]:


Dependents_train = pd.to_numeric(train.Dependents)
Dependents_test = pd.to_numeric(test.Dependents)


# In[22]:


train.drop(['Dependents'], axis = 1, inplace = True)
test.drop(['Dependents'], axis = 1, inplace = True)


# In[23]:


train = pd.concat([train, Dependents_train], axis = 1)
test  = pd.concat([test, Dependents_test], axis = 1)


# In[24]:


train.info()


# In[25]:


test.info()


# ## Data Visualization

# In[41]:


plt.figure(figsize=(30,15))
sns.pairplot(train)
plt.show()


# In[46]:


sns.heatmap(train.corr() ,cmap='rainbow_r')


# In[52]:


corr = train.corr()
corr.style.background_gradient(cmap="Pastel2_r")


# ## ML model

# In[27]:


X = train.drop('Loan_Status', axis=1)
y = train['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)


# ### Decision Tree

# In[28]:


model_DT = DecisionTreeClassifier()
model_DT.fit(X_train, y_train)
y_predict = model_DT.predict(X_test)
print(classification_report(y_test,y_predict))




# In[29]:


model_DT_accuracy = accuracy_score(y_predict,y_test)
print(f"{round(model_DT_accuracy *100,2)} % Accuracte")


# ### Random Forest

# In[30]:


model_RF = RandomForestClassifier()
model_RF.fit(X_train,y_train)
y_predict = model_RF.predict(X_test)
print(classification_report(y_test,y_predict))


# In[31]:


model_RF_accuracy = accuracy_score(y_predict,y_test)
print(f"{round(model_RF_accuracy *100,2)} % Accuracte")


# ### XGBoost 

# In[32]:


model_XGB = XGBClassifier()
model_XGB.fit(X_train, y_train)
y_predict = model_XGB.predict(X_test)
print(classification_report(y_test,y_predict))


# In[33]:


model_XGB_accuracy = accuracy_score(y_predict,y_test)
print(f"{round(model_XGB_accuracy*100,2)}% Accurate")


# ### Logistic Regression

# In[34]:


model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)
y_predict = model_LR.predict(X_test)
print(classification_report(y_test,y_predict))


# In[35]:


model_LR_accuracy = accuracy_score(y_predict,y_test)
print(f"{round(model_LR_accuracy*100,2)}% Accurate")


# ## Model Scores

# In[36]:


scores = [model_DT_accuracy,model_RF_accuracy,model_XGB_accuracy,model_LR_accuracy]

models = pd.DataFrame({
    'n_neighbors': ["Decision Tree","Random Forest","XGBoost", "Logistic Regression"],
    'Score': scores
})


# In[37]:


models['Score'] *= 100


# In[38]:


models.sort_values(by = 'Score',ascending=False)


# <font size="4"><I><B>
# The Logisitic Regression Model is the most accurate model : 83.24% Accurate
#     </font></B></I>
# 
