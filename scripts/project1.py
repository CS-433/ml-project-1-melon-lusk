#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Useful starting lines
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from extend import *
from gradient_descent import *
from stochastic_gradient_descent import *
from hessiant_descent import *
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


get_ipython().system('jupyter nbconvert --to script project1.ipynb')


# ## Load the training data into feature matrix, class labels, and event ids:

# In[2]:


#from proj1_helpers import *
from proj1_helpers_tanh import *
DATA_TRAIN_PATH = '../data/train.csv'
y, x, ids = load_csv_data(DATA_TRAIN_PATH)


# ## Do your thing crazy machine learning thing here :) ...

# In[19]:


tX = np.copy(x)


# In[20]:


mean_by_col = []
for i in range(tX.shape[1]):
    col = tX[:,i]
    places_999 = (col == -999.0)
    mean_col = col[col!=-999.0].mean()
    col[places_999] = mean_col
    tX[:,i] = col
    mean_by_col.append(mean_col)


# In[21]:


np.all(tX > 0,axis=0).sum()


# In[22]:


positive_columns = tX[:,np.all(tX > 0,axis=0)]
positive_columns_log = np.log(positive_columns)


# In[23]:


positive_columns.shape


# In[24]:


degree = 7
tX = build_poly(tX, degree)
print(tX.shape)
tX = tX[:,1:]
print(tX.shape)
tX = extend(tX,[ np.cos, np.sin, ])
print(tX.shape)


# In[25]:


tX = np.c_[(tX, positive_columns_log)]


# In[26]:


tX.shape


# In[27]:


mean = np.mean(tX,axis = 0)
std = np.std(tX,axis = 0)
tX = (tX-mean)/std


# In[28]:


mean.shape


# In[29]:


tX = np.c_[(np.ones(tX.shape[0]) , tX)]


# In[30]:


tX.shape


# In[1]:


from tanh_gradient_descent import *


# In[33]:


w = np.zeros((tX.shape[1]) )
max_iters = 10000 #choosing number of iterations
gamma = 5 * 1e-2
#w,loss = adaptative_step_gradient_descent(y, tX, w, max_iters, gamma)
w,loss = tanh_gradient_descent(y,tX,w,max_iters,gamma)


# In[ ]:


loss


# ## Generate predictions and save ouput in csv format for submission:

# In[ ]:


DATA_TEST_PATH = '../data/test.csv' # TODO: download train data and supply path here 
_, xx, ids_test = load_csv_data(DATA_TEST_PATH)


# In[ ]:


tX_test = xx.copy()


# In[ ]:


tX = tX_test


# In[ ]:


for i in range(tX.shape[1]):
    col = tX[:,i]
    places_999 = (col == -999.0)
    mean_col = col[col!=-999.0].mean()
    col[places_999] = mean_col
    tX[:,i] = col


# In[ ]:


positive_columns = tX_test[:,np.all(tX_test > 0,axis=0)]
positive_columns_log = np.log(positive_columns)


# In[ ]:


positive_columns_log.shape


# In[ ]:


tX_test = tX


# In[ ]:


tX_test = build_poly(tX_test, degree)
print(tX_test.shape)
tX_test = tX_test[:,1:]
print(tX_test.shape)
#tX_test = extend(tX_test,[np.cos,np.sin])
#print(tX_test.shape)
tX_test = np.c_[(tX_test, positive_columns_log)]
print(tX_test.shape)
tX_test = (tX_test-mean)/std


# In[ ]:


tX_test.shape


# In[ ]:


tX_test = np.c_[(np.ones(tX_test.shape[0]) , tX_test)]


# In[ ]:


OUTPUT_PATH = '../data/output.csv' # TODO: fill in desired name of output file for submission
y_pred = predict_labels(w, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

