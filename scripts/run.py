
# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from extend import *
from gradient_descent import *
from stochastic_gradient_descent import *
from newton_method import *
from logistic_regression import *


# ## Load the training data into feature matrix, class labels, and event ids:


#from proj1_helpers import *
from proj1_helpers_tanh import *
DATA_TRAIN_PATH = '../data/train.csv'
y, x, ids = load_csv_data(DATA_TRAIN_PATH)


# ## Do your thing crazy machine learning thing here :) ...


tX = np.copy(x)


#data cleaning
mean_by_col = []
for i in range(tX.shape[1]):
    col = tX[:,i]
    places_999 = (col == -999.0)
    mean_col = col[col!=-999.0].mean()
    col[places_999] = mean_col
    tX[:,i] = col
    mean_by_col.append(mean_col)


np.all(tX > 0,axis=0).sum()


#feature expansion on only certain features
positive_columns = tX[:,np.all(tX > 0,axis=0)]
positive_columns_log = np.log(positive_columns)
not_zero_columns = tX[:, np.all(tX != 0,axis=0)]
not_zero_columns_inv = 1 / not_zero_columns


#feature expansion
degree = 7
tX = build_poly(tX, degree)
tX = tX[:,1:]
tX = extend(tX,[ np.cos, np.sin])


#Adding the special feature expansions
tX = np.c_[(tX, positive_columns_log)]
tX = np.c_[(tX, not_zero_columns_inv)]
tX = np.c_[(tX, not_zero_columns_inv**2)]

for i in range(18) :
    for j in range(18) :
        if j > i : 
            tX = np.c_[(tX, tX[:, i] * tX[:, j])]
            tX = np.c_[(tX, tX[:, i + 30] * tX[:, j + 30])]
            tX = np.c_[(tX, tX[:, i + 60] * tX[:, j + 60])]


mean = np.mean(tX,axis = 0)
std = np.std(tX,axis = 0)
tX = (tX-mean)/std


tX = np.c_[(np.ones(tX.shape[0]) , tX)]

from tanh_gradient_descent import *


w = np.zeros((tX.shape[1]) )
max_iters = 500 #choosing number of iterations
gamma = 1.0
w,loss = newton_method(y,tX,w,max_iters,gamma)
loss


# ## Generate predictions and save ouput in csv format for submission:


DATA_TEST_PATH = '../data/test.csv' # TODO: download train data and supply path here 
_, xx, ids_test = load_csv_data(DATA_TEST_PATH)


tX_test = xx.copy()


for i in range(tX_test.shape[1]):
    col = tX_test[:,i]
    places_999 = (col == -999.0)
    mean_col = col[col!=-999.0].mean()
    col[places_999] = mean_col
    tX_test[:,i] = col


positive_columns = tX_test[:,np.all(tX_test > 0,axis=0)]
positive_columns_log = np.log(positive_columns)
not_zero_columns = tX_test[:, np.all(tX_test != 0,axis=0)]
not_zero_columns_inv = 1 / not_zero_columns


positive_columns_log.shape



tX_test = build_poly(tX_test, degree)
tX_test = tX_test[:,1:]
tX_test = extend(tX_test,[np.cos,np.sin])
tX_test = np.c_[(tX_test, positive_columns_log)]
tX_test = np.c_[(tX_test, not_zero_columns_inv)]
tX_test = np.c_[(tX_test, not_zero_columns_inv**2)]

for i in range(18) :
    for j in range(18) :
        if j > i : 
            tX_test = np.c_[(tX_test, tX_test[:, i] * tX_test[:, j])]
            tX_test = np.c_[(tX_test, tX_test[:, i + 30] * tX_test[:, j + 30])]
            tX_test = np.c_[(tX_test, tX_test[:, i + 60] * tX_test[:, j + 60])]
tX_test = (tX_test-mean)/std

tX_test = np.c_[(np.ones(tX_test.shape[0]) , tX_test)]

OUTPUT_PATH = '../data/output.csv' # TODO: fill in desired name of output file for submission
y_pred = predict_labels(w, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)




