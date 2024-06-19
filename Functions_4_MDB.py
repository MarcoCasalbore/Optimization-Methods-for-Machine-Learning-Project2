# -*- coding: utf-8 -*- 
"""
Created on Sat Dec 23 18:02:05 2023

@author: Bianca
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from cvxopt import matrix, solvers

C = 1
gamma = 2
tolerance = 1e-5

def lettura_dati(file_train, file_test): 
    train = pd.read_csv(file_train)
    test = pd.read_csv(file_test)

    indexes = [2,3,6]
 
    train_set = train[train['label'].isin(indexes)]
    test_set = test[test['label'].isin(indexes)]

    X = train_set.values[:,1:]
    Y = train_set.values[:,0]

    X_Test = test_set.values[:,1:]
    Y_Test = test_set.values[:,0]

    ind_2 = np.where(Y==2)
    ind_2_Test = np.where(Y_Test==2) 

    ind_3 = np.where(Y==3)
    ind_3_Test = np.where(Y_Test==3)

    ind_6 = np.where(Y==6)
    ind_6_Test = np.where(Y_Test==6) 


    X2 = X[ind_2[0][:1000]]
    Y2 = Y[ind_2[0][:1000]]
    
    X3 = X[ind_3[0][:1000]] 
    Y3 = Y[ind_3[0][:1000]]
    
    X6 = X[ind_6[0][:1000]] 
    Y6 = Y[ind_6[0][:1000]]
  
    X2test = X_Test[ind_2_Test[0][:200]]
    Y2test = Y_Test[ind_2_Test[0][:200]]

    X3test = X_Test[ind_3_Test[0][:200]]
    Y3test = Y_Test[ind_3_Test[0][:200]]

    X6test = X_Test[ind_6_Test[0][:200]]
    Y6test = Y_Test[ind_6_Test[0][:200]] 

    X = np.concatenate((X2,X3,X6))
    Y = np.concatenate((Y2, Y3,Y6))
    X_test = np.concatenate((X2test,X3test,X6test))
    Y_test = np.concatenate((Y2test, Y3test, Y6test))
    return X, Y, X_test, Y_test 


def binary_class(x, label_value):
    out = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] == label_value:
            out[i]=1
        else:
            out[i]=-1 
    return out 

def converter(y):
    out = np.zeros((len(y),1))
    for i in range(len(y)):
        if y[i] == 2: 
            out[i] = 0
            
        elif y[i] == 3:
            out[i] = 1
            
        elif y[i] == 6:
            out[i] = 2
    return out 

def normalization(x_train,x_test): 
    norm = MinMaxScaler()
    x_train = norm.fit_transform(x_train)
    x_test = norm.transform(x_test)
    return x_train,x_test

def polynomial_kernel(x1, x2, gamma):
    product = np.dot(x1, x2.T)   
    return (product + 1)**gamma  

def support_vectors(alfa, tolerance, C):
    condition = np.logical_and(alfa > tolerance, alfa < C - tolerance)
    return np.where(condition)   

def decision_fun(X, x, Y, gamma, alfa, tolerance, C): 
    index_sv = support_vectors(alfa, tolerance, C)[0]
    kernel = polynomial_kernel(X, x, gamma) 
    product = (alfa * Y.reshape(-1,1))
    result = np.sum(kernel * product, axis = 0)
    number_sv = len(index_sv)
    kernel_sv = polynomial_kernel(X, X[index_sv], gamma) 
    try:
        bias = (1/number_sv) * (np.sum(Y[index_sv] - np.sum(kernel_sv * (alfa * Y.reshape(-1,1)), axis = 0)))
    except:
        bias = 0 
    
    return np.sign(result + bias)

def M_value(alfa, Y, tolerance, C, kernel):
    Q = (kernel * Y.reshape(-1,1)) * Y
    y = Y.reshape(-1,1)
    grad = -(np.dot(Q, alfa) - 1) * y
    S = np.where(np.logical_or(np.logical_and(alfa <= C-tolerance, y == -1), np.logical_and(alfa >= tolerance, y == 1)))[0]
    M = np.min(grad[S]) 
    return M 

def m_value(alfa, Y, tolerance, C, kernel):
    Q = (kernel * Y.reshape(-1,1)) * Y 
    y = Y.reshape(-1,1)
    grad = -(np.dot(Q, alfa) - 1) * y
    R = np.where(np.logical_or(np.logical_and(alfa <= C - tolerance, y == 1), np.logical_and(alfa >= tolerance , y == -1)))[0]
    m = np.max(grad[R]) 
    return m 

def objective_function(alfa, gamma, X, Y):  
    K = polynomial_kernel(X,X, gamma)
    Q = (K * Y.reshape(-1,1)) * Y
    Q_total = (Q * alfa) * alfa.reshape(1,-1)
    
    return (1/2) * (np.sum(np.sum(Q_total, axis = 0))) - np.dot(np.ones((1,len(alfa))),alfa)

solvers.options['show_progress'] = False
def optimization(X, Y, gamma, tolerance, C, label): 
    # X = X_train_scaled 
    Y_new = binary_class(Y, label)
    kernel = polynomial_kernel(X, X, gamma)  
    Q = matrix((kernel * Y_new.reshape(-1,1)) * Y_new) 
    
    q = matrix(-np.ones(len(X))) # q VECTOR 
    
    G_ones = np.eye(len(X)) 
    G_minus_ones = np.diag(-1 * np.ones(len(X))) 
    G_tot = np.concatenate((G_ones, G_minus_ones)) 
    G = matrix(G_tot)
    
    h_C = C*np.ones(len(X)).reshape(-1,1)
    h_zero = np.zeros(len(X)).reshape(-1,1) 
    h = matrix(np.concatenate((h_C, h_zero))) 
    
    A = matrix(Y_new.reshape(1,-1))
    b = matrix(np.zeros(1)) 
    
    sol = solvers.qp(Q, q, G, h, A, b) 
    alfa_star = np.array(sol['x'])  
    iterations_counter = sol['iterations'] 
    training_prediction = decision_fun(X, X, Y_new, gamma, alfa_star, tolerance, C) 

    return training_prediction, alfa_star , Y_new, iterations_counter 
 
def multi_class(X_Train, X_Test, Y_train ,Y_test, tolerance, gamma, C, labels = [2,3,6]):
    training_prediction_2, alfa_star_2, Y_2, iterations_counter_2  = optimization(X_Train, Y_train, gamma, tolerance, C, labels[0])
    training_prediction_3, alfa_star_3, Y_3, iterations_counter_3 = optimization(X_Train, Y_train, gamma, tolerance, C, labels[1])
    training_prediction_6, alfa_star_6, Y_6, iterations_counter_6 = optimization(X_Train, Y_train, gamma, tolerance, C, labels[2])
    
    pred_train_2_new = training_prediction_2.reshape(1,-1)
    pred_train_3_new = training_prediction_3.reshape(1,-1)
    pred_train_6_new = training_prediction_6.reshape(1,-1) 
    training_prediction_tot = np.concatenate((pred_train_2_new, pred_train_3_new, pred_train_6_new))
      
    class_train = training_prediction_tot.argmax(axis = 0)  
    y_converted_train = converter(Y_train) 
    training_accuracy = np.sum(class_train.ravel() == y_converted_train.ravel())/y_converted_train.shape[0]
    
    test_prediction_2 = decision_fun(X_Train, X_Test, Y_2, gamma, alfa_star_2, tolerance, C)
    test_prediction_3 = decision_fun(X_Train, X_Test, Y_3, gamma, alfa_star_3, tolerance, C)
    test_prediction_6 = decision_fun(X_Train, X_Test, Y_6, gamma, alfa_star_6, tolerance, C)
    
    test_prediction_tot = np.concatenate((test_prediction_2.reshape(1,-1), test_prediction_3.reshape(1,-1), test_prediction_6.reshape(1,-1)))
    class_test = test_prediction_tot.argmax(axis = 0).reshape(-1,1)
    
    y_converted_test = converter(Y_test) 
    test_accuracy = np.sum(class_test.ravel()==y_converted_test.ravel())/y_converted_test.shape[0]
    iterations_total = iterations_counter_3 + iterations_counter_3 + iterations_counter_6 
    
    conf_matrix = confusion_matrix(y_converted_test.ravel(), class_test.ravel())
    
    return training_accuracy, test_accuracy, iterations_total, conf_matrix  
 




