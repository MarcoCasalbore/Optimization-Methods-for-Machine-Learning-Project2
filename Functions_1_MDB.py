# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:53:39 2023

@author: Bianca
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from cvxopt import matrix, solvers

C = 1
gamma = 2
tolerance = 1e-5

def lettura_dati(file_train, file_test): 
    train = pd.read_csv(file_train)
    test = pd.read_csv(file_test)

    indexes = [2,3]
 
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

    X2 = X[ind_2[0][:1000]]
    #etichetta 2 
    Y2 = np.ones(1000) 

    X3 = X[ind_3[0][:1000]] 
    Y3 = -np.ones(1000)  

    X2test = X_Test[ind_2_Test[0][:200]]
    Y2test = np.ones(X2test.shape[0]) 

    X3test = X_Test[ind_3_Test[0][:200]]
    Y3test = -np.ones(X3test.shape[0])


    X = np.concatenate((X2,X3))
    Y = np.concatenate((Y2,Y3))
    X_test = np.concatenate((X2test,X3test))
    Y_test = np.concatenate((Y2test,Y3test)) 
    return X, Y, X_test, Y_test 

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

solvers.options['show_progress'] = False  

def optimization(X, Y, gamma, tolerance, C): 
    kernel = polynomial_kernel(X, X, gamma)  
    Q = matrix((kernel * Y.reshape(-1,1)) * Y) #Q MATRIX 
    
    q = matrix(-np.ones(len(X))) # q VECTOR 
    
    G_ones = np.eye(len(X)) 
    G_minus_ones = np.diag(-1 * np.ones(len(X))) 
    G_tot = np.concatenate((G_ones, G_minus_ones)) #G MATRIX
    G = matrix(G_tot)
    
    h_C = C*np.ones(len(X)).reshape(-1,1) 
    h_zero = np.zeros(len(X)).reshape(-1,1) 
    h = matrix(np.concatenate((h_C, h_zero))) #h VECTOR 
    
    A = matrix(Y.reshape(1,-1))
    b = matrix(np.zeros(1)) 
    
    sol = solvers.qp(Q,q, G, h, A, b)  
    alfa_star = np.array(sol['x']) 
    
    return sol, alfa_star 

parametri_grid = {"C": [1, 10, 100], "gamma":[2,3,4,5]}    
def grid_search(k, parametri_grid, X, Y, tolerance):
    # X = X_train
    # Y = Y_train
    accuracy = 0
    kf = KFold(k, random_state = 1836553, shuffle = True) 
    for i in range(len(parametri_grid['C'])):
        c = parametri_grid['C'][i]
        for j in range(len(parametri_grid['gamma'])): 
            Gamma = parametri_grid['gamma'][j] 
            acc_list = []
            for train_index, test_index in kf.split(X):
                X_train_cv, X_validation = X[train_index], X[test_index]
                Y_train_cv, Y_validation = Y[train_index], Y[test_index]  
                 
                X_train_cv, X_validation = normalization(X_train_cv, X_validation)
                
                kernel = polynomial_kernel(X_train_cv, X_train_cv, Gamma)  
                Q = matrix((kernel * Y_train_cv.reshape(-1,1)) * Y_train_cv) 
                
                q = matrix(-np.ones(len(X_train_cv))) 
                
                G_ones = np.eye(len(X_train_cv)) 
                G_minus_ones = np.diag(-1 * np.ones(len(X_train_cv))) 
                G_tot = np.concatenate((G_ones, G_minus_ones)) 
                G = matrix(G_tot)
                
                h_C = c*np.ones(len(X_train_cv)).reshape(-1,1)
                h_zero = np.zeros(len(X_train_cv)).reshape(-1,1) 
                h = matrix(np.concatenate((h_C, h_zero)))  
                
                A = matrix(Y_train_cv.reshape(1,-1))
                b = matrix(np.zeros(1))  
                
                sol = solvers.qp(Q,q, G, h, A, b) 
                alfa_star = np.array(sol['x'])  
                
                validation_prediction = decision_fun(X_train_cv, X_validation, Y_train_cv, Gamma, alfa_star, tolerance, c)
                validation_accuracy = np.sum(validation_prediction.ravel() == Y_validation.ravel())/Y_validation.shape[0]
                acc_list = np.append(acc_list, validation_accuracy) 
                
            mean_acc = np.mean(acc_list) 
            if mean_acc > accuracy: 
                accuracy = mean_acc 
                C_ottimo = parametri_grid['C'][i] 
                Gamma_ottimo =  parametri_grid['gamma'][j] 
                
    return C_ottimo, Gamma_ottimo, accuracy 


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

    
