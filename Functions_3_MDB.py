# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:34:19 2023

@author: Bianca
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from cvxopt import matrix, solvers
from sklearn.preprocessing import MinMaxScaler

C = 1 
gamma = 2
tolerance = 1e-4
epsilon = 1e-5
q = 2 

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

def objective_function(alfa, gamma, X, Y):  
    K = polynomial_kernel(X,X, gamma)
    Q = (K * Y.reshape(-1,1)) * Y
    Q_total = (Q * alfa) * alfa.reshape(1,-1)
    
    return (1/2) * (np.sum(np.sum(Q_total, axis = 0))) - np.dot(np.ones((1,len(alfa))),alfa)

def gradient(alfa, Y, C, kernel):
    # Y = Y_train
    Q = (kernel * Y.reshape(-1,1)) * Y
    y = Y.reshape(-1,1)
    grad = -(np.dot(Q, alfa) - 1) * y 
    
    return grad
 
def m_value(alfa, Y, tolerance, C,gradient, K,q):
    gradient = gradient.reshape((len(alfa), 1))
    R = np.where(np.logical_or(np.logical_and(alfa <= C-tolerance, Y==1), np.logical_and(alfa >= tolerance ,Y == -1)))[0]
    m_grad = -gradient[R] * Y[R]
    m = np.max(m_grad)
    
    return m 

def M_value(alfa, Y, tolerance, C, gradient, K,q):
    gradient = gradient.reshape((len(alfa), 1))
    S = np.where(np.logical_or(np.logical_and(alfa <= C-tolerance, Y==-1), np.logical_and(alfa >= tolerance ,Y == 1)))[0]
    M_grad = - gradient[S] * Y[S]
    M = np.min(M_grad)
    
    return M 

def Working_set(alfa, Y, tolerance, C, gradient, q):
    gradient = gradient.reshape((len(alfa), 1)) 
    S = np.where(np.logical_or(np.logical_and(alfa <= C-tolerance, Y==-1), np.logical_and(alfa >= tolerance ,Y == 1)))[0]
    R = np.where(np.logical_or(np.logical_and(alfa <= C-tolerance, Y==1), np.logical_and(alfa >= tolerance ,Y == -1)))[0]
    M_grad = - gradient[S] * Y[S]
    m_grad = -gradient[R] * Y[R]
    q2=np.argsort(M_grad.ravel())[0:int(q/2)]
    q2_ind=np.array(S, dtype = int)[q2] 
    q1=np.argsort(-m_grad.ravel())[0:int(q/2)]
    q1_ind=np.array(R, dtype = int)[q1]
    # q1 è l'indice i in R: i_index
    # q2  è l'indice j in S: j_index
    return q1_ind, q2_ind   
    
solvers.options['show_progress'] = False

def decomposition_method(X, Y, epsilon, gamma, C, q):
    # X_train = X
    kernel = polynomial_kernel(X, X, gamma)
     
    index_array = np.arange(X.shape[0])
 
    alfa = np.zeros((X.shape[0],1))
    gradient = -np.ones((len(alfa), 1))
    
    y_train = Y.reshape(-1,1)
    m = m_value(alfa, y_train, tolerance, C,gradient, kernel, q)
    M  = M_value(alfa, y_train, tolerance, C,gradient, kernel, q)
    
    counter = 0
    while (m-M) >= epsilon: 

        i_index, j_index = Working_set(alfa, y_train, tolerance, C, gradient, q)
        indexes = np.concatenate((i_index, j_index))
        Q_indexes = matrix(np.outer(Y[indexes], Y[indexes]) * kernel[np.ix_(indexes, indexes)])
        
        direction = np.array([y_train[i_index], -y_train[j_index]]).reshape(2,1)
        
        numerator = np.dot(gradient[indexes].reshape(1,2), direction)
        denominator  = np.dot(np.dot(direction.T, Q_indexes), direction) 
        t_star = float(-numerator/denominator) 

        t_star = min(t_star, 10000)
        
        #condizioni per t max ammissibile:
         
        if direction[0] > 1e-5:
            t_max_1 = (C - alfa[i_index])/direction[0]  
        else:
            t_max_1 = (alfa[i_index])/np.abs(direction[0]) 
            
        if direction[1] > 1e-5:
            t_max_2 = (C - alfa[j_index])/direction[1] 
        else:
            t_max_2 = (alfa[j_index])/np.abs(direction[1]) 
       
        t_max = min(t_max_1, t_max_2)
        
        t_star = min(t_star, t_max)
        counter += 1 
        
        Q_columns = (kernel[:,indexes] * Y.reshape(-1,1)) * Y[indexes]
        alfa_new = alfa[indexes] + t_star*direction
         
        gradient = gradient + np.dot(Q_columns, alfa_new - alfa[indexes]) 
        alfa[indexes] = alfa_new
        

        m = m_value(alfa, y_train, tolerance, C,gradient, kernel,q)
        M = M_value(alfa, y_train, tolerance, C,gradient, kernel,q)  
    
    return np.array(alfa), m, M, counter  
