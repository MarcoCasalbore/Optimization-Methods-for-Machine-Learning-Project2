# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:57:21 2023

@author: Bianca
"""

from Functions_1_MDB import * 

file_train = 'FashionMNIST\\fashion-mnist_train.csv'
file_test = 'FashionMNIST\\fashion-mnist_test.csv'  
X_train, Y_train, X_test, Y_test = lettura_dati(file_train, file_test)
 
X_train_scaled, X_test_scaled = normalization(X_train, X_test) 

initial_time = time.time()
sol, alfa_star = optimization(X_train_scaled, Y_train, gamma, tolerance, C)
final_time = time.time()  

training_prediction = decision_fun(X_train_scaled, X_train_scaled, Y_train, gamma, alfa_star, tolerance, C) 
test_prediction = decision_fun(X_train_scaled, X_test_scaled, Y_train, gamma, alfa_star, tolerance, C) 

training_accuracy = np.sum(training_prediction.ravel() == Y_train.ravel())/Y_train.shape[0]
test_accuracy = np.sum(test_prediction.ravel() == Y_test.ravel())/Y_test.shape[0] 

K = polynomial_kernel(X_train_scaled,X_train_scaled,gamma)
m = m_value(alfa_star, Y_train, tolerance, C, K) 
M = M_value(alfa_star, Y_train, tolerance, C, K) 
 
print("C value:", C)
print("Gamma value:", gamma)
print("Training accuracy", (training_accuracy * 100), "%")  
print("Test accuracy", (test_accuracy) * 100, '%') 
print("Initial value of the objective function:", 0)
print("Final value of the objective function:", sol['primal objective'])
print("Number of iterations:", sol['iterations'])
print("Time used for optimization:", final_time - initial_time) 
conf_matrix = confusion_matrix(Y_test.ravel(),test_prediction.ravel()) 
print("KKT violations:", m - M) 
print("Confusion matrix", conf_matrix) 

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[2,3])    
disp.plot(cmap='Blues')   
plt.show()  
 
