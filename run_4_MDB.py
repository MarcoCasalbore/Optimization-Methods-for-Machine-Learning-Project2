# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 20:15:49 2023

@author: Bianca
"""

from Functions_4_MDB import *

file_train = 'FashionMNIST\\fashion-mnist_train.csv' 
file_test = 'FashionMNIST\\fashion-mnist_test.csv' 
X_train, Y_train, X_test, Y_test = lettura_dati(file_train, file_test) 
 
X_train_scaled, X_test_scaled = normalization(X_train, X_test)  
initial_time = time.time()
training_prediction_2, alfa_star2, Y_2, iterations_2  = optimization(X_train_scaled, Y_train, gamma, tolerance, C, 2)
training_prediction_3, alfa_star3, Y_3, iterations_3  = optimization(X_train_scaled, Y_train, gamma, tolerance, C, 3)
training_prediction_6, alfa_star6, Y_6, iterations_6  = optimization(X_train_scaled, Y_train, gamma, tolerance, C, 6)
final_time = time.time() 
training_accuracy, test_accuracy, iterations, conf_matrix = multi_class(X_train_scaled, X_test_scaled, Y_train ,Y_test, tolerance, gamma, C, labels = [2,3,6])

print("C value:", C)
print("Gamma value:", gamma)
print("Time used for optimization:", final_time - initial_time) 
print("Training accuracy", (training_accuracy * 100), "%")  
print("Test accuracy", round(test_accuracy,4) * 100, '%')  
print("Number of iterations:",  iterations) 

kernel = polynomial_kernel(X_train_scaled,X_train_scaled,gamma)
m_2 = m_value(alfa_star2, Y_2, tolerance, C, kernel)
M_2 = M_value(alfa_star2, Y_2, tolerance, C, kernel) 
print("KKT violations of 2 against all:", m_2 - M_2)

m_3 = m_value(alfa_star3, Y_3, tolerance, C, kernel)  
M_3 = M_value(alfa_star3, Y_3, tolerance, C, kernel) 
print("KKT violations of 3 against all:", m_3 - M_3)

m_6 = m_value(alfa_star6, Y_6, tolerance, C, kernel)
M_6 = M_value(alfa_star6, Y_6, tolerance, C, kernel) 
print("KKT violations of 6 against all:", m_6 - M_6) 

final_of_2 = objective_function(alfa_star2, gamma, X_train_scaled, Y_2)
final_of_3 = objective_function(alfa_star3, gamma, X_train_scaled, Y_3) 
final_of_6 = objective_function(alfa_star6, gamma, X_train_scaled, Y_6)
print("Final value of the objective function 2 against all:", float(final_of_2))
print("Final value of the objective function 3 against all:", float(final_of_3))
print("Final value of the objective function 6 against all:", float(final_of_6))
print("Confusion matrix:", conf_matrix) 
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=[2,3,6]) 
disp.plot(cmap = 'ocean')  
plt.show()