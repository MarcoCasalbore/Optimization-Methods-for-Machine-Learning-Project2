# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:04:51 2023

@author: Bianca
"""

from Functions_2_MDB import *

file_train = 'FashionMNIST\\fashion-mnist_train.csv'
file_test = 'FashionMNIST\\fashion-mnist_test.csv' 
X_train, Y_train, X_test, Y_test = lettura_dati(file_train, file_test) 

X_train_scaled, X_test_scaled = normalization(X_train, X_test)  

initial_time = time.time() 
alfa_ottimo, m, M, iterations = decomposition_method(X_train_scaled, Y_train, epsilon, tolerance, gamma, C, q)
final_time = time.time()

training_prediction = decision_fun(X_train_scaled, X_train_scaled, Y_train, gamma, alfa_ottimo, tolerance, C) 
test_prediction = decision_fun(X_train_scaled, X_test_scaled, Y_train, gamma, alfa_ottimo, tolerance, C) 
  
training_accuracy = np.sum(training_prediction.ravel() == Y_train.ravel())/Y_train.shape[0] 
test_accuracy = np.sum(test_prediction.ravel() == Y_test.ravel())/Y_test.shape[0]  
final_of = objective_function(alfa_ottimo, gamma, X_train_scaled, Y_train)

print("C value:", C)
print("Gamma value:", gamma)
print("q value:", q)
print("Training accuracy:", (training_accuracy * 100), "%")  
print("Test accuracy:", (test_accuracy) * 100, '%')
print("Initial value of the objective function:", 0) 
print("Final value of the objective function:", float(final_of))
print("Number of iterations:",  iterations)     
print("Time used for optimization:",final_time- initial_time) 
print("KKT violations:", m - M)

conf_matrix = confusion_matrix(Y_test.ravel(),test_prediction.ravel()) 
disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels=[2,3])   
disp.plot(cmap='Oranges')    
plt.show()  
print("Confusion matrix", conf_matrix) 
 