#-------------------------------------------------------------------------
# AUTHOR: Tony Gonzalez
# FILENAME: perceptron.py
# SPECIFICATION: Trains a Single Layer Perceptron
# FOR: CS 4210- Assignment #3
# TIME SPENT: 45 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

algorithms = ["Perceptron", "MLP"]
p_max_accuracy = -float('inf')
mlp_max_accuracy = -float('inf')

for learning_rate in n: # iterates over n

    for shuffle_option in r: # iterates over r

        for algorithm in algorithms: # iterates over the algorithms

            if algorithm == "Perceptron":
                clf = Perceptron(eta0=learning_rate, shuffle=shuffle_option, max_iter=1000)
            else:
                clf = MLPClassifier(activation="logistic",
                                    learning_rate_init=learning_rate,
                                    hidden_layer_sizes=25,
                                    shuffle=shuffle_option,
                                    max_iter=1000)

            # Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            n_correct = 0

            # Make predictions
            for (X_testSample, Y_testSample) in zip(X_test, y_test):
                pred = clf.predict([X_testSample])

                if pred == Y_testSample:
                    n_correct += 1

            # Calculate accuracy and update if needed
            curr_accuracy = n_correct / len(X_test)

            if curr_accuracy > p_max_accuracy and algorithm == "Perceptron":
                p_max_accuracy = curr_accuracy
                print(f"Highest Perception accuracy so far: {curr_accuracy:.4f}, Parameters: learning rate={learning_rate}, shuffle={shuffle_option}")
            elif curr_accuracy > mlp_max_accuracy and algorithm == "MLP":
                mlp_max_accuracy = curr_accuracy
                print(f"Highest MLP accuracy so far: {curr_accuracy:.4f}, Parameters: learning rate={learning_rate}, shuffle={shuffle_option}")
