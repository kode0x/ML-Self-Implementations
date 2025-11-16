import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MultipleLinearRegression:
    def __init__(self):
        self.intercept = None
        self.coefficient = None    

    def __str__(self) -> str:
        return "Linear Regression Model Fitted Using The Ordinary Least Squares (OLS) Method"

    def parameters(self):
        print(f"Intercept: {self.intercept}") 
        print(f"Coefficients: {self.coefficient}")

    def fit(self, X_train, Y_train):
        X_train = np.insert(X_train, 0, 1, axis=1)
        beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train
        self.intercept = beta[0]
        self.coefficient = beta[1:]

    def predict(self, X_test):
        X_test = np.array(X_test)
        X_test = np.insert(X_test, 0, 1, axis=1)
        Y_pred = np.dot(X_test, np.append(self.intercept, self.coefficient))
        return Y_pred

    def plot(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        plt.figure(figsize=(10, 6))
        plt.scatter(Y_test, Y_pred, color='blue', alpha=0.6)
        plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        plt.show()
