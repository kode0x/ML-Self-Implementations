import matplotlib.pyplot
import numpy
import pandas
import seaborn


class SimpleLinearRegression:
    def __init__(self):
        self.intercept = None  # b0
        self.slope = None  # b1

    def __str__(self) -> str:
        return "Linear Regression Model Fitted Using The Ordinary Least Squares (OLS) Method"

    def parameters(self):
        print(f"Intercept (b0): {self.intercept}")
        print(f"Slope (b1): {self.slope}")

    def fit(self, X: numpy.ndarray, y: numpy.ndarray):
        X = numpy.array(X)
        y = numpy.array(y)

        x_mean = numpy.mean(X)
        y_mean = numpy.mean(y)

        numerator = numpy.sum((X - x_mean) * (y - y_mean))
        denominator = numpy.sum((X - x_mean) ** 2)

        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean

    def predict(self, X):
        X = numpy.array(X)
        return self.intercept + self.slope * X

    def plot(self, X, Y):
        seaborn.regplot(x=X, y=Y, ci=None, color="red", line_kws={"color": "blue"})
        matplotlib.pyplot.show()
