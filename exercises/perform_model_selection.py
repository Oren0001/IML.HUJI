from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MIN_RANGE = -1.2
MAX_RANGE = 2
K = 5
MAX_DEGREE = 11


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps
    # Gaussian noise and split into training- and testing portions
    x = MIN_RANGE + np.random.rand(n_samples) * (MAX_RANGE - MIN_RANGE)
    f_x = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    eps = np.random.randn(n_samples) * noise
    dataset = f_x + eps
    train_X, test_X, train_y, test_y = train_test_split(x, dataset, train_size=2 / 3)
    fig1 = make_subplots(rows=1, cols=2, subplot_titles=["True (noiseless) Model",
                                                         "Train and Test Samples With Noise"])
    fig1.add_traces([go.Scatter(x=x, y=f_x, mode="markers", showlegend=False),
                     go.Scatter(x=train_X, y=train_y, mode="markers", name="Train"),
                     go.Scatter(x=test_X, y=test_y, mode="markers", name="Test")],
                    rows=[1, 1, 1], cols=[1, 2, 2])
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    sets = np.remainder(np.arange(train_X.size), K)
    avg_train_err = np.zeros(MAX_DEGREE)
    avg_validation_err = np.zeros(MAX_DEGREE)
    for k in range(K):
        t_x, t_y = train_X[sets != k], train_y[sets != k]
        v_x, v_y = train_X[sets == k], train_y[sets == k]
        fitted = [PolynomialFitting(d).fit(t_x, t_y) for d in range(MAX_DEGREE)]
        train_loss = [fitted[d]._loss(t_x, t_y) for d in range(MAX_DEGREE)]
        validation_loss = [fitted[d]._loss(v_x, v_y) for d in range(MAX_DEGREE)]
        avg_train_err += np.array(train_loss) / K
        avg_validation_err += np.array(validation_loss) / K
    x_ = np.arange(MAX_DEGREE)
    fig2 = go.Figure(
        [go.Scatter(x=x_, y=avg_train_err, mode="markers+lines", name="Train Error"),
         go.Scatter(x=x_, y=avg_validation_err, mode="markers+lines", name="Validation Error")],
        layout=go.Layout(title="Average Training and Validation Errors As a Function Of Degrees",
                         xaxis_title="Degrees", yaxis_title="Avg Errors"))
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(avg_validation_err)
    print(f"The polynomial degree for which the lowest validation error was achieved is: {best_k}")
    best_fit = PolynomialFitting(best_k).fit(train_X, train_y)
    best_error = best_fit._loss(test_X, test_y)
    print(f"Test error for best value of k is: {round(best_error, 2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting
    regularization parameter values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(100, 5)
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
