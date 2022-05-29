from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from sklearn.model_selection import train_test_split
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MIN_RANGE = -1.2
MAX_RANGE = 2
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
    fig = make_subplots(rows=1, cols=2, subplot_titles=["True (noiseless) Model",
                                                        "Train and Test Samples With Noise"])
    fig.add_traces([go.Scatter(x=x, y=f_x, mode="markers", showlegend=False),
                    go.Scatter(x=train_X, y=train_y, mode="markers", name="Train"),
                    go.Scatter(x=test_X, y=test_y, mode="markers", name="Test")],
                   rows=[1, 1, 1], cols=[1, 2, 2])
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    avg_train_err = np.zeros(MAX_DEGREE)
    avg_validation_err = np.zeros(MAX_DEGREE)
    for d in range(MAX_DEGREE):
        avg_train_err[d], avg_validation_err[d] = cross_validate(PolynomialFitting(d), train_X,
                                                                 train_y, mean_square_error)
    x_ = np.arange(MAX_DEGREE)
    go.Figure(
        [go.Scatter(x=x_, y=avg_train_err, mode="markers+lines", name="Train Error"),
         go.Scatter(x=x_, y=avg_validation_err, mode="markers+lines", name="Validation Error")],
        layout=go.Layout(title="Average Training and Validation Errors As a Function Of Degrees",
                         xaxis_title="Degrees", yaxis_title="Avg Errors")).show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = int(np.argmin(avg_validation_err))
    print(f"The polynomial degree for which the lowest validation error was achieved is: {best_k}.")
    p = PolynomialFitting(best_k)
    p.fit(train_X, train_y)
    best_error = mean_square_error(test_y, p.predict(test_X))
    print(f"Test error for best value of k is: {round(best_error, 2)}.\n")


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
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0, 20, num=n_evaluations)
    avg_t_err_ridge = np.zeros(n_evaluations)
    avg_v_err_ridge = np.zeros(n_evaluations)
    for i, lam in enumerate(lambdas):
        avg_t_err_ridge[i], avg_v_err_ridge[i] = cross_validate(RidgeRegression(lam), train_X,
                                                                train_y, mean_square_error)

    avg_t_err_lasso = np.zeros(n_evaluations)
    avg_v_err_lasso = np.zeros(n_evaluations)
    for i, lam in enumerate(lambdas):
        avg_t_err_lasso[i], avg_v_err_lasso[i] = cross_validate(Lasso(lam), train_X,
                                                                train_y, mean_square_error)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Ridge", "Lasso"])
    fig.add_traces(
        [go.Scatter(x=lambdas, y=avg_t_err_ridge, mode="lines", name="Ridge Train Error"),
         go.Scatter(x=lambdas, y=avg_v_err_ridge, mode="lines", name="Ridge Validation Error"),
         go.Scatter(x=lambdas, y=avg_t_err_lasso, mode="lines", name="Lasso Train Error"),
         go.Scatter(x=lambdas, y=avg_v_err_lasso, mode="lines", name="Lasso Validation Error")],
        rows=[1, 1, 1, 1], cols=[1, 1, 2, 2])
    fig.update_xaxes(title_text="Regularization Parameter Values")
    fig.update_yaxes(title_text="Errors")
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_ridge = lambdas[np.argmin(avg_v_err_ridge)]
    best_lam_lasso = lambdas[np.argmin(avg_v_err_lasso)]
    print("The regularization parameter which achieved the best validation error is - \n"
          f"For Ridge: {best_lam_ridge}.\nFor Lasso: {best_lam_lasso}.\n")

    print("Given these values, test errors are - ")
    ridge = RidgeRegression(best_lam_ridge)
    ridge.fit(train_X, train_y)
    test_err_ridge = mean_square_error(test_y, ridge.predict(test_X))
    print(f"For Ridge: {test_err_ridge}.")
    lasso = Lasso(best_lam_lasso)
    lasso.fit(train_X, train_y)
    test_err_lasso = mean_square_error(test_y, lasso.predict(test_X))
    print(f"For Lasso: {test_err_lasso}.")
    lr = LinearRegression()
    lr.fit(train_X, train_y)
    test_err_lr = mean_square_error(test_y, lr.predict(test_X))
    print(f"For Least Squares: {test_err_lr}.")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(100, 5)
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
