from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
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
    return avg_train_err, avg_validation_err
