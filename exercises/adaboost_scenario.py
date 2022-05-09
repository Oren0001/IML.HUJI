import numpy as np
from typing import Tuple
from IMLearn.learners.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def my_decision_surface(predict, xrange, yrange, T, density=120, dotted=False,
                        colorscale=custom, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()], T)

    if dotted:
        return go.Scatter(
            x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers",
            marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False),
            hoverinfo="skip", showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale,
                      reversescale=False, opacity=.7, connectgaps=True, hoverinfo="skip",
                      showlegend=False, showscale=showscale)


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    train_X, train_y = generate_data(train_size, noise)
    test_X, test_y = generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    train_errors, test_errors = np.zeros(n_learners), np.zeros(n_learners)
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    for t in range(n_learners):
        train_errors[t] = adaboost.partial_loss(train_X, train_y, t + 1)
        test_errors[t] = adaboost.partial_loss(test_X, test_y, t + 1)
    x = np.arange(1, n_learners + 1)
    go.Figure([go.Scatter(x=x, y=train_errors, mode="lines", name="Train errors"),
               go.Scatter(x=x, y=test_errors, mode="lines", name="Test errors")],
              layout=go.Layout(title="Train-Test Errors As a Function of Fitted Learners Number",
                               xaxis_title="Number of Fitted Learners",
                               yaxis_title="Errors")).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[f"Number of Iterations = {i}" for i in T])
    for i, t in enumerate(T):
        fig2.add_traces(
            [my_decision_surface(adaboost.partial_predict, lims[0], lims[1], t, showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=test_y, colorscale=[custom[0], custom[-1]]))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig2.update_layout(title=f"Decision Boundaries Using An Ensemble Up To Iterations: {T}",
                       margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig2.show()

    # Question 3: Decision surface of best performing ensemble

    # Question 4: Decision surface with weighted samples


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
