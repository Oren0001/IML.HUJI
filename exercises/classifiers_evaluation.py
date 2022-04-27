import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
from sklearn.model_selection import train_test_split


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers.
    File is assumed to be an ndarray of shape (n_samples, 3) where the first 2 columns
    represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
    linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "../datasets/linearly_separable.npy"),
                 ("Linearly Inseparable", "../datasets/linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        p = Perceptron(callback=lambda fit, X_, y_: losses.append(fit._loss(X_, y_)))
        p.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        go.Figure([go.Scatter(x=np.arange(len(losses)), y=losses, mode="markers")],
                  layout=go.Layout(title="Loss as a function of fitting iteration over"
                                         f" {n} dataset",
                                   xaxis_title="Fitting Iterations",
                                   yaxis_title="Loss Values", showlegend=False)).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black",
                      showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for n, f in [("gaussian1", "../datasets/gaussian1.npy"),
                 ("gaussian2", "../datasets/gaussian2.npy")]:
        # Load dataset
        X, y = load_dataset(f)
        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75)

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes().fit(train_X, train_y)
        y_gnb_pred = gnb.predict(test_X)
        lda = LDA().fit(train_X, train_y)
        y_lda_pred = lda.predict(test_X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on
        # the left and LDA predictions on the right. Plot title should specify dataset used
        # and subplot titles should specify algorithm and accuracy.
        # Create subplots
        from IMLearn.metrics import accuracy
        gnb_accuracy = accuracy(test_y, y_gnb_pred)
        lda_accuracy = accuracy(test_y, y_lda_pred)
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            f"Gaussian Naive Bayes Predictions with Accuracy = {gnb_accuracy}",
            f"LDA Predictions with Accuracy = {lda_accuracy}"])
        fig.update_layout(title=f"{n} Dataset", margin=dict(t=100))

        # Add traces for data-points setting symbols and colors
        symbols = np.array(["circle", "square", "square"])
        for i, y_pred in enumerate([y_gnb_pred, y_lda_pred], 1):
            fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                     marker=dict(color=y_pred, symbol=symbols[y_pred],
                                                 colorscale=["red", "green", "pink"]),
                                     showlegend=False), row=1, col=i)

        # Add `X` dots specifying fitted Gaussians' means
        for i, classifier in enumerate([gnb, lda], 1):
            fig.add_trace(go.Scatter(x=classifier.mu_[:, 0], y=classifier.mu_[:, 1], mode="markers",
                                     marker=dict(color="black", symbol='x'),
                                     showlegend=False), row=1, col=i)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i, classifier in enumerate([gnb, lda], 1):
            for j in range(classifier.classes_.size):
                fig.add_trace(get_ellipse(classifier.mu_[j], classifier.cov_), row=1, col=i)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
