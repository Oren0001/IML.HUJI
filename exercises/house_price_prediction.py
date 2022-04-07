from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()

    for feature in ["id", "date", "lat", "long"]:
        df = df.drop(feature, 1)

    for feature in ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built",
                    "sqft_living15", "sqft_lot15", "zipcode"]:
        df = df[df[feature] > 0]
    for feature in ["bedrooms", "bathrooms", "floors", "sqft_basement", "yr_renovated",
                    "condition", "grade"]:
        df = df[df[feature] >= 0]
    df = df[(df["waterfront"] >= 0) & (df["waterfront"] <= 1)]
    df = df[(df["view"] >= 0) & (df["view"] <= 4)]

    df["yr_built"] = (df["yr_built"] / 10).astype(int)
    df = df.rename(columns={"yr_built": "decade_built"})
    df = pd.get_dummies(df, columns=['decade_built'])
    df["zipcode"] = df["zipcode"].astype(int)
    df = pd.get_dummies(df, columns=['zipcode'])

    df.insert(0, "intercept", 1, True)
    return df.drop("price", 1), df.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    std_y = np.std(y)
    to_drop = ["intercept", "zipcode", "decade"]
    for (feature_name, feature_vec) in X.iteritems():
        if all(x == -1 for x in map(feature_name.find, to_drop)):
            rho = np.cov(feature_vec, y)[0, 1] / (np.std(feature_vec) * std_y)
            fig = go.Figure([go.Scatter(x=feature_vec, y=y, mode="markers")],
                            layout=go.Layout(title=f"Pearson Correlation between {feature_name} "
                                                   f"and response <br>equals to: {rho}",
                                             xaxis_title=f"{feature_name} Values",
                                             yaxis_title="Response Values",
                                             showlegend=False))
            fig.write_image(f"{output_path}/{feature_name}.jpeg")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon
    # of size (mean-2*std, mean+2*std)
    means = np.zeros((91,))
    stds = np.zeros((91,))
    percentage = np.arange(10, 101)
    for i, p in enumerate(percentage):
        loss = np.zeros((10,))
        for j in range(10):
            # 1
            sample = X.sample(frac=p / 100)
            y_ = y.reindex_like(sample)
            # 2
            lr = LinearRegression()
            lr._fit(sample, y_)
            # 3
            loss[j] = lr._loss(test_X, test_y)
        # 4
        means[i] = np.mean(loss)
        stds[i] = np.std(loss)

    fig = go.Figure([go.Scatter(x=percentage, y=means - 2 * stds, fill=None, mode="lines",
                                line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=percentage, y=means + 2 * stds, fill="tonexty", mode="lines",
                                line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=percentage, y=means, mode="markers+lines",
                                marker=dict(color="black", size=1), showlegend=False)],
                    layout=go.Layout(title="Mean loss as a function of increasing "
                                           "percentage of training data",
                                     xaxis_title="Percentage of Training Data",
                                     yaxis_title="Mean Loss"))
    fig.show()
