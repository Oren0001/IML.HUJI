import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    df = df.drop("City", 1)
    df = pd.get_dummies(df)
    df.rename(columns={"Country_Israel": "Israel", "Country_Jordan": "Jordan",
                       "Country_South Africa": "South Africa",
                       "Country_The Netherlands": "The Netherlands"}, inplace=True)

    df = df[df["Year"] <= 2022]
    df = df[(df["Month"] >= 1) & (df["Month"] <= 12)]
    df = df[(df["Day"] >= 1) & (df["Month"] <= 31)]
    df = df[(df["Temp"] >= -65) & (df["Temp"] <= 60)]

    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df


def question_2(data):
    data = data[data["Israel"] == 1].drop(["Israel", "Jordan", "South Africa",
                                           "The Netherlands"], 1).reset_index(drop=True)
    data_copy = data

    data["Year"] = data["Year"].astype(str)
    px.scatter(data, x="DayOfYear", y="Temp", color="Year",
               title="Temperature Change As a Function of 'DayOfYear'").show()

    data = data.groupby(["Month"]).agg({"Temp": np.std}).reset_index()
    data.rename(columns={"Temp": "Temp std"}, inplace=True)
    px.bar(data, x="Month", y="Temp std",
           title="Standard Deviation of Daily Temperatures as a Function of 'Month'").show()

    return data_copy.DayOfYear, data_copy.Temp


def question_3(data):
    data = data.groupby(["Israel", "Jordan", "South Africa", "The Netherlands", "Month"]).agg(
        {"Temp": (np.std, np.average)}).reset_index()
    data.columns = ["Israel", "Jordan", "South Africa", "The Netherlands",
                    "Month", "Temp STD", "Temp Average"]

    netherlands = data.loc[:11, ["Month", "Temp STD", "Temp Average"]].rename(
        columns={"Temp STD": "Netherlands Temp STD", "Temp Average": "Netherlands Temp Average"})
    africa = data.loc[11:23, ["Temp STD", "Temp Average"]].rename(columns={
        "Temp STD": "Africa Temp STD", "Temp Average": "Africa Temp Average"}).reset_index(
        drop=True)
    jordan = data.loc[23:35, ["Temp STD", "Temp Average"]].rename(columns={
        "Temp STD": "Jordan Temp STD", "Temp Average": "Jordan Temp Average"}).reset_index(
        drop=True)
    israel = data.loc[35:, ["Temp STD", "Temp Average"]].rename(columns={
        "Temp STD": "Israel Temp STD", "Temp Average": "Israel Temp Average"}).reset_index(
        drop=True)
    data = pd.concat([netherlands, africa, jordan, israel], axis=1, join='inner')

    line1 = px.line(data, x="Month", y="Netherlands Temp Average", error_y="Netherlands Temp STD")
    line1.update_traces(line=dict(color="black"), name="Netherlands Temp' Average", showlegend=True)
    line2 = px.line(data, x="Month", y="Africa Temp Average", error_y="Africa Temp STD")
    line2.update_traces(line=dict(color="green"), name="Africa Temp' Average", showlegend=True)
    line3 = px.line(data, x="Month", y="Jordan Temp Average", error_y="Jordan Temp STD")
    line3.update_traces(line=dict(color="red"), name="Jordan Temp' Average", showlegend=True)
    line4 = px.line(data, x="Month", y="Israel Temp Average", error_y="Israel Temp STD")
    line4.update_traces(line=dict(color="blue"), name="Israel Temp' Average", showlegend=True)
    fig = go.Figure(data=line1.data + line2.data + line3.data + line4.data,
                    layout=go.Layout(title="Average Temperature As a function of 'Month' "
                                           "<br>With Error Bars of Standard Deviation",
                                     xaxis_title="Months", yaxis_title="Average Temperature"))
    fig.show()


def question_4(X, y):
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    losses = np.zeros((10,))
    degrees = np.arange(1, 11)
    for k in degrees:
        lr = PolynomialFitting(k).fit(train_X, train_y)
        losses[k - 1] = round(lr._loss(test_X, test_y), 2)
    print(losses)
    data = pd.DataFrame(data={"Degrees": degrees, "MSE": losses})
    px.bar(data, x="Degrees", y="MSE", title="MSE As a function of Polynomial Degrees").show()


def question_5(data, X, y):
    lr = PolynomialFitting(6).fit(X, y)
    countries = ["Jordan", "South Africa", "The Netherlands"]
    losses = np.zeros((3,))
    for i, country in enumerate(countries):
        df = data[data[country] == 1].drop(countries, 1).reset_index(drop=True)
        test_X = df.DayOfYear
        test_y = df.Temp
        losses[i] = round(lr._loss(test_X, test_y), 2)

    res = pd.DataFrame(data={"Countries": countries, "MSE": losses})
    px.bar(res, x="Countries", y="MSE", title="MSE As a Function of Countries Given "
                                              "Polynomial Degree of 6").show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    X, y = question_2(data)

    # Question 3 - Exploring differences between countries
    question_3(data)

    # Question 4 - Fitting model for different values of `k`
    question_4(X, y)

    # Question 5 - Evaluating fitted model on different countries
    question_5(data, X, y)