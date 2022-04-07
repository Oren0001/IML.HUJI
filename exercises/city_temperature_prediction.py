import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
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
    df.to_csv("preprocessed_temp.csv")
    return df


def question_2(data):
    data = data[data["Israel"] == 1].drop(["Jordan", "South Africa", "The Netherlands"], 1)

    data["Year"] = data["Year"].astype(str)
    px.scatter(data, x="DayOfYear", y="Temp", color="Year",
               title="Temperature Change As a Function of 'DayOfYear'").show()

    data = data.groupby(["Month"]).agg({"Temp": np.std}).reset_index()
    data.rename(columns={"Temp": "Temp std"}, inplace=True)
    px.bar(data, x="Month", y="Temp std",
           title="Standard Deviation of Daily Temperatures as a Function of 'Months'").show()


def question_3(data):
    data = data.groupby(["Israel", "Jordan", "South Africa", "The Netherlands", "Month"])
    data.apply(print)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    question_2(data)

    # Question 3 - Exploring differences between countries
    question_3(data)

    # Question 4 - Fitting model for different values of `k`

    # Question 5 - Evaluating fitted model on different countries
