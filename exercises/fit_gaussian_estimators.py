from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

MU = 10
VAR = 1
SAMPLES_NUM = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(MU, VAR, SAMPLES_NUM)
    x = np.arange(SAMPLES_NUM)
    fig1 = go.Figure([go.Scatter(x=x, y=samples, mode="markers")],
                    layout=go.Layout(title="1000 samples: x1,...,x1000 iid~ N(10,1)",
                                     xaxis_title="Number of Samples",
                                     yaxis_title="Sample Values",
                                     showlegend=False, height=500, width=1000))
    fig1.show()
    u = UnivariateGaussian().fit(samples)
    print(f"({u.mu_}, {u.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    x, y = list(), list()
    for sample_size in range(10, 1001, 10):
        x.append(sample_size)
        y.append(abs(UnivariateGaussian().fit(samples[:sample_size]).mu_ - MU))
    fig2 = go.Figure([go.Scatter(x=x, y=y, mode="markers")],
                     layout=go.Layout(title="Absolute distance between estimated and "
                                            "the true value of expectation",
                                      xaxis_title="Sample Size",
                                      yaxis_title="Absolute Distance",
                                      showlegend=False, height=300, width=1000))
    fig2.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    ordered_samples = np.sort(samples)
    pdf = u.pdf(ordered_samples)
    fig3 = go.Figure([go.Scatter(x=ordered_samples, y=pdf, mode="markers")],
                     layout=go.Layout(title="Empirical PDFs under the fitted model",
                                      xaxis_title="Sample Values",
                                      yaxis_title="PDF",
                                      showlegend=False, height=500, width=1000))
    fig3.show()

def test_multivariate_gaussian():
    pass
    # Question 4 - Draw samples and print fitted model

    # Question 5 - Likelihood evaluation

    # Question 6 - Maximum likelihood


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
