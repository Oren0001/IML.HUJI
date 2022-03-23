import numpy.random

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

UNIVARIATE_MU = 10
VAR = 1
MULTIVARIATE_MU = np.array([0, 0, 4, 0])
COV = np.array([[1, 0.2, 0, 0.5],
                [0.2, 2, 0, 0],
                [0, 0, 1, 0],
                [0.5, 0, 0, 1]])
SAMPLES_NUM = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(UNIVARIATE_MU, VAR, SAMPLES_NUM)
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
        y.append(abs(UnivariateGaussian().fit(samples[:sample_size]).mu_ - UNIVARIATE_MU))
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
    # Question 4 - Draw samples and print fitted model
    samples = numpy.random.multivariate_normal(MULTIVARIATE_MU, COV, SAMPLES_NUM)
    n = samples.shape[0]
    fig4 = go.Figure([
        go.Scatter(x=['x1'] * n, y=samples[:, 0], mode="markers", marker=dict(color="Red")),
        go.Scatter(x=['x2'] * n, y=samples[:, 1], mode="markers",
                   marker=dict(color="Green")),
        go.Scatter(x=['x3'] * n, y=samples[:, 2], mode="markers",
                   marker=dict(color="Blue")),
        go.Scatter(x=['x4'] * n, y=samples[:, 3], mode="markers",
                   marker=dict(color="pink"))],
        layout=go.Layout(title=r"$\text{1000 Samples: x1,...,x1000 iid~ N}(\mu, \Sigma)$",
                         xaxis_title="Random Variables", yaxis_title="Sample Values",
                         showlegend=False, height=700, width=400))
    fig4.show()
    m = MultivariateGaussian().fit(samples)
    print(m.mu_)
    print(m.cov_)

    # Question 5 - Likelihood evaluation
    v = np.linspace(-10, 10, 200)
    log_likelihood = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            mu = np.array([v[i], 0, v[j], 0])
            log_likelihood[i, j] = MultivariateGaussian().log_likelihood(mu, COV, samples)
    fig5 = go.Figure([go.Heatmap(x=v, y=v, z=log_likelihood)],
                     layout=go.Layout(title="Log-likelihood Evaluation",
                                      xaxis_title="f3", yaxis_title="f1",
                                      showlegend=False, height=700, width=700))
    fig5.show()

    # Question 6 - Maximum likelihood
    f1_max = v[np.max(np.argmax(log_likelihood, axis=0))]
    f3_max = v[np.max(np.argmax(log_likelihood, axis=1))]
    print(f"(f1, f3) = ({f1_max:0.3f}, {f3_max:0.3f})")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
