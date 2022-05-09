from IMLearn.learners.classifiers.gaussian_naive_bayes import GaussianNaiveBayes
from IMLearn.learners.classifiers.linear_discriminant_analysis import LDA
from IMLearn.learners.classifiers.perceptron import Perceptron
from exercises.classifiers_evaluation import load_dataset
import numpy as np

S1 = np.array([[0, 0],
               [1, 0],
               [2, 1],
               [3, 1],
               [4, 1],
               [5, 1],
               [6, 2],
               [7, 2]])

S2 = np.array([[1, 1, 0],
               [1, 2, 0],
               [2, 3, 1],
               [2, 4, 1],
               [3, 3, 1],
               [3, 4, 1]])


def increase(count):
    count[0] += 1


if __name__ == '__main__':
    # Question 1
    gnb = GaussianNaiveBayes().fit(np.delete(S1, 1, axis=1), S1[:, 1])
    print(f"The estimated class probability of class 0 is: {gnb.pi_[0]}")
    print(f"The estimated expectation of class 1: {gnb.mu_[1][0]}\n")

    # Question 2
    gnb = GaussianNaiveBayes().fit(np.delete(S2, 2, axis=1), S2[:, 2])
    print(f"The estimated variance of feature 1 in class 0 is: {gnb.vars_[0, 0]}")
    print(f"The estimated variance of feature 1 in class 1 is: {gnb.vars_[1, 0]}\n")

    # Question 3
    gnb = GaussianNaiveBayes().fit(np.delete(S1, 1, axis=1), S1[:, 1])
    print(f"The estimated expectation of class 2 is: {gnb.mu_[2][0]}\n")

    # Question 4
    X, y = load_dataset("../../datasets/linearly_separable.npy")
    count = [0]
    Perceptron(callback=lambda fit, X_, y_: increase(count)).fit(X, y)
    print(f"The training loss converges to 0 after {count[0]} iterations.\n")

    # Question 5
    Perceptron(include_intercept=False).fit(X, y)
