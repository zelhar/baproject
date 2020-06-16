import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.stats import poisson
from tqdm import tqdm

# plt.ioff()
plt.ion()

########################## Assigment 4

## Power Iteration
def powerIterate(A, alpha=0.85, epsilon=1e-7, maxiter=10 ** 7, directmethod=False):
    """The inputs: A: a non 2d array representing
    an adjacency matrix of a graph.
    alpha: the restart parameter, must be between 0 and 1.
    epsilon: convergence accuracy requirement
    maxiter: additional stop condition for the loop,
    whichever reached first stops the loop.
    If directmethod=True the calculation is 
    done using the direct method rather than iteration"""
    n = len(A)
    # normaliz A column-wise:
    d = A.sum(axis=0).reshape((1, n))
    d[d == 0] = 1  # avoid division by 0
    A = A / d
    # comnine A with the restart matrix
    W = np.ones((n, n)) / n
    W = (1 - alpha) * W + alpha * A  # the transition matrix
    # create a random state vector:
    # x = np.random.random((n, 1))
    # x = x / x.sum()
    # create a uniform state vector:
    x = np.ones((n, 1)) / n
    if directmethod:
        I = np.identity(n)
        B = I - alpha * A
        B = np.linalg.inv(B)
        p = (1 - alpha) * np.dot(B, x)
        return p, W
    t = np.zeros(maxiter)
    for i in tqdm(range(maxiter)):
        y = np.dot(W, x)
        t[i] = np.linalg.norm((x.flatten() - y.flatten()), ord=1)
        # above, flatten so the norm will be for vectors, I think
        if t[i] < epsilon:
            return y, W, t[: i + 1]
        else:
            x = y
    return x, W, t


def biasedPropagate(A, p0, alpha=0.85, beta=0.999, epsilon=1e-7, maxiter=10 ** 7):
    """The inputs: A: a non 2d array representing
    an adjacency matrix of a graph.
    p0: The biased restart distribution. 
    alpha: the restart parameter, must be between 0 and 1.
        1-alpha is the restart probability
    epsilon: convergence accuracy requirement
    maxiter: additional stop condition for the loop,
    whichever reached first stops the loop.
    beta: pseudocount parameter to ensures regularity of the matrix.
        1-beta is the restart probability
    So the transitional matrix incorporates both the biased and the uniform
    distribution."""
    n = len(A)
    # normaliz A column-wise:
    d = A.sum(axis=0).reshape((1, n))
    d[d == 0] = 1  # avoid division by 0
    W = A / d
    # comnine A with the restart matrix
    p0 = p0 / np.sum(p0)
    B = p0.reshape((n, 1)) + np.zeros_like(A)  # make p0 a column vector
    print(B)
    W = (1 - alpha) * B + alpha * W  # the transition matrix with bias
    E = np.ones((n, n)) / n  # the unbiased restart matrix
    print(E)
    W = (1 - beta) * E + beta * W  # the transition matrix with the unbiased
    print(W)
    # create a random state vector:
    # x = np.random.random((n, 1))
    # x = x / x.sum()
    # create a uniform state vector:
    x = np.ones((n, 1)) / n
    t = np.zeros(maxiter)
    for i in tqdm(range(maxiter)):
        y = np.dot(W, x)
        t[i] = np.linalg.norm((x.flatten() - y.flatten()), ord=1)
        # above, flatten so the norm will be for vectors, I think
        if t[i] < epsilon:
            return y, W, t[: i + 1]
        else:
            x = y
    return x, W, t


# test
A = np.random.randint(low=0, high=2, size=(7, 7))
p0 = np.array([1 / 2, 1 / 2, 0, 0, 0, 0, 0])

p, W, t = biasedPropagate(A, p0)

p, W, t = powerIterate(A)

q, T = powerIterate(A, directmethod=True)


# generate random graphs
ws = nx.watts_strogatz_graph(n=50, k=7, p=0.3)
ba = nx.barabasi_albert_graph(n=50, m=3)
er = nx.erdos_renyi_graph(n=100, p=0.08, seed=42)
er2 = nx.erdos_renyi_graph(n=100, p=0.01, seed=42)  # bad result with the direct
er3 = nx.erdos_renyi_graph(n=100, p=0.4, seed=42)

adj_er = np.array(nx.adj_matrix(er).todense())

adj_er2 = np.array(nx.adj_matrix(er2).todense())
# get the page rank

p, W, t = powerIterate(adj_er)
q, T = powerIterate(adj_er, directmethod=True)

p, W, t = powerIterate(adj_er2)
q, T = powerIterate(adj_er2, directmethod=True)

np.linalg.norm(np.dot(W, p) - p, ord=1)
np.linalg.norm(np.dot(T, q) - q, ord=1)
np.linalg.norm(p - q, ord=1)

plt.plot(t)


adj_ws = np.array(nx.adj_matrix(ws).todense())
adj_ba = np.array(nx.adj_matrix(ba).todense())

p, W, t = powerIterate(adj_ws)
q, T = powerIterate(adj_ws, directmethod=True)
np.linalg.norm(np.dot(W, p) - p, ord=1)
np.linalg.norm(np.dot(T, q) - q, ord=1)
np.linalg.norm(p - q, ord=1)

p, W, t = powerIterate(adj_ba)
q, T = powerIterate(adj_ba, directmethod=True)
np.linalg.norm(np.dot(W, p) - p, ord=1)
np.linalg.norm(np.dot(T, q) - q, ord=1)
np.linalg.norm(p - q, ord=1)


# add the page rank info to the edges
n = len(er.nodes)
for i in range(n):
    er.nodes[i]["prank"] = p.flatten()[i]

nx.write_gml(er, "er_008.gml")
