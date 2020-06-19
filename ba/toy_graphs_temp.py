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
    done using the direct method rather than iteration.
    """
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
            return y.flatten(), W, t[: i + 1]
        else:
            x = y
    return x.flatten(), W, t


def biasedPropagate(A, bias, alpha=0.85, beta=1, epsilon=1e-7, maxiter=10 ** 7):
    """The inputs: A: a non 2d array representing
    an adjacency matrix of a graph.
    bias: The biased restart distribution. 
    alpha: the restart parameter, must be between 0 and 1.
        1-alpha is the restart probability
    epsilon: convergence accuracy requirement
    maxiter: additional stop condition for the loop,
    whichever reached first stops the loop.
    beta: pseudocount parameter to ensures regularity of the matrix.
        1-beta is the restart probability. If the graph is known to be
        connected, than it regular (and only if, see notes doc).
    So the transitional matrix incorporates both the biased and the uniform
    distribution.
    """
    n = len(A)
    # normaliz A column-wise:
    d = A.sum(axis=0).reshape((1, n))
    d[d == 0] = 1  # avoid division by 0
    W = A / d
    # comnine A with the restart matrix
    bias = bias / np.sum(bias)
    B = bias.reshape((n, 1)) + np.zeros_like(A)  # make bias a column vector
    W = (1 - alpha) * B + alpha * W  # the transition matrix with bias
    if beta < 1:
        E = np.ones((n, n)) / n  # the unbiased restart matrix
        W = (1 - beta) * E + beta * W  # the transition matrix with the unbiased
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
            return y.flatten(), W, t[: i + 1]
        else:
            x = y
    return x.flatten(), W, t


# test
A = np.random.randint(low=0, high=2, size=(7, 7))
p0 = np.array([1 / 2, 1 / 2, 0, 0, 0, 0, 0])
p, W, t = biasedPropagate(A, p0)
p, W, t = powerIterate(A)
q, T = powerIterate(A, directmethod=True)

er = nx.erdos_renyi_graph(n=25, p=0.08, seed=42)
nx.draw(er, with_labels=True)
plt.show()

ba = nx.barabasi_albert_graph(n=25, m=3)
nx.draw(ba, with_labels=True)
plt.show()

ws = nx.watts_strogatz_graph(n=25, k=4, p=0.3)
nx.draw(ws, with_labels=True)
plt.show()

# I think it makes sense to experiement with small barbasi-albert type
# graphs. As far as I can tell they are more similar to real PPIN than
# either Erdos-Renyi (not connected, no hub preference, not small world)
# or Watts-Strogatz (I think it has no hub preference but maybe it is
# actually better)

n = 15
m = 3
seed = 42

ba = nx.barabasi_albert_graph(n=n, m=m, seed=seed)

nx.draw(ba, with_labels=True)

plt.show()

bias = np.zeros(n)
bias[12] = 1
bias[6] = 1


bias = np.zeros(n)
bias[25] = 1
bias[28] = 1
bias[30] = 1

nx.draw(ba, with_labels=True, node_color=bias)

adj_ba = np.array(nx.adj_matrix(ba).todense())

p, W, t = biasedPropagate(adj_ba, bias, alpha=0.85)

p = p.flatten()
x = np.arange(len(p))

plt.bar(x, p)
plt.show()

(p > 0.07).sum()

colors = [0 if i > 0.05 else 1 for i in p]

colors = [0 if i > 1.32 / len(p) else 1 for i in p]

nx.draw(ba, with_labels=True, node_color=colors)
plt.show()


def findKins(G, points=[0], alpha=0.85):
    """This function takes a graph G (assumed to be 
    connected and bidirectional) and a starting point
    and iteratively tries to find nodes that together increase their
    average pagerank"""
    n = len(G.nodes())
    m = len(points)
    print(n, m)
    if m >= n:
        return points
    # The initial biased is concentrated on the starting point(s)
    bias = np.zeros(n)
    bias[points] = 1
    edges = np.array(nx.adj_matrix(G).todense())
    p, W, t = biasedPropagate(edges, bias, alpha=alpha)
    avg_rank = p[points].mean()
    print(points)
    print(p[points])
    print(avg_rank)
    for i in range(m, n):
        p, W, t = biasedPropagate(edges, bias, alpha=alpha)
        new_points = np.argsort(-p)[0 : i + 1]
        temp_avg = p[new_points].mean()
        if temp_avg > avg_rank:
            avg_rank = temp_avg
            bias[:] = 0
            bias[new_points] = 1
            points = new_points
            # probably the new points are the old points + one addtional
            # point ...
            print(points)
            print(p[points])
            print(avg_rank)
        else:
            return points


points = findKins(G=ba, points=[27, 21], alpha=0.89)

points = findKins(G=ba, points=[6, 12], alpha=0.89)

points = findKins(G=ba, points=[1, 11, 13, 5], alpha=0.85)
colors = np.zeros_like(ba.nodes)
colors[points] = 1
nx.draw(ba, with_labels=True, node_color=colors)
plt.show()

points = [13, 11]
p, W, t = biasedPropagate(adj_ba, bias, alpha=0.85)
colors = [0 if x > 1 / len(ba.nodes()) else 1 for x in p]
nx.draw(ba, with_labels=True, node_color=colors)
plt.show()


def extendLabel(G, label=[0], alpha=0.85):
    """Given a graph, a list of nodes assumed to have the same
    labeled functionality, this function generates the stationary
    distribution of the graph with bias on the labeled nodes and alpha
    parameter. It then searches to other nodes that have pagerank higher
    than the minimal (or avg?) pagerank of the labeled nodes and returns
    the extended list.
    """
    n = len(G.nodes())
    m = len(label)
    bias = np.zeros(n)
    bias[label] = 1
    edges = np.array(nx.adj_matrix(G).todense())
    p, W, t = biasedPropagate(edges, bias, alpha=alpha)
    avg_rank = p[points].mean()
    min_rank = p[points].min()
    ext_label = [i for i in range(n) if p[i] >= min_rank]
    ext_label2 = [i for i in range(n) if p[i] >= min_rank or i in label]
    return ext_label2


points

extlabel = extendLabel(ba, label=[1, 2, 3, 4], alpha=0.85)

extlabel = extendLabel(ba, label=[13, 12], alpha=0.80)
extlabel

colors = np.zeros_like(ba.nodes)
colors[extlabel] = 1
nx.draw(ba, with_labels=True, node_color=colors)
plt.show()


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


### manually generated graph
l=[]
petGraph = nx.Graph()
petGraph.add_nodes_from(range(7))
petGraph.add_edges_from([(i, 3) for i in range(3)])
petGraph.add_edges_from([(i, 4) for i in range(5, 7)])
petGraph.add_edge(3, 4)

nx.draw(petGraph, with_labels=True)

adj_m = np.array(nx.adj_matrix(petGraph).todense())
p,W,t = powerIterate(adj_m, alpha=0.85)

l.append(p)

x = np.arange(len(p))
plt.bar(x, p)

petGraph.add_edge(0,4)
nx.draw(petGraph, with_labels=True)

adj_m = np.array(nx.adj_matrix(petGraph).todense())
p,W,t = powerIterate(adj_m, alpha=0.85)

l.append(p)

p

x = np.arange(len(p))
plt.bar(x, p)

p

petGraph.add_node(7)
petGraph.add_edge(7,0)
nx.draw(petGraph, with_labels=True)


adj_m = np.array(nx.adj_matrix(petGraph).todense())
p,W,t = powerIterate(adj_m, alpha=0.85)
l.append(p)
p

x = np.arange(len(p))
plt.bar(x, p)


petGraph = nx.Graph()
petGraph.add_nodes_from(range(5))
petGraph.add_edges_from([(i, 0) for i in range(1,5)])
nx.draw(petGraph, with_labels=True)


adj_m = np.array(nx.adj_matrix(petGraph).todense())
p,W,t = powerIterate(adj_m, alpha=0.85)
l.append(p)
p
x = np.arange(len(p))
plt.bar(x, p)

petGraph.add_node(5)
petGraph.add_edge(5,0)
nx.draw(petGraph, with_labels=True)

adj_m = np.array(nx.adj_matrix(petGraph).todense())
p,W,t = powerIterate(adj_m, alpha=0.85)
l.append(p)
p
x = np.arange(len(p))
plt.bar(x, p)











