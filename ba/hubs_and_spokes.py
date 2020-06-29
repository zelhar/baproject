import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.stats import poisson
from tqdm import tqdm

# plt.ioff()
plt.ion()

################ Functions #######################################

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


## Power Iteration For Graph as input
def powerIterateG(G, alpha=0.85, epsilon=1e-7, maxiter=10 ** 7, directmethod=False):
    """The inputs: G: a non networkx type grap.
    alpha: the restart parameter, must be between 0 and 1.
    epsilon: convergence accuracy requirement
    maxiter: additional stop condition for the loop,
    whichever reached first stops the loop.
    If directmethod=True the calculation is 
    done using the direct method rather than iteration.
    """
    A = np.array(nx.adj_matrix(G).todense())
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
        return p.flatten(), W
    t = np.zeros(maxiter)
    for i in tqdm(range(maxiter)):
        y = np.dot(W, x)
        t[i] = np.linalg.norm((x.flatten() - y.flatten()), ord=1)
        # above, flatten so the norm will be for vectors, I think
        if t[i] < epsilon:
            return y.flatten(), W
        else:
            x = y
    return x.flatten(), W


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


def biasedPropagateG(G, bias, alpha=0.85, beta=1, epsilon=1e-7, maxiter=10 ** 7):
    """The inputs: G: a networkx type graph.
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
    A = np.array(nx.adj_matrix(G).todense())
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
            return y.flatten(), W
        else:
            x = y
    return x.flatten(), W


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


def pageRanksConcentratedBiasG(G, alpha=0.85):
    """Input: graph G, restart parameter alpha, and stopping criterions.
    output: Graph G with additional properties. For each vertex v, it adds 
    property 'br_v' so that br_v[i] is the pagerank of node i when we use
    restart distribution that is concentrated on v.
    The nodes are assumed to be represented by integers.
    In addition the same data is returned as ad-array, where the stationary
    distribution with bias on node i is the ith raw.
    """
    n = len(G.nodes())
    W = np.zeros((n, n))
    for vertex in tqdm(G.nodes()):
        bias = np.zeros(n)
        bias[vertex] = 1
        p, _ = biasedPropagateG(G, bias=bias, alpha=alpha)
        W[vertex] = p
        for i in G.nodes():
            G.nodes[i]["br_" + str(vertex)] = p[i]
    return G, W


################## Graph experiments ######################################
# Maybe the simplest connected graph is a star, and mor generally - a hub and
# spoke grpaph. So every node is either a leaf or a hub. The problem is- every
# connected graph fits that discription. We also require no traingles in the
# graph. Then it really is a hubs and spokes.
# How do we connect hubs? either with an edge, or with a middleman node, that
# binds only hubs.
# Also note: if we take a star with n leaves- its dual is an n-clique with the
# center disconnected. It's dual is also the 2-paths (make a new graph with edge
# between nodes if there is a path of length 2 between them).

G = nx.Graph()

# we're gonna turn G into an increasingly bigger star and see the corrupting
# effect of its growing stardom on the hub and its leaves.
# 0 is the hub node

G.add_nodes_from(range(3))
G.add_edges_from([(0, i) for i in range(1, 3)])
nx.draw(G, with_labels=True)

p, W = powerIterateG(G, alpha=0.99)
p
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.85)
p
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.5)
p
plt.bar(range(len(p)), p)

# alpha near 0 should be very similar to the resart distribution (in this case,
# the uniform distribution, with little effect of the graph topology).
p, W = powerIterateG(G, alpha=0.05)
p
plt.bar(range(len(p)), p)

# add a node, repeat the experiment.
G.add_node(3)
G.add_edge(0, 3)
nx.draw(G, with_labels=True)

p, W = powerIterateG(G, alpha=0.99)
p[:2]
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.85)
p[:2]
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.5)
p[:2]
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.05)
p[:2]
plt.bar(range(len(p)), p)

G.add_node(4)
G.add_edge(0, 4)
nx.draw(G, with_labels=True)

p, W = powerIterateG(G, alpha=0.99)
p[:2]
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.85)
p[:2]
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.5)
p[:2]
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.05)
p[:2]
plt.bar(range(len(p)), p)

G.add_node(5)
G.add_edge(0, 5)
nx.draw(G, with_labels=True)

p, W = powerIterateG(G, alpha=0.99)
p[:2]
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.85)
p[:2]
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.5)
p[:2]
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.05)
p[:2]
plt.bar(range(len(p)), p)

# So we see that with alpha close to 1, the hub's pagerank approaches .5
# Now what happens when we connect two hubs?

G.add_edges_from([range(6, 20)])
G.add_edges_from([(6, i) for i in range(7, 20)])
G.add_edge(0, 6)
nx.draw(G, with_labels=True)

# The bigger star wins
# Two hubs together get over 0.5 of the page rank weight
p, W = powerIterateG(G, alpha=0.99)
p[:]
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.85)
p[:]
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.5)
p[:]
plt.bar(range(len(p)), p)

p, W = powerIterateG(G, alpha=0.05)
p[:]
plt.bar(range(len(p)), p)


H = pageRanksConcentratedBiasG(G)


p, _ = biasedPropagateG(G, bias=[1, 0, 0, 0])


### Trying to import graphml network

G = nx.readwrite.read_graphml("IMExEColi.graphml", node_type=int)

nx.draw(G)

plt.show()

G.nodes()

G.nodes["537"]

G.edges()

G.edges[0]


p, W = powerIterateG(G, alpha=0.85, directmethod=True)

p = biasedPropagateG(G, bias=[1, 0, 0, 0], alpha=0.85)

p[0:10]
plt.bar(range(len(p)), p)


# Testing a little spider graph
G = nx.Graph()
G.add_nodes_from(range(20))
G.add_edges_from([(0, i) for i in range(1, 8)])

G.add_edge(0, 8)
G.add_edge(8, 9)
G.add_edges_from([(9, i) for i in range(10, 15)])

G.add_edge(15, 9)

G.add_edges_from([(i, j) for i in range(15, 19) for j in range(i + 1, 20)])

nx.draw(G, with_labels=True)
plt.show()

plt.close()

H, W = pageRanksConcentratedBiasG(G, alpha=0.85)

plt.figure(figsize=(8, 8))
nx.draw(H, with_labels=True, node_color=W[8], node_size=W[8]*10000)
plt.title("the little spider graph.")
plt.show()

plt.colorbar(W[0])

plt.pcolormesh(W)

plt.imshow(W, cmap='hot', interpolation='nearest')

plt.colorbar()

fig, ax = plt.subplots()
im = ax.imshow(W, cmap='hot', interpolation='nearest')
cbar = ax.figure.colorbar(im)
ax.set_xticks(np.arange(20))
ax.set_yticks(np.arange(20))
ax.set_title("Heatmap of the biased page ranks for the little spider graph.")

def heatmap(mat, title):
    """Help function designed to plot a heatmap of
    the biased stationaries array of a graph.
    """
    n,m = mat.shape
    fig, ax = plt.subplots()
    im = ax.imshow(mat, cmap='hot', interpolation='nearest')
    cbar = ax.figure.colorbar(im)
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(n))
    ax.set_title(title)

plt.savefig("Heatmap Little Spider.png")

H, W50 = pageRanksConcentratedBiasG(G, alpha=0.5)
heatmap(W50, "Heatmap corresponding to alpha=0.5")
plt.savefig("Heatmap Little Spider alpha=0.5.png")
plt.close()


H, W10 = pageRanksConcentratedBiasG(G, alpha=0.1)
heatmap(W10, "Heatmap corresponding to alpha=0.1")
plt.savefig("Heatmap Little Spider alpha=0.1.png")
plt.close()

H, W90 = pageRanksConcentratedBiasG(G, alpha=0.9)
heatmap(W90, "Heatmap corresponding to alpha=0.9")
plt.savefig("Heatmap Little Spider alpha=0.9.png")
plt.close()

H, W25 = pageRanksConcentratedBiasG(G, alpha=0.25)
heatmap(W25, "Heatmap corresponding to alpha=0.25")
plt.savefig("Heatmap Little Spider alpha=0.25.png")
plt.close()

H, W85 = pageRanksConcentratedBiasG(G, alpha=0.85)
heatmap(W85, "Heatmap corresponding to alpha=0.85")
plt.savefig("Heatmap Little Spider alpha=0.85.png")
plt.close()

b = np.zeros(len(W))
b[16]=1

nx.draw(H, with_labels=True, node_color=W[8], node_size=W[8]*10000)

p, _ = biasedPropagateG(G, bias=b, alpha=0.85)

p[15:20]

plt.bar(range(20), p)
plt.bar(range(len(p)), p)

y = np.zeros((20,20))
for i in range(20):
    b = np.zeros(20)
    b[i]=1
    y[i], _ = biasedPropagateG(G, b, 0.85)

#y[16] = p

heatmap(p.reshape((1,20)), "")

heatmap(y, "")
