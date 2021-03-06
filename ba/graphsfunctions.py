import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.stats import poisson
from tqdm import tqdm


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
    The Matrix W is column-normalized so implicitly A[i,j]>0 means and edge FROM
    (column) j to (row) i.
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
    #for i in tqdm(range(maxiter)):
    for i in range(maxiter):
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
    alpha: the restart parameter, must be between 0 and 1. epsilon: convergence
    accuracy requirement maxiter: additional stop condition for the loop,
    whichever reached first stops the loop. If directmethod=True the calculation
    is done using the direct method rather than iteration. returns the
    stationary distribution as 1d-array and the transition matrix of the
    process.
    Note that in networkx the adj_matrix is row-to-column so A[i,j]=1
    means from i to j. BUT the returned matrix here is column-to-row!
    For undirected there is no problem because the adj_matrix is
    symmetric.
    """
    A = np.array(nx.adj_matrix(G).todense())
    #transpose A because we are going to column-normalize it
    #If G is a directed graph in networkx A[i,j]=1 means from i to j and we want
    # it to mean from j to i so we can column-normalize the matrix.
    A = A.transpose()
    n = len(A)
    # normaliz A column-wise:
    d = A.sum(axis=0).reshape((1, n))
    d[d == 0] = 1  # avoid division by 0
    A = A / d
    # combine A with the restart matrix
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
    #for i in tqdm(range(maxiter)):
    for i in range(maxiter):
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
    A should be in row-to-column format so A[i,j]>0 means from j to j.
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
    #for i in tqdm(range(maxiter)):
    for i in range(maxiter):
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
    A = A.transpose() #we want A[i,j]=1 to mean from j to i
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
    #for i in tqdm(range(maxiter)):
    for i in range(maxiter):
        y = np.dot(W, x)
        t[i] = np.linalg.norm((x.flatten() - y.flatten()), ord=1)
        # above, flatten so the norm will be for vectors, I think
        if t[i] < epsilon:
            return y.flatten(), W
        else:
            x = y
    return x.flatten(), W

def biasedPropagateGv2(G, bias, alpha=0.85, beta=1, epsilon=1e-7, maxiter=10 ** 7):
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
    outputs the resulting stationary distribution and nothing else.
    """
    A = np.array(nx.adj_matrix(G).todense())
    A = A.transpose() #we want A[i,j]=1 to mean from j to i
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
    #for i in tqdm(range(maxiter)):
    for i in range(maxiter):
        y = np.dot(W, x)
        t[i] = np.linalg.norm((x.flatten() - y.flatten()), ord=1)
        # above, flatten so the norm will be for vectors, I think
        if t[i] < epsilon:
            return y.flatten()
        else:
            x = y
    return x.flatten()



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
    #for vertex in tqdm(G.nodes()):
    #for vertex in G.nodes():
    nodes = list(G.nodes()) #don't expect nodes to be a range
    for vertex in range(n):
        bias = np.zeros(n)
        bias[vertex] = 1
        p, _ = biasedPropagateG(G, bias=bias, alpha=alpha)
        W[vertex] = p
        for i in range(n):
            G.nodes[nodes[i]]["br_" + str(list(G.nodes())[vertex])] = p[i]
            #G.nodes[i]["br_" + str(vertex)] = p[i]
    return G, W

def pageRanksConcentratedBiasGv2(G, alpha=0.85):
    """
    Input: graph G, restart parameter alpha, and stopping criterions.
    output: Influence matrix W.
    where the stationary
    distribution with bias on node i is the ith ROW. So W[i] is like calling 
    biasedPropagateG with bias concentrated on node i.
    """
    n = len(G.nodes())
    W = np.zeros((n, n))
    #for vertex in tqdm(G.nodes()):
    #for vertex in G.nodes():
    nodes = list(G.nodes()) #don't expect nodes to be a range
    for vertex in range(n):
        bias = np.zeros(n)
        bias[vertex] = 1
        p, _ = biasedPropagateG(G, bias=bias, alpha=alpha)
        W[vertex] = p
    return W

def pageRanksConcentratedBias(A, alpha=0.85):
    """Input A: non-negative matrix which represents a weighted adjacency matrix
    of a connected graph, 
    Input alpha: restart parameter.
    output: Matrix W.
    The nodes are assumed to be represented by integers.
    the stationary
    distribution with bias on node i is the ith raw.
    """
    n = len(A)
    W = np.zeros((n, n))
    #for vertex in tqdm(range(n)):
    for vertex in range(n):
        bias = np.zeros(n)
        bias[vertex] = 1
        p, _, __ = biasedPropagate(A, bias=bias, alpha=alpha)
        W[vertex] = p
    return W


def heatmap(mat, title):
    """Help function designed to plot a heatmap of
    the biased stationaries array of a graph.
    """
    n, m = mat.shape
    fig, ax = plt.subplots()
    im = ax.imshow(mat, cmap="hot", interpolation="nearest")
    cbar = ax.figure.colorbar(im)
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(n))
    ax.set_title(title)


def reducedInfluenceMatrixG(G, alpha=0.85, delta=0):
    """Input: Graph G, restart parameter alpha, and minimal influence delta.
    Output: 2d-array W, where W[i,j] is the minimum between the influence of i
    on j (= the j-th coefficient of the stationary probabilty propagated from i)
    and the influence of j on i, or 0 in case said minimum is smaller than the
    threshold delta.
    """
    n = len(G.nodes())
    _, W = pageRanksConcentratedBiasG(G, alpha=alpha)
    Delta = W >= delta
    W = Delta * W
    C = W - W.transpose() < 0
    A = C * W
    A = A + A.transpose()
    B = W == W.transpose()
    W = B*W + A
    return W


def bottomUpCluster(T, k):
    """
    Input T: a weighted adjacency matrix of a connected graph.
    obtained from a non-negative matrix or a connected graph.
    Input k: number of desired clusters k <= len(p).
    Output: a List of k lists, which represents a partition of p 
    into k clusters.
    """
    p, _, __ = powerIterate(T)
    W = pageRanksConcentratedBias(T)
    H = nx.Graph()
    H.add_nodes_from(range(len(p)))
    nlist = list(H.nodes())
    while nx.number_connected_components(H) > k:
        s = np.argmin(p[nlist])
        x = nlist[s]
        nlist.pop(s)
        t = np.argmax([W[x,i] for i in nlist])
        H.add_edge(x, nlist[t])
    CCs = [list(c) for c in nx.connected_components(H)]
    return CCs


def bottomUpClusterG(G, W, k):
    """
    Input G: undirected graph.
    Input W: the influence Matrix of G. It is better to calculate this matrix
    once and then pass it as an argument to this function rather than repeating
    this slow calulation with each call of this algorithm.
    Input k: numer of desired clusters k <= len(p).
    Output: a List of k lists, which represents a partition of p 
    into k clusters.
    """
    p, _  = powerIterateG(G)
    H = nx.Graph()
    H.add_nodes_from(range(len(p)))
    nlist = list(H.nodes())
    while nx.number_connected_components(H) > k:
        s = np.argmin(p[nlist])
        x = nlist[s]
        nlist.pop(s)
        t = np.argmax([W[x,i] for i in nlist])
        H.add_edge(x, nlist[t])
    CCs = [list(c) for c in nx.connected_components(H)]
    return CCs

def bottomUpClusterGImproved(G,k):
    """
    Input G: undirected graph.
    Input k: numer of desired clusters k <= len(p).
    Output: a List of k lists, which represents a partition of p 
    into k clusters.
    """
    G = nx.convert_node_labels_to_integers(G)
    A = np.array(nx.adj_matrix(G).todense())
    d = A.sum(axis=0)
    T = A / d
    K = diffKernel(T)
    n = len(G.nodes())
    I = np.identity(n)
    W = pageRanksConcentratedBiasGv2(G)
    p, _  = powerIterateG(G)
    H = nx.Graph()
    H.add_nodes_from(range(len(p)))
    nlist = list(H.nodes())
    while nx.number_connected_components(H) > k:
        s = np.argmin(p[nlist])
        x = nlist[s]
        nlist.pop(s)
        t = np.argmax([W[x,i] for i in nlist])
        H.add_edge(x, nlist[t])
    CCs = [list(c) for c in nx.connected_components(H)]
    cc = np.zeros(n)
    for i in range(1,k):
        cc[CCs[i]] = i
    return cc
#    nlist = list(H.nodes())
#    #ccc = [[]] * k
#    ccc = [[] for i in range(k)]
#    print("ccc = ", ccc)
#    for x in range(len(nlist)):
#        print("rechecking ", x)
#        p = W[x].copy()
#        print(p.sum())
#        p[x] = 0 #ignore self propagation
#        print(p.sum())
#        l = [p[c].sum() for c in CCs]
#        print(l)
#        i = np.argmax(l)
#        print(i)
#        ccc[i].append(x)
#        print(ccc)
#        print("Done \n")
#    return ccc

def edgeGraphG(G):
    """Input: G undirected graph.
    Output: EG whose vertices are the edges of G, numbered in the same order of
    G.edges(). The edge
    they represent is preserved as a node property 'edg'. Two vertices in G' are
    connected by an edge if their respected edges in G have a common vertex.
    """
    temp = [set(e) for e in G.edges()]
    edgesG = np.array(G.edges())
    n = len(edgesG)
    EG = nx.Graph()
    EG.add_nodes_from(range(n))
    edgesEG = [(x,y) for x in range(n-1)
            for y in range(x+1,n)
            if not temp[x].isdisjoint(temp[y])]
    EG.add_edges_from(edgesEG)
    for x in EG.nodes():
        EG.nodes[x]['edg'] = edgesG[x]
    return EG

def redoNode(G, clusters, x):
    """
    Input G: a graph. It is assumed that the nodes are labeled by an initial
    segments of the naturals 0,1,...,last. Use
    'nx.convert_node_labels_to_integers' before using this function if this is
    not the case.
    clusters: list or 1d array of integers which represents division of
    the nodes into clusters.
    -1 represent cluster undetermined, nonnegative represent real cluster
    assigment.
    x: an int which represents the x node of G.nodes()
    """
    clusters = np.array(clusters)
    clusterTypes = np.unique(clusters)
    clusterTypes.sort()
    nodes = list(G.nodes())
    neighbors = list(nx.neighbors(G, x))
    print(neighbors)
    cs = np.unique(clusters[neighbors])
    pmax = 0
    cmax = -1
    for c in cs:
        if c < 0:
            continue
        b = np.zeros_like(G.nodes)
        l = [i for i in neighbors if clusters[i] == c and c>=0]
        print(l)
        b[l] = 1
        print(b)
        p, _ = biasedPropagateG(G, bias=b)
        print(c, p[x])
        if p[x] >= pmax:
            pmax = p[x]
            cmax = c
    return cmax

def diffKernel(T, alpha=0.15):
    """
    Input T: Transition (column-normalized).
    Input alpha: restart probabilty (default=0.15)
    Output K: Kalpha [I - (1-alpha)T]^-1 is the diffusion kernel of the
    process. So if q is some restart distribution (bias) then p=Kq is the
    stationary distribution of the Markov process with restarts to q.
    """
    d = T.sum(axis=0)
    d[d == 0] = 1  # avoid division by 0
    A = T/d
    n = len(A)
    I = np.identity(n)
    B = I - (1 - alpha) * A
    K = alpha * np.linalg.inv(B)
    return K





