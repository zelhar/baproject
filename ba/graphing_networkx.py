import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd

from scipy.stats import poisson

#plt.ioff()
plt.ion()

n, p = 150, 0.08

G = nx.erdos_renyi_graph(n=n,p=p,seed=8)

nx.draw(G, with_labels=True)

plt.show()

time.sleep(5)

plt.close()

deg_centrality = nx.degree_centrality(G)

betweenness = nx.betweenness_centrality(G)

closeness = nx.closeness_centrality(G)

node_degrees = [G.degree(i) for i in range(n)]

plt.hist([G.degree(i) for i in range(n)])

plt.scatter(range(150), y=[poisson.pmf(k=i, mu=n*p) for i in range(150)])

print("done")

df = pd.DataFrame([])

df['node'] = np.arange(150, dtype='int')

df['degree'] = node_degrees

df['deg_centrality'] = np.array(list(deg_centrality.values()))

df['betweenness'] = betweenness.values()

df['closeness'] = closeness.values()

df.to_csv('graph_properties.tsv', index=False, header=True, sep='\t')
df.to_csv('graph_properties.csv', index=False, header=True)

nx.write_gml(G, 'myGraph.gml')

ps = np.array([poisson.pmf(k=i, mu=n*p) for i in range(n)])

ds = np.array(node_degrees)

xs = [ds, ps]

plt.hist(xs)

xs = np.arange(10)

l=[i for i in range(10) if i%2==0] 

xs[l]=0

plt.hist(xs)

xs = np.arange(0,30,0.1)

plt.plot(xs, [n*poisson.pmf(k=i,mu=n*p) for i in xs])

plt.hist(ds)


plt.close()



########################## Assigment 4

## Power Iteration
def powerIterate(A, 
        alpha=0.85, 
        epsilon=1e-7, 
        maxiter=10**7,
        directmethod=False):
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
    d[d == 0] = 1 #avoid division by 0
    A = A / d
    # comnine A with the restart matrix
    W = np.ones((n, n)) / n
    W = (1 - alpha) * W + alpha * A  # the transition matrix
    # create a random state vector:
    #x = np.random.random((n, 1))
    #x = x / x.sum()
    # create a uniform state vector:
    x = np.ones((n,1)) / n
    if directmethod:
        I = np.identity(n)
        B = I - alpha * A 
        B = np.linalg.inv(B)
        p = (1 - alpha) * np.dot(B, x)
        return p, W
    t = np.zeros(maxiter)
    for i in range(maxiter):
        y = np.dot(W, x)
        t[i] = np.linalg.norm((x.flatten() - y.flatten()), ord=1)
        #above, flatten so the norm will be for vectors, I think
        if t[i] < epsilon:
            return y, W, t[:i+1]
        else:
            x = y
    return x, W, t


# test
A = np.random.randint(low=0, high=2, size=(5,5))
p, W, t = powerIterate(A)
q, T = powerIterate(A, directmethod=True)


# generate random graphs
ws = nx.watts_strogatz_graph(n=50, k=7, p=0.3)
ba = nx.barabasi_albert_graph(n=50, m=3)
er = nx.erdos_renyi_graph(n=100, p=0.08, seed=42) 
#er = nx.erdos_renyi_graph(n=100, p=0.01, seed=42)  #bad result with the direct
#er = nx.erdos_renyi_graph(n=100, p=0.4, seed=42) 

adj_er = np.array(
        nx.adj_matrix(er).todense())

# get the page rank

p, W, t = powerIterate(adj_er)
q, T = powerIterate(adj_er, directmethod=True)

np.linalg.norm(np.dot(W,p) - p, ord=1)
np.linalg.norm(np.dot(T,q) - q, ord=1)
np.linalg.norm(p - q, ord=1)

plt.plot(t)


adj_ws = np.array( nx.adj_matrix(ws).todense())
adj_ba = np.array( nx.adj_matrix(ba).todense())

p, W, t = powerIterate(adj_ws)
q, T = powerIterate(adj_ws, directmethod=True)
np.linalg.norm(np.dot(W,p) - p, ord=1)
np.linalg.norm(np.dot(T,q) - q, ord=1)
np.linalg.norm(p - q, ord=1)

p, W, t = powerIterate(adj_ba)
q, T = powerIterate(adj_ba, directmethod=True)
np.linalg.norm(np.dot(W,p) - p, ord=1)
np.linalg.norm(np.dot(T,q) - q, ord=1)
np.linalg.norm(p - q, ord=1)




# add the page rank info to the edges
n=len(er.nodes)
for i in range(n):
    er.nodes[i]['prank'] = p.flatten()[i]

nx.write_gml(er, 'er_008.gml')

