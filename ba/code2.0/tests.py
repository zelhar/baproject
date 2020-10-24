import numpy as np
import networkx as nx
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import poisson
from tqdm import tqdm


from graph_utils import *


#### Tests

G = nx.karate_club_graph()

p = biasedPropagateGA(G)

T = np.array(nx.adj_matrix(G).todense())
T = T.transpose()  # we want A[i,j]=1 to mean from j to i

K = diffKernel(T)

p0 = np.ones(len(T)) / len(T)

q = np.dot(K, p0)

(p - q).sum()

p
q

alpha = 0.15

H = diffKernelG(G)

r = np.dot(H, p0)

(p - r).sum()

np.linalg.norm((p - q), 1)
np.linalg.norm((p - r), 1)
np.linalg.norm((q - r), 1)

G = nx.read_graphml("../yeast_4_groups.graphml")  # watch out keyerror 'long' bug.
G = nx.convert_node_labels_to_integers(G)
G.nodes[0]["Group"]
G = nx.to_networkx_graph(G)
ccs = nx.connected_components(G)
largest_cc = max(nx.connected_components(G), key=len)
S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
G = G.subgraph(largest_cc).copy()
G = nx.convert_node_labels_to_integers(G)

x, df = predictMethod2_diffkernel(G, tries=50, alpha=0.2)

y, df = predictMethod2_diffkernel(G, tries=50, alpha=0.6)

z, df = predictMethod2_diffkernel(G, tries=50, alpha=0.35)
z
np.mean(z)

z, df = predictMethod2_diffkernel(G, tries=50, alpha=0.31) #
# best result with alpha=0.31 (0.8926)
z
np.mean(z)


x1, df1 = predictMethod2(G, tries=5, alpha=0.2)

x2, df2 = predictMethod2(G, tries=5, alpha=0.2, knownfraction=0.7)

x3, df3 = predictMethod2(G, tries=5, alpha=0.2, knownfraction=0.3)
