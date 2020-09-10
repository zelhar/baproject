import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.stats import poisson
from tqdm import tqdm

import skimage as ski
from skimage import io

import PIL as pil

from skimage.transform import rescale, resize, downscale_local_mean

from scipy.special import rel_entr

import matplotlib as mpl

# plt.ioff()
plt.ion()

from IPython.display import Image

################ Functions #######################################
from graphsfunctions import *


#### Plots for the Bachelor Thesis ####

G = nx.DiGraph()
G.add_nodes_from(range(7))
G.add_edges_from([(i,0) for i in range(1,6)])

G.add_edge(5,6)
#G.add_edge(6,6)
G.add_edge(6,5)
G.add_edge(0,5)

nx.draw_spring(G, with_labels=True)

nx.draw(G, with_labels=True)

nx.draw_kamada_kawai(G)


G.adj

A = nx.drawing.nx_agraph.to_agraph(G)

A.layout('dot')

A.draw('test2.png')

Image(A.draw(format='png'))

x = io.imread('test.png')
x = io.imread('test2.png')
x = io.imread('test3.png')

io.imshow(x)

A = np.array(nx.adj_matrix(G).todense())

# in networkx A[i,j]=1 means from i to j. I want to reverse that so:
A = A.transpose()
A
p, _, _ = powerIterate(A)
p

nx.draw(G, with_labels=True, node_color=p, node_size=p*50000)
plt.savefig("directed_graph_example.png")
plt.close()

Gud = nx.Graph(incoming_graph_data=G)

t, _ = powerIterateG(Gud)

nx.draw(Gud, with_labels=True, node_color=t, node_size=t*50000)
plt.savefig("undirected_graph_example.png")
plt.close()


### Tests
G = nx.cycle_graph(12)
pos = nx.spring_layout(G, iterations=200)
fig = nx.draw(G, pos, node_color=range(12), node_size=800, cmap=plt.cm.Blues)

G = nx.florentine_families_graph()
nx.draw(G, with_labels=True)

G = nx.Graph()
G.add_nodes_from(range(6))
G.add_edges_from([(0,i) for i in range(1,4)])
G.add_edge(5,4)
G.add_edge(3,4)

p, w = powerIterateG(G)
p

w[0].sum() #not1

w[:,0].sum() #1

nx.draw_spring(G, with_labels=True, node_size=100, node_color=p,
        cmap=plt.cm.viridis)


A = np.array(nx.adj_matrix(G).todense())
n = len(A)

A = A.transpose()
A

# normaliz A column-wise:
d = A.sum(axis=0).reshape((1, n))
d

d[d == 0] = 1  # avoid division by 0
d

A = A / d
A

# combine A with the restart matrix
alpha=0.85
W = np.ones((n, n)) / n
W
W = (1 - alpha) * W + alpha * A  # the transition matrix
W

b = [1, 0,0,0,1,0,0]

b= np.ones(7)
p,w = biasedPropagateG(G, b)

### trying this 'spectral_partitioning' on karate
G = nx.karate_club_graph()

colors = [0 if G.nodes[v]['club'] == 'Mr. Hi' else 1 for 
        v in G.nodes()]
colors = np.array(colors)
colors
colours = ['yellow' if x==0 else 'green' for x in colors]

clublabel = ['Hi' if i==0 else 'Off' for i in colors]

clublabel = [str(i) + ":" + clublabel[i] for i in G.nodes()]

clubdict = dict(zip(G.nodes(), clublabel))
clubdict

T = np.array(nx.adj_matrix(G).todense())

n=len(T)
n

# column normalized
A = T / T.sum(axis=0)

K = diffKernel(A)

eigvals, eigvects = np.linalg.eig(K)

eigvals

x = np.argsort(eigvals)
x

# The fiedler eigenvector is the one corresponding to the second largest
# eigenvalue. We use it to create the partition.

y = eigvects[:,x[-2]]

z = np.sign(y)

d = np.array([1 if x<0 else 0 for x in z]) 

colors - d #2 mistakes

nx.draw_spring(G, with_labels=True, node_color=d, node_size=800,
        cmap=plt.cm.coolwarm)

testvals, testvects = np.linalg.eig(A)

testvects == eigvects

testvects[:,0] + eigvects[:,0]

testvects[:,1] - eigvects[:,1]






























































































