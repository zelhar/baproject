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
from graphsfunctions import *


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


# Testing a small barabsi-alberts graph

seed = 42
G = nx.barabasi_albert_graph(n=20, m=1, seed=seed)

nx.draw(G, with_labels=True)

H, W85 = pageRanksConcentratedBiasG(G, alpha=0.85)
heatmap(W85, "Heatmap of BA20-1 Model and alpha=0.85")
plt.savefig("Heatmap BA20-1 alpha=0.85.png")
plt.close()

nx.draw(H, with_labels=True, node_color=W85[8], node_size=W85[8]*10000)

H, W10 = pageRanksConcentratedBiasG(G, alpha=0.10)
heatmap(W10, "Heatmap of BA20-1 Model and alpha=0.10")
plt.savefig("Heatmap BA20-1 alpha=0.10.png")
plt.close()


# make a more complex graph...
seed = 42
G = nx.barabasi_albert_graph(n=25, m=2, seed=seed)

nx.draw(G, with_labels=True)

H, W85 = pageRanksConcentratedBiasG(G, alpha=0.85)
heatmap(W85, "Heatmap of BA25-2 Model and alpha=0.85")
plt.savefig("Heatmap BA25-2 alpha=0.85.png")
plt.close()

nx.draw(H, with_labels=True, node_color=W85[8], node_size=W85[8]*1000)

H, W10 = pageRanksConcentratedBiasG(G, alpha=0.10)
heatmap(W10, "Heatmap of BA25-2 Model and alpha=0.10")
plt.savefig("Heatmap BA25-2 alpha=0.10.png")
plt.close()

# reducedInfluenceMatrixG
W = reducedInfluenceMatrixG(G, delta=0.05)

heatmap(W, "")

Gd = nx.Graph()
Gd.add_nodes_from(range(25))
edges = [(i,j) for i in range(25) for j in range(i,25) if W[i,j]>0]

Gd.add_edges_from(edges)

nx.draw(Gd, with_labels=True)


l = [W[e] for e in Gd.edges()]

l

nx.draw(Gd, with_labels=True, width=4, edge_color = range(40))


