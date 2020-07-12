import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.stats import poisson
from tqdm import tqdm

import skimage as ski
from skimage import io

from skimage.transform import rescale, resize, downscale_local_mean

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
# Compute the reduced influence matrix and create the corresponding graph and
# the connected components.
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

# Exploring graph clustering methods. 
# Since we are dealing with propagation and interested in connected scale free
# graphs such as Barabsi-Albert, I am trying to apply propagation to perform
# this task. 
# Idea 1: The hottest node is going to form the first cluster. We then extend it
# by all nodes that are connected to it in a sufficiently hot path. Remove them
# from the graph and repeat on the smaller graph. Thought should then be
# dedicated to the matter of setting the parameters: what is 'hot' (delta), how
# fast we propagate (alpha), how do we test statistical significance and
# robustness. Define null model? Use connected edge swaps for robustness tests?
n=50
m=1
seed=42

G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)

nx.draw(G, with_labels=True)

nx.draw_spectral(G, with_labels=True)

x = nx.degree(G, range(n))
x=dict(x)
x=x.values()
x=list(x)
x
x=np.array(x)

# just toying around
nx.draw_circular(G, with_labels=True, node_color=x, node_size=50*x)
nx.draw_spring(G, with_labels=True)
nx.draw_kamada_kawai(G, with_labels=True, node_color=x, node_size=50*x)
G = nx.florentine_families_graph()
nx.draw_spring(G, with_labels=True)

# trying to cluster a graph ...
G = nx.dual_barabasi_albert_graph(n=50, m1=1, m2=2, p=0.7, seed=seed)
H = G.copy()
p, _ = powerIterateG(G, alpha=0.85)
plt.bar(range(50), p)

nx.draw_kamada_kawai(G, with_labels=True, node_color=p, node_size=5000*p)

nx.draw_circular(G, with_labels=True, node_color=p, node_size=5000*p)

s = np.argmax(p)
s

W = reducedInfluenceMatrixG(G, delta=0)
heatmap(W, "")

H = nx.Graph()
H.add_nodes_from(G.nodes())

edges = [(i,j) for i in range(49) for j in range(i+1,50) if W[i,j]>0]

H.add_edges_from(edges)

nx.draw_circular(H, with_labels=True, node_color=p, node_size=5000*p)

for delta in np.arange(0.01, 1, 0.01):
    remlist = [e for e in list(H.edges()) if W[e] <= delta]
    H.remove_edges_from(remlist)
    if nx.number_connected_components(H) > 1:
        print("break", len(H.edges()))
        break

len(H.edges())

nx.draw(H)

H = nx.Graph()
H.add_nodes_from(G.nodes())
nlist = list(H.nodes())

nx.draw_circular(H, with_labels=True )

s = np.argmin(p[nlist])
s #coldest node, lets see to which clust it belongs
nlist.remove(s)
t = np.argmax([W[s,i] for i in nlist])
t
H.add_edge(s,t)
nx.draw_circular(H, with_labels=True )

# repeat
s = np.argmin(p[nlist])

H = nx.Graph()
H.add_nodes_from(G.nodes())
nlist = list(H.nodes())
while nx.number_connected_components(H) > 3:
    s = np.argmin(p[nlist])
    x = nlist[s]
    print(s,x)
    nlist.pop(s)
    t = np.argmax([W[x,i] for i in nlist])
    H.add_edge(x,nlist[t])

nx.draw_circular(H, with_labels=True)

nx.number_connected_components(H)

nx.draw_kamada_kawai(H, with_labels=True)

CCs = [list(c) for c in nx.connected_components(H)]

CCs

colors = np.zeros(50)

colors[CCs[1]]=1
colors[CCs[2]]=2


nx.draw_kamada_kawai(H, with_labels=True, node_color=colors)

nx.draw_spring(H, with_labels=True, node_color=colors)

nx.draw_spring(G, with_labels=True, node_color=colors)


# Testing image clustering
A = np.arange(9).reshape((3,3))
A

cx = lambda x: (x // 3, x % 3)

ix = lambda x: x[0]*3 + x[1]

G = nx.Graph()

G.add_nodes_from([(i,j) for i in range(3) for j in range(3)])

G.nodes()

G.add_edges_from([((i,j), (i+1,j)) for i in range(2) for j in range(3)])


G.add_edges_from([((i,j), (i,j+1)) for i in range(3) for j in range(2)])

nx.draw_circular(G, with_labels=True)

T = np.zeros((9,9))

for i,j in G.edges():
    print(ix(i),ix(j))
    T[ix(i), ix(j)] = abs(A[i] - A[j])

T = np.transpose(T) + T
T

W = T / T.sum(axis=0)

A.sum(axis=0)
A.sum(axis=1)


W.sum(axis=0)


p, Q, _ = powerIterate(W)


H = nx.Graph()
H.add_nodes_from(range(9))
nlist = list(H.nodes())
while nx.number_connected_components(H) > 3:
    s = np.argmin(p[nlist])
    x = nlist[s]
    nlist.pop(s)
    t = np.argmax([W[x,i] for i in nlist])
    H.add_edge(x,nlist[t])


CCs = [list(c) for c in nx.connected_components(H)]

CCs[0]
CCs[1]
CCs[2]

bottomUpCluster(p, 3)

X = np.arange(9)
X
X.reshape((3,3))

for i in range(3):
    X[CCs[i]] = i

X
plt.imshow(X.reshape((3,3)))


im = io.imread('chimp-665439.jpg', as_gray=True)

im.shape


x = ski.util.crop(im, ((0,0), (120,120))) 

io.imshow(x)

x.shape


y = resize(x, (64,64))

io.imshow(y)

z = y.flatten()

w = np.zeros((64*64,64*64))

#w = w + np.identity(64*64)

for i in range(63):
    for j in range(63):
        ix = 64*i + j
        w[ix, ix+1] = 1 / (abs(y[i,j] - y[i,j+1]) + 1)
        w[ix, ix + 64] = 1 / (abs(y[i,j] - y[i+1,j]) + 1)

w[-1,-2] = 1 / (abs(y[63,63] - y[63,62]) + 1)
w[-1,-65] = 1 / (abs(y[63,63] - y[62,63]) + 1)

w = w + np.transpose(w)

w[64*63].sum()

#w = w - np.identity(64*64)

p, _, __ = powerIterate(w, alpha=0.85)

cc = bottomUpCluster(p, 99)

X = np.zeros(64*64)
for i in range(99):
    X[cc[i]] = i


plt.imshow(X.reshape(64,64))



w = np.zeros((3*3,3*3))
for i in range(2):
    for j in range(2):
        ix = 3*i + j
        w[ix, ix+1] = 1
        #w[ix+1, ix] = 1
        w[ix, ix + 3] = 1
        #w[ix+3, ix] = 1

w 

for i in range(2):
    w[-i,-i - 1] = 1
    w[-i,-i - 3] = 1

w[-1,-2] = 1
w[8,5] = 1

w = w + np.transpose(w)

w[64*63].sum()


G = nx.Graph()
G.add_nodes_from(range(9))
G.add_edges_from([(i,i+1) for i in range(8)])
nx.draw(G, with_labels=True)
G.add_edges_from([(i,i) for i in range(9)])
A = nx.adj_matrix(G).todense()
powerIterate(A, alpha=1)


# second mini attempt
# we start with a 4x4 matrix, a mini image:
A = np.zeros((4,4))
T = np.zeros((16,16))
n=4
n**2
for i in range(n**2 - 1):
    if (i+1) % n > 0:
        T[i,i+1] = 1
    if (i+n) < n**2:
        T[i,i+n] = 1
T.sum()
T = T + np.transpose(T)
T
p, _, __ = powerIterate(T)
T = T + np.identity(n**2)
p, _, __ = powerIterate(T, alpha=1)

# now with a real image again
n=50
im = io.imread('chimp-665439.jpg', as_gray=True)
im.shape
x = ski.util.crop(im, ((0,0), (120,120))) 
x.shape

io.imshow(x)

y = resize(x, (n,n))
z = y.flatten()

io.imshow(y)

T = np.zeros((n**2,n**2))

for i in range(n**2 - 1):
    if (i+1) % n > 0:
        r = i // n
        c = i % n
        T[i,i+1] = 1 / (1 + abs(
            y[r,c] - y[r,c+1]))
    if (i+n) < n**2:
        T[i,i+n] = 1 / (1 + abs(
            y[r,c] - y[r+1,c]))

T.sum() 
64*63*2

T = T + np.transpose(T)

p, _, __ = powerIterate(T)

ww = pageRanksConcentratedBias(T)

cc = bottomUpCluster(T, 560)

len(cc)

X = np.zeros(n*n)
for i in range(560):
    #X[cc[i]] = i
    X[cc[i]] = z[cc[i]].mean()


plt.imshow(X.reshape(n,n))

io.imshow(X.reshape(n,n))

io.imshow(y)

io.imsave('test2.png', y)
io.imsave('reconstruct_test2.png', X.reshape(n,n))


yy=np.zeros((n,n))
yy.shape

for i in range(0, n, 2):
    yy[i] = y[i]
    yy[i+1] = y[i]
for i in range(0, n, 2):
    yy[:,i+1] = yy[:,i]

io.imshow(yy)

for i in range(0, n, 2):
    for j in range(0,n,2):
        l = ((x,y) for x in range(i,i+2) for y in range(j,j+2))
        print(l)
        break
        yy[l] = y[i,j]
        #yy[[(x,y) for x in range(i,i+2) for y in range(j,j+2)]] = y[i,j]

io.imshow(yy)

