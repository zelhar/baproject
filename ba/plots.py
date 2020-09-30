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

p, _ = powerIterateG(G)

colors = [0 if G.nodes[v]['club'] == 'Mr. Hi' else 1 for 
        v in G.nodes()]
colors = np.array(colors)
colors #0=Hi, 1=Off
colours = ['yellow' if x==0 else 'green' for x in colors]

clublabel = ['Hi' if i==0 else 'Off' for i in colors]

clublabel = [str(i) + ":" + clublabel[i] for i in G.nodes()]

clubdict = dict(zip(G.nodes(), clublabel))
clubdict

nx.draw_spring(G, with_labels=True, node_color=colours, labels=clubdict,
        node_shape='s', node_size=5000*p)
plt.savefig("Karate_ground_truth.png")
plt.close()

#W = reducedInfluenceMatrixG(G, delta=0)

W = pageRanksConcentratedBiasGv2(G)


cc = bottomUpClusterG(G, W, 2)
clusters = np.zeros(len(W))
clusters[cc[1]]=1
clusters

colorcode = []
for i in range(len(colors)):
    x = 'yelllow'
    if colors[i] == 0:
        x = 'yellow' if clusters[i] == 0 else 'cyan'
    else:
        x = 'green' if clusters[i] == 1 else 'magenta'
    colorcode.append(x)

nx.draw_spring(G, with_labels=True, node_color=colorcode, labels=clubdict,
        node_shape='s', node_size=5000*p)
plt.savefig("Karate_coolwarmclustering.png")
plt.close()

A = np.array(nx.adj_matrix(G).todense())
n=len(A)
n

# column normalized
d = A.sum(axis=0)
d

R = np.dot(d.reshape((n,1)), d.reshape((1,n)))
R

m = len(G.edges)*2
m

T = A / A.sum(axis=0)

K = diffKernel(T)

p8 = W[8] #influence vector of 9

p8[8]
colors[8]
clusters[8]

p8[8]=0 #ignore self influence

p8_0 = p8 * (1 - colors) #mr hi
p8_1 = p8 * colors #officer

p8_0.sum()
p8_1.sum()

test_hi = colors
test_hi[1]
test_off = 1 - test_hi
test_off[8] = 0 #no self effect
test_off[8]#no self effect

np.dot(K, test_hi)[8] / test_hi.sum()
np.dot(K, test_off)[8] / test_off.sum()

ccc = bottomUpClusterGImproved(G, W, 2)

clusters2 = np.zeros(len(W))
clusters2[ccc[0]]=1
clusters2

eigvals, eigvects = np.linalg.eig(K)
#eigvals, eigvects = np.linalg.eig(T)

eigvals

x = np.argsort(eigvals)
x
y = eigvects[:,1]
z = y.copy()
fig = plt.plot(z, '.-')
plt.title('Karate Club Fiedler Vector - Unsorted')
plt.grid(True)
plt.savefig('karate_fiedler_unsorted.png')
plt.close()

z.sort()
fig = plt.plot(z, '.-')
plt.title('Karate Club Fiedler Vector - Sorted')
plt.grid(True)
plt.savefig('karate_fiedler_sorted.png')
plt.close()


z0 = eigvects[:,0].copy()
z0.sort()
plt.plot(z0, '.-')


B = K - R/m

u,v = np.linalg.eig(B)

z = np.sign(v[1,:])

z = np.sign(z + 1)
colors - z # no :(

B = np.dot(np.transpose(K), K)

foo, bar = np.linalg.eig(B)

moo = np.sign(bar[:,1]) + 1
moo = np.sign(moo)
moo

moo = 1 - moo
moo

moo - colors #just one error!

boo = 2*moo + 1
boo

boo = 4*colors + boo
boo

nx.draw_spring(G, with_labels=True, node_color=boo, cmap=plt.cm.coolwarm,
        node_size=8000*p, node_shape='s', labels=clubdict)

plt.savefig("Karate_spectralKK2clustering.png")
plt.close()

v1 = np.sign(bar[:,1])

np.dot(v1, np.dot(B,v1))

b1 = np.ones_like(v1)
np.dot(b1, np.dot(B,b1))
b1[0] = -1
np.dot(b1, np.dot(B,b1))

BB = np.diag(B.sum(axis=0)) - B
BB
looveal,loovects = np.linalg.eig(BB)

# The fiedler eigenvector is the one corresponding to the second largest
# eigenvalue. We use it to create the partition.

u,v = np.linalg.eig(T)
u
v

y = v[:,1]
y

y = eigvects[:,x[-2]]
y

y = np.sign(y)
y = np.sign(y+1)
y

y = 1 -y
y = y*2
y = y+1
y

y+colors - 1


nx.draw_spring(G, with_labels=True, node_color=(y+colors-1), cmap=plt.cm.coolwarm,
        node_size=8000*p, node_shape='s', labels=clubdict)
plt.savefig("Karate_spectralT2clustering.png")
plt.close()

d = np.array([1 if x<0 else 0 for x in z]) 

colors - d #2 mistakes

nx.draw_spring(G, with_labels=True, node_color=d, node_size=800,
        cmap=plt.cm.coolwarm)

testvals, testvects = np.linalg.eig(A)

testvects == eigvects

testvects[:,0] + eigvects[:,0]

testvects[:,1] - eigvects[:,1]

###

D = np.diag(d)
L = D - A

u,v = np.linalg.eig(L)
np.argmin(u) 
np.argsort(u)
u[7]
x = v[:,7]
np.dot(L,x)
x = v[:,9] #2nd largest
x = np.sign(x)
x
x+colors
nx.draw_spring(G, with_labels=True, node_color=(x+colors+1), cmap=plt.cm.coolwarm,
        node_size=8000*p, node_shape='s', labels=clubdict)
plt.savefig("Karate_spectral_Laplacian2clustering.png")
plt.close()

KK = np.transpose(K)*K

u,v = np.linalg.eig(KK)
np.argmin(u) 
np.argsort(u)
x = v[:,1]
x = np.sign(x)
x
x = 1-x
x+colors
nx.draw_spring(G, with_labels=True, node_color=(x+colors), cmap=plt.cm.coolwarm,
        node_size=8000*p, node_shape='s', labels=clubdict)
plt.savefig("Karate_spectral_KK(elementwise)2clustering.png")
plt.close()

u,v = np.linalg.eig(A)

x = v[:,1]
x = np.sign(x)
x = (1-x)*2

nx.draw_spring(G, with_labels=True, node_color=(x+colors), cmap=plt.cm.coolwarm,
        node_size=8000*p, node_shape='s', labels=clubdict)
plt.savefig("Karate_spectral_A_(not normalized)_2clustering.png")
plt.close()


M = np.dot(np.transpose(K),K)

ftest = lambda x: np.dot(x, np.dot(M, x))

#### Lets construct a graph with 3 cliques and see how it can be clustered by
#### fiedler eigenvector of A and of K^tK

G = nx.Graph()
G.add_nodes_from(range(12))
G.add_edges_from([(i,j) for i in range(1,5) for j in range(i)])
G.add_edges_from([(i,j) for i in range(6,9) for j in range(5,i)])
G.add_edges_from([(i,j) for i in range(10,12) for j in range(9,i)])
G.add_edge(4,5)
G.add_edge(8,9)
G.add_node(12)
G.add_edge(9,12)
G.add_edge(4,12)

p, _ = powerIterateG(G)

nx.draw_spring(G, with_labels=True, node_size=5000*p)

A = np.array(nx.adj_matrix(G).todense())
d = A.sum(axis=0)
T = A / d
T.sum(axis=0)
T[1].sum()
T[:,0].sum()

eu, ev = np.linalg.eig(T)

eu

y = ev[:,1]

fig = plt.plot(y, '.-')
plt.grid(True)
plt.title('Toy Graph Fiedler Vector')
plt.savefig('Toygraph_Fiedler.png')
plt.close()

x = ev[:,1]
x = np.sign(x)
x


nx.draw_spring(G, with_labels=True, node_color=x, cmap=plt.cm.prism,
        node_size=8000*p, node_shape='s')
plt.savefig("example_spectralT2clustering.png")
plt.close()
#bing

n = len(G.nodes())
I = np.identity(n)
W = pageRanksConcentratedBiasGv2(G)
K = diffKernel(T)

np.dot(K, I[0]) - W[0] #good

B = np.dot(np.transpose(K), K)

eku, ekv = np.linalg.eig(B)

y = ekv[:,1]
y = np.sign(y)
y

nx.draw_spring(G, with_labels=True, node_color=y, cmap=plt.cm.prism,
        node_size=8000*p, node_shape='s')
plt.savefig("example_spectralKK2clustering.png")
plt.close()
#bingo!

G=nx.convert_node_labels_to_integers(G)

c3 = bottomUpClusterGImproved(G,3)
c2 = bottomUpClusterGImproved(G,2)

nx.draw_spring(G, with_labels=True, node_color=c2, cmap=plt.cm.prism,
        node_size=8000*p, node_shape='s')
plt.savefig("example_coolwarm2clusters.png")
plt.close()

nx.draw_spring(G, with_labels=True, node_color=c3, cmap=plt.cm.prism,
        node_size=8000*p, node_shape='s')
plt.savefig("example_coolwarm3clusters.png")
plt.close()









































































