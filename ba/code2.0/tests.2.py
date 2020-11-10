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

G = nx.read_graphml("../yeast_4_groups.graphml")  # watch out keyerror 'long' bug.
G = nx.convert_node_labels_to_integers(G)
G.nodes[0]["Group"]
G = nx.to_networkx_graph(G)
ccs = nx.connected_components(G)
largest_cc = max(nx.connected_components(G), key=len)
S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
G = G.subgraph(largest_cc).copy()
G = nx.convert_node_labels_to_integers(G)

groups = [G.nodes[x]["Group"] for x in G.nodes()]

groups

known_unknown = np.random.rand(len(groups)) < 0.4

known_unknown.size
known_unknown.sum()

df, x, y = method6(G, groups, known_unknown)

df

x

y

y.shape
y.sum()

test = df['predict'] == groups

test.sum() / len(test)

gg = nx.Graph()
gg.add_nodes_from(G.nodes(data=True))

gg.nodes[0]

#gg.add_nodes_from(np.array(G.nodes())[known_unknown])

l = np.array(G.nodes())[known_unknown]

group_names = np.unique(groups)

group_nums = dict(zip(group_names, range(1, 100)))

for i in range(len(group_names)):
    gg.add_node(-i-1, Group=group_names[i])

for i in l:
    gg.add_edge(i, -group_nums[G.nodes[i]['Group']])

gg.edges()

#plt.ion()
plt.ioff()

nx.draw_spring(ggg)
plt.show()

gg.nodes()

K = diffKernelG(G, alpha=0.2)
T = np.transpose(K)

T = T - np.identity(len(T))

pr = np.dot(K, np.ones(len(K)))
order_pr = np.argsort(pr)

list_to_go = [i for i in G.nodes() if known_unknown[i] == False]


np.argmax(T[0])

all_nodes = list(G.nodes())
to_do_list = list_to_go.copy()

ggg= gg.copy()
while(nx.number_connected_components(ggg) > 4):
    i = np.argmin(pr[to_do_list])
    x = to_do_list.pop(i)
    if x == 560:
        print("jjjjjjjjjjjjjjjjjjjjj")
    all_nodes.remove(x)
    y = np.argmax(T[x][all_nodes])
    print(len(to_do_list), x, all_nodes[y])
    ggg.add_edge(x,all_nodes[y])


ccs = list(nx.connected_components(ggg))

ccs

groups

[groups[i] for i in ccs[1]]

df =pd.DataFrame()

df['group'] = groups
df

predicts = groups.copy()

for i in range(4):
    c = np.min(list(ccs[i]))
    for x in ccs[i]:
        if x >= 0:
            predicts[x] = gg.nodes[c]['Group']

df['pred'] = predicts

x = df['pred'] == df['group']

x.sum() / x.size

