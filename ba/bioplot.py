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

##### Begin Graphing Session

G = nx.read_graphml('yeast_4_groups.graphml')   # watch out keyerror 'long' bug.
                                                # edit source file change long
                                                # to int

G=nx.convert_node_labels_to_integers(G)

G.nodes[0]['Group']

G = nx.to_networkx_graph(G)

G

ccs = nx.connected_components(G)

ccs

largest_cc = max(nx.connected_components(G), key=len)

S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

S

G = G.subgraph(largest_cc).copy()

G.nodes()

G = nx.convert_node_labels_to_integers(G)

G.nodes()

nx.draw_spring(G)

groups = [G.nodes[x]['Group'] for x in G.nodes()]

groups

color_code_dict = dict(zip(np.unique(groups), ['blue', 'red', 'yellow', 'green']))

node_colors = [color_code_dict[k] for k in groups]

nx.draw_spring(G, node_color=node_colors)

len(G.nodes)
len(node_colors)

# The plan: keep 50% of members of each group with their known assignment. The
# Rest of the nodes are going to be 'unknow'. Run the decision algorithm on the
# unkown members, assign each of them the likelies group and test for correction
# while doing that. keep track of the true/false score. plots...

df = pd.DataFrame()

df['Group'] = groups

df

know_unknown = np.random.random(len(G.nodes)) > 0.5

df['Known'] = know_unknown # True if group membership is known

df['Group'][randomnumbers]

all_nodes = np.arange(len(G.nodes))

known_nodes = all_nodes[know_unknown]

unkown_nodes = all_nodes[know_unknown == False]

known_nodes
unkown_nodes

v = 3
bias = np.zeros_like(G.nodes)
bias[v] = 1
bias

p = biasedPropagateGv2(G, bias=bias)

biasedPropagateGv2(G, bias)

q = p*know_unknown

x = q[np.array(groups) == 'stress']


def decision_function(v, G, group_membership_list, know_unknown_list):
    """Input v: an node assumed to be an int and taken from the list of unkown
    nodes. Input G: The graph. 
    Input group_membership_list: List of strings which specifies for each node 
    to which group belongs (including the 'unkonw' nodes) this is required for
    thesting the correctness of the decision.
    Input know_unknown_list: boolean list which
    specifies for each node of the graph whether its membership is known or
    unkown. 
    output guess_group: string the group that the function assigns the node to.
    output correctness: True or false depending on the correctness of the
    guess_group vs the real group_membership_list[v] value.
    """
    known_nodes = all_nodes[know_unknown]
    unkown_nodes = all_nodes[know_unknown == False]
    bias = np.zeros_like(G.nodes)
    bias[v] = 1
    bias
    p = biasedPropagateGv2(G, bias=bias)
    group_names = np.unique(group_membership_list)
    q = p * know_unknown_list #0 on all unkown nodes
    testscores = np.zeros_like(group_names)
    for g in range(len(group_names)):
        x = q[np.array(groups) == group_names[g]] 
        testscores[g] = x.sum()
    decide = group_names[np.argmax(testscores)]
    correctness = decide == group_membership_list[v]
    return decide, correctness

decision_function(3, G, groups, know_unknown)

know_unknown2 = know_unknown.copy()
score = 0
for v in unkown_nodes:
    _, test = decision_function(v, G, groups, know_unknown2)
    score += test
    know_unknown2[v] = True
    print(test)

score / len(unkown_nodes) #got 0.777

# I think by updating known and unkown list on the go we might achive better
# results than 0.77
# na just got 0.78

pageRank = biasedPropagateGv1(G, np.ones_like(G.nodes))
orderedNodeList = np.argsort(pageRank)

orderedUnkownNodeList = orderedNodeList[know_unknown == False]

know_unknown2 = know_unknown.copy()
score = 0
for v in orderedUnkownNodeList:
    _, test = decision_function(v, G, groups, know_unknown2)
    score += test
    know_unknown2[v] = True
    print(test)

score / len(unkown_nodes) #got 0.85

# the rational: I think determining the hotter unkown nodes first makes sense as
# their determination is probably easier. Then we use them for the following
# rounds.

nx.draw_spring(G, node_color=node_colors, with_labels=True, node_shape='s')

pos = nx.spring_layout(G, k=30)

nx.draw(G, pos=nx.spring_layout(G), node_size=50, node_color=node_colors)
