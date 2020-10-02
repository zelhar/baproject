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

G = nx.read_graphml('test-session.graphml', edge_key_type='str')

nx.draw_spring(G)

G.nodes['897']

G.convert_node_labels_to_integers()

G=nx.convert_node_labels_to_integers(G)

G.nodes[0]['Group']

nx.connected_components(G)

G.to_agraph?

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
