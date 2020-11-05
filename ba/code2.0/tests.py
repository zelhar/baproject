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
np.mean(x)

y, df = predictMethod2_diffkernel(G, tries=50, alpha=0.6)
np.mean(y)

z, df = predictMethod2_diffkernel(G, tries=50, alpha=0.35)
z
np.mean(z)

z, df = predictMethod2_diffkernel(G, tries=50, alpha=0.31) #
# best result with alpha=0.31 (0.8926)
z
np.mean(z)

df

#alphas = np.arange(start=0.05, step=0.05, stop=0.9)
alphas = np.linspace(start=0.05, stop=0.9, num=10)

accs = np.zeros_like(alphas)

fracs = np.linspace(start=0.05, stop=0.9, num=alphas.size)

accs_frac = np.zeros_like(alphas)

for i in range(alphas.size):
    x, df = predictMethod2_diffkernel(G, tries=50, alpha=alphas[i])
    accs[i] = np.mean(x)
    x, df = predictMethod2_diffkernel(G, tries=50, knownfraction=fracs[i])
    accs_frac[i] = np.mean(x)
    print(i)

df = pd.DataFrame()

accs
accs_frac

plt.ioff()
plt.plot(alphas, accs, 'bs', fracs, accs_frac, 'g^')
plt.xlabel('parameter (alpha, knownfraction)')
plt.ylabel('Accuracy')
plt.title(
    'Accuracy as function of Alpha (square) and labeled coverage (triangle)')
#plt.legend()

plt.show()

plt.savefig('alpha_fraq_graph.png')

plt.close()

x1, df1 = predictMethod2_diffkernel(G, tries=5, alpha=0.2)
np.mean(x1)

x, df = predictMethod2_diffkernel(G, tries=5, alpha=0.2, knownfraction=0.0000)
np.mean(x)

x, df = predictMethod2_diffkernel(G, tries=5, alpha=0.2, knownfraction=0.1)
np.mean(x)

x, df = predictMethod2_diffkernel(G, tries=5, alpha=0.2, knownfraction=0.2)
np.mean(x)

x, df = predictMethod2_diffkernel(G, tries=5, alpha=0.2, knownfraction=0.3)
np.mean(x)

x, df = predictMethod2_diffkernel(G, tries=5, alpha=0.2, knownfraction=0.4)
np.mean(x)

x, df = predictMethod2_diffkernel(G, tries=5, alpha=0.2, knownfraction=0.5)
np.mean(x)

x, df = predictMethod2_diffkernel(G, tries=5, alpha=0.2, knownfraction=0.6)
np.mean(x)

x, df = predictMethod2_diffkernel(G, tries=5, alpha=0.2, knownfraction=0.7)
np.mean(x)

x, df = predictMethod2_diffkernel(G, tries=5, alpha=0.2, knownfraction=0.8)
np.mean(x)

x, df = predictMethod2_diffkernel(G, tries=1, alpha=0.3, knownfraction=0.34)
np.mean(x)

x, df = predictMethod2_diffkernel(G, tries=1, alpha=0.002, knownfraction=0.5)
np.mean(x)

x, df = predictMethod2_diffkernel(G, tries=1, alpha=0.902, knownfraction=0.5)
np.mean(x)

# specificity and sensitivity
z, df = predictMethod2_diffkernel(G, tries=1, alpha=0.31, knownfraction=0.35) #
# best result with alpha=0.31 (0.8926)
z
np.mean(z)

df

real_labels = df['Group (ground truth)'].copy()

predict_labels = df['Rep 0_predict'].copy()

unknown_list = df['Rep 0'] != 'known'


real_labels_unknown = real_labels[unknown_list]

predict_labels_unknown = predict_labels[unknown_list]

predict_labels_unknown

correct_incorrect = real_labels_unknown == predict_labels_unkown # True=Correct

results_df = pd.DataFrame(columns=['Label', 'P', 'TP', 'FP', 'N', 'TN', 'FN'])

# golgi
results_df['Label'] = ['golgi', 'DNA_repl', 'meiosis', 'stress']

golgi_real = real_labels_unknown == 'golgi' # positions that are in fact golgi
golgi_pred = predict_labels_unknown == 'golgi' # positions that are predicted golgi

TP_golgi = golgi_pred[golgi_real]

TP_golgi.sum()

golgi_real.sum()

FP_golgi = golgi_pred[golgi_real == False] 
FP_golgi.sum()

sen_golgi = TP_golgi.sum()/golgi_real.sum()
sen_golgi #0.90

golgi_neg_real = real_labels_unknown != 'golgi' # positions that are in fact not golgi
golgi_neg_pred = predict_labels_unknown != 'golgi' # positions that are predicted not golgi

TN_golgi = golgi_neg_pred[golgi_neg_real]

spec_golgi = TN_golgi.sum() / golgi_neg_real.sum()
spec_golgi #0.88

FN_golgi = golgi_neg_pred[golgi_neg_real == False]

results_df['P'][0] = golgi_real.sum()
results_df['TP'][0] = TP_golgi.sum()
results_df['FP'][0] = FP_golgi.sum()
results_df['N'][0] = golgi_neg_real.sum()
results_df['TN'][0] = TN_golgi.sum()
results_df['FN'][0] = FN_golgi.sum()


# DNA_repl
DNA_repl_real = real_labels_unknown == 'DNA_repl' # positions that are in fact DNA_repl
DNA_repl_pred = predict_labels_unknown == 'DNA_repl' # positions that are predicted DNA_repl

TP_DNA_repl = DNA_repl_pred[DNA_repl_real]

TP_DNA_repl.sum()

DNA_repl_real.sum()

sen_DNA_repl = TP_DNA_repl.sum()/DNA_repl_real.sum()
sen_DNA_repl #0.96

DNA_repl_neg_real = real_labels_unknown != 'DNA_repl' # positions that are in fact not DNA_repl
DNA_repl_neg_pred = predict_labels_unknown != 'DNA_repl' # positions that are predicted not DNA_repl

TN_DNA_repl = DNA_repl_neg_pred[DNA_repl_neg_real]

spec_DNA_repl = TN_DNA_repl.sum() / DNA_repl_neg_real.sum()
spec_DNA_repl #0.91

FP_DNA_repl = DNA_repl_pred[DNA_repl_real == False] 
FN_DNA_repl = DNA_repl_neg_pred[DNA_repl_neg_real == False]
results_df['P'][1] = DNA_repl_real.sum()
results_df['TP'][1] = TP_DNA_repl.sum()
results_df['FP'][1] = FP_DNA_repl.sum()
results_df['N'][1] = DNA_repl_neg_real.sum()
results_df['TN'][1] = TN_DNA_repl.sum()
results_df['FN'][1] = FN_DNA_repl.sum()


# meiosis
meiosis_real = real_labels_unknown == 'meiosis' # positions that are in fact meiosis
meiosis_pred = predict_labels_unknown == 'meiosis' # positions that are predicted meiosis

TP_meiosis = meiosis_pred[meiosis_real]

TP_meiosis.sum()

meiosis_real.sum()

sen_meiosis = TP_meiosis.sum()/meiosis_real.sum()
sen_meiosis #0.56 17 out of 30

meiosis_neg_real = real_labels_unknown != 'meiosis' # positions that are in fact not meiosis
meiosis_neg_pred = predict_labels_unknown != 'meiosis' # positions that are predicted not meiosis

TN_meiosis = meiosis_neg_pred[meiosis_neg_real]

spec_meiosis = TN_meiosis.sum() / meiosis_neg_real.sum()
spec_meiosis #0.98


FP_meiosis = meiosis_pred[meiosis_real == False] 
FN_meiosis = meiosis_neg_pred[meiosis_neg_real == False]
results_df['P'][2] = meiosis_real.sum()
results_df['TP'][2] = TP_meiosis.sum()
results_df['FP'][2] = FP_meiosis.sum()
results_df['N'][2] = meiosis_neg_real.sum()
results_df['TN'][2] = TN_meiosis.sum()
results_df['FN'][2] = FN_meiosis.sum()

# stress
stress_real = real_labels_unknown == 'stress' # positions that are in fact stress
stress_pred = predict_labels_unknown == 'stress' # positions that are predicted stress

TP_stress = stress_pred[stress_real]

TP_stress.sum()

stress_real.sum()

sen_stress = TP_stress.sum()/stress_real.sum()
sen_stress #0.54 25/46

stress_neg_real = real_labels_unknown != 'stress' # positions that are in fact not stress
stress_neg_pred = predict_labels_unknown != 'stress' # positions that are predicted not stress

TN_stress = stress_neg_pred[stress_neg_real]

spec_stress = TN_stress.sum() / stress_neg_real.sum()
spec_stress #0.99

temp = results_df.iloc[:,1:].sum(axis=0)

results_df = results_df.append({'Label':'Total'}, ignore_index=True)

results_df.iloc[4, 1:] = temp

results_df

results_df = results_df.iloc[:,:-1]

FP_stress = stress_pred[stress_real == False] 
FN_stress = stress_neg_pred[stress_neg_real == False]
results_df['P'][3] = stress_real.sum()
results_df['TP'][3] = TP_stress.sum()
results_df['FP'][3] = FP_stress.sum()
results_df['N'][3] = stress_neg_real.sum()
results_df['TN'][3] = TN_stress.sum()
results_df['FN'][3] = FN_stress.sum()

results_df['Sens'] = results_df['TP'] / results_df['P']
results_df['Spec'] = results_df['TN'] / results_df['N']
results_df['Acc'] = (results_df['TP'] + results_df['TN']) /(results_df['N']+ results_df['P'])
results_df['PPV'] = results_df['TP'] /(results_df['TP']+ results_df['FP'])

results_df.to_csv('results_alpha20frac35_contigency.tsv',index=False, sep='\t')


df.to_csv('fff',index=False, header=0, sep='\t')

from matplotlib.backends.backend_pdf import PdfPages

pd.options.display.float_format = '{:,.2f}'.format

results_df

#https://stackoverflow.com/questions/32137396/how-do-i-plot-only-a-table-in-matplotlib
plt.ion()

fig, ax =plt.subplots(figsize=(12,4))
#ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=results_df.values,colLabels=results_df.columns,loc='center')

#https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
pp = PdfPages("foo.pdf")
pp.savefig(fig, bbox_inches='tight')
pp.close()

import io

f = io.open('results_alpha20frac35_contigency.txt', 'w')

print(results_df, file=f)

print(results_df, file=f)

f.close()

print('s'')
