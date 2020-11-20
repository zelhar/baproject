import numpy as np
import networkx as nx
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import poisson
from tqdm import tqdm


# propagate function power method
def biasedPropagateGA(G, A=None, bias=None, alpha=0.15, epsilon=1e-7, maxiter=10 ** 7):
    """input G: a networkx type graph OR alternatively-
    input A: matrix
    which must be non negative and will be normalized into a weighted adjaceny matrix.
    Also we assume A[i,j] means from j to i (we do Av rather then vA)
    input bias: The biased restart distribution. must be
    non-negative and will be converted into a distribution. If no biased is
    given it is assumed to be uniform.
    alpha: the restart parameter, must be between 0 and 1.
        alpha is the restart probability (1-alpha) the continue on
        the graph walk probability. default is 0.15
    epsilon: convergence accuracy requirement default is 1e-7
    maxiter: additional stop condition for the loop,
    whichever reached first stops the loop. default is 1e7
    OUTPUT: the resulting stationary distribution and nothing else,
    as a flattened 1-d vector.
    """
    if A == None:
        A = np.array(nx.adj_matrix(G).todense())
        A = A.transpose()  # we want A[i,j]=1 to mean from j to i
    n = len(A)
    # normaliz A column-wise:
    d = A.sum(axis=0).reshape((1, n))
    d[d == 0] = 1  # avoid division by 0
    W = A / d
    # comnine A with the restart matrix
    if bias is None:
        bias = np.ones(n)
    bias = bias / np.sum(bias)
    B = bias.reshape((n, 1)) + np.zeros_like(A)  # make bias a column vector
    W = alpha * B + (1 - alpha) * W  # the transition matrix with bias
    x = np.ones((n, 1)) / n
    t = np.zeros(maxiter)
    # for i in tqdm(range(maxiter)):
    for i in range(maxiter):
        y = np.dot(W, x)
        t[i] = np.linalg.norm((x.flatten() - y.flatten()), ord=1)
        # above, flatten so the norm will be for vectors, I think
        if t[i] < epsilon:
            return y.flatten()
        else:
            x = y
    return x.flatten()


def diffKernel(T, alpha=0.15):
    """
    Input T: Transition (column-normalized).
    Input alpha: restart probabilty (default=0.15)
    Output K: K=alpha [I - (1-alpha)T]^-1 is the diffusion kernel of the
    process. So if q is some restart distribution (bias) then p=Kq is the
    stationary distribution of the Markov process with restarts to q.
    """
    d = T.sum(axis=0)
    d[d == 0] = 1  # avoid division by 0
    A = T / d
    n = len(A)
    I = np.identity(n)
    B = I - (1 - alpha) * A
    K = alpha * np.linalg.inv(B)
    return K


def diffKernelG(G, alpha=0.15):
    """Input G: a networkx Graph.
    Input alpha: restart parameter, default=0.15.
    output: K=alpha [I - (1-alpha)T]^-1 is the diffusion kernel of the
    process. So if q is some restart distribution (bias) then p=Kq is the
    stationary distribution of the Markov process with restarts to q.
    """
    A = np.array(nx.adj_matrix(G).todense())
    A = A.transpose()  # we want A[i,j]=1 to mean from j to i
    n = len(A)
    # normaliz A column-wise:
    d = A.sum(axis=0).reshape((1, n))
    d[d == 0] = 1  # avoid division by 0
    A = A / d
    I = np.identity(n)
    B = I - (1 - alpha) * A
    K = alpha * np.linalg.inv(B)
    return K

def decision_function_diffkernel(v, G, K, group_membership_ar, known_unknown_ar):
    """
    This function uses the given diffusion kernel of the graph to
    calculate the propagation, rather than using the power method.
    Input v: an node assumed to be an int and taken from the list of unkown
    nodes. Input G: The graph.
    Input K: the diffusion kernel of the graph.
    Input group_membership_ar: Array of strings which specifies for each node
    to which group belongs (including the 'unkonw' nodes) this is required for
    thesting the correctness of the decision.
    Input known_unknown_ar: boolean 1d array which
    specifies for each node of the graph whether its membership is known or
    unkown.
    output prediction: A string which is the predicted group affiliation.
    output correctness: True or false depending on the correctness of the
    prediction vs the real group_membership_list[v] value.
    """
    all_nodes = np.arange(len(G.nodes))
    known_nodes = all_nodes[known_unknown_ar]
    unknown_nodes = all_nodes[known_unknown_ar == False]
    bias = np.zeros_like(G.nodes)
    bias[v] = 1
    bias
    p = np.dot(K, bias)
    group_names = np.unique(group_membership_ar)
    q = p * known_unknown_ar  # 0 on all unkown nodes
    #print(known_unknown_ar.sum())
    testscores = np.zeros_like(group_names)
    for g in range(len(group_names)):
        x = q[group_membership_ar == group_names[g]]
        testscores[g] = x.sum()
    decide = group_names[np.argmax(testscores)]
    correctness = decide == group_membership_ar[v]
    #print(correctness, decide, group_membership_ar[v])
    return decide, correctness

def predictMethod2_diffkernel(G, tries=1, knownfraction=0.5, seed=42, alpha=0.2):
    """
    Like predictMethod2 but uses diffusion kernel rather than power
    method.
    Input G: a graph.
    Input tries: how many repetitions to perform, each witha
    different randomly selected 'known' group.
    Input knownfraction: portion of the known nodes out all the
    nodes.
    Input seed: random seed.
    Input alpha: restart parameter.
    """
    groups = [G.nodes[x]["Group"] for x in G.nodes()]
    groups = np.array(groups)  # array is better than a simple list ...
    group_labeling = np.unique(groups)
    df = pd.DataFrame()
    df["Group (ground truth)"] = groups
    scores = np.zeros(tries)
    # determine the pageRanks
    K = diffKernelG(G, alpha=alpha)
    pageRank = biasedPropagateGA(G, bias=np.ones_like(G.nodes), alpha=0.2)
    orderedNodeList = np.argsort(pageRank)
    # set the random seed
    np.random.seed(seed=seed)
    for t in range(tries):
        groups_predict = groups.copy()
        known_unknown = np.random.random(len(G.nodes)) < knownfraction  # known=1
        all_nodes = np.arange(len(G.nodes))
        known_nodes = all_nodes[known_unknown]
        #print(known_nodes.size, "known nodes")
        unknown_nodes = all_nodes[known_unknown == False]
        #print(unknown_nodes.size, "unknown nodes")
        orderedUnkownNodeList = orderedNodeList[known_unknown == False]
        #print(orderedUnkownNodeList.size, "orderedUnKnown: ")
        known_unknown2 = known_unknown.copy()
        #print("known_unknown2 ", known_unknown2.size, known_unknown2.sum())
        score2 = 0
        df["Rep " + str(t)] = "known"
        df["Rep " + str(t) + "_predict"] = groups_predict
        for v in orderedUnkownNodeList:
            # predict, test = decision_function(v, G, groups, known_unknown2, alpha=alpha)
            #predict, test = decision_function_diffkernel(
            #    v, G, K, groups, known_unknown2
            #)
            predict, test = decision_function_diffkernel(
                v, G, K, groups_predict, known_unknown2
            )
            groups_predict[v] = predict
            #print(test, predict)
            score2 += test
            known_unknown2[v] = True  # mark v as 'known'
            df["Rep " + str(t)][v] = "correct"
            df["Rep " + str(t) + "_predict"][v] = predict
            if not test:  # mark as incorrect and colormark if mistake
                df["Rep " + str(t)][v] = "mistake"
        score2 = score2 / len(unknown_nodes)  # got 0.85
        scores[t] = score2
    return scores, df

def predictMethod25_diffkernel(G, tries=1, knownfraction=0.5, seed=42, alpha=0.2):
    """
    Like predictMethod2 but uses diffusion kernel rather than power
    method.
    Input G: a graph.
    Input tries: how many repetitions to perform, each witha
    different randomly selected 'known' group.
    Input knownfraction: portion of the known nodes out all the
    nodes.
    Input seed: random seed.
    Input alpha: restart parameter.
    """
    groups = [G.nodes[x]["Group"] for x in G.nodes()]
    groups = np.array(groups)  # array is better than a simple list ...
    group_labeling = np.unique(groups)
    df = pd.DataFrame()
    df["Group (ground truth)"] = groups
    scores = np.zeros(tries)
    # determine the pageRanks
    K = diffKernelG(G, alpha=alpha)
    pageRank = biasedPropagateGA(G, bias=np.ones_like(G.nodes), alpha=0.2)
    orderedNodeList = np.argsort(-pageRank)
    #np.random.shuffle(orderedNodeList)
    # set the random seed
    np.random.seed(seed=seed)
    for t in range(tries):
        groups_predict = groups.copy()
        known_unknown = np.random.random(len(G.nodes)) < knownfraction  # known=1
        all_nodes = np.arange(len(G.nodes))
        known_nodes = all_nodes[known_unknown]
        #print(known_nodes.size, "known nodes")
        unknown_nodes = all_nodes[known_unknown == False]
        #print(unknown_nodes.size, "unknown nodes")
        orderedUnkownNodeList = orderedNodeList[known_unknown == False]
        #print(orderedUnkownNodeList.size, "orderedUnKnown: ")
        known_unknown2 = known_unknown.copy()
        #print("known_unknown2 ", known_unknown2.size, known_unknown2.sum())
        score2 = 0
        df["Rep " + str(t)] = "known"
        df["Rep " + str(t) + "_predict"] = groups_predict
        #print([pageRank[i] for i in orderedNodeList])
        for v in orderedUnkownNodeList:
            # predict, test = decision_function(v, G, groups, known_unknown2, alpha=alpha)
            #predict, test = decision_function_diffkernel(
            #    v, G, K, groups, known_unknown2
            #)
            predict, test = decision_function_diffkernel(
                v, G, K, groups_predict, known_unknown2
            )
            groups_predict[v] = predict
            #print(test, predict)
            score2 += test
            known_unknown2[v] = True  # mark v as 'known'
            df["Rep " + str(t)][v] = "correct"
            df["Rep " + str(t) + "_predict"][v] = predict
            if not test:  # mark as incorrect and colormark if mistake
                df["Rep " + str(t)][v] = "mistake"
        score2 = score2 / len(unknown_nodes)  # got 0.85
        scores[t] = score2
    return scores, df

def method6(G, group_membership_ar, known_unknown_ar):
    """
    """
    # Initialization
    df = pd.DataFrame()
    df["Group (ground truth)"] = group_membership_ar
    df["known"] = known_unknown_ar
    df["predict"] = "unknown"
    #groups_real = [G.nodes[x]["Group"] for x in G.nodes()]
    K = diffKernelG(G, alpha=0.2)
    T = np.transpose(K) # I rather work with rows here
    T = T - np.identity(len(T)) # not intereseted in self loops
    Pwns = np.identity(len(K)).astype('int64') # initially each  i pwns itself only
    Grps = np.zeros(len(K)).astype('int64') - 1 #-1 means no affiliation yet
    group_names = np.unique(group_membership_ar)
    group_to_num_dict = dict(zip(group_names, range(len(group_names))))
    group_to_num_dict["unknown"] = -1 
    group_num_to_name_dict = dict(enumerate(group_names))
    group_num_to_name_dict[-1] = "unknown"
    print(group_to_num_dict, group_num_to_name_dict)
    for i in range(len(K)):
        if known_unknown_ar[i] == True:
            Grps[i] = group_to_num_dict[group_membership_ar[i]]
    pageR = np.dot(K, np.ones(len(K))/len(K))
    orderedNodes = np.argsort(pageR)
    for i in orderedNodes:
        j = np.argmax(T[i]) #heaviest neighbor of i
        if Grps[i] == -1: #i unaffiliated
            Pwns[j][i] = 1 # j pwns i
            Grps[Pwns[j] == 1] = Grps[j] # the belong in the same group
        else: # i already affiliated
            if Grps[j] == -1: # j unaffiliated
                Pwns[j][i] = 1 # j pwns i
                Grps[Pwns[j] == 1] = Grps[i] # the belong in the same group
    df["predict_num"] = Grps
    for i in range(len(K)):
        df["predict"][i] = group_num_to_name_dict[Grps[i]]
    return df, Grps, Pwns


            
            

    



