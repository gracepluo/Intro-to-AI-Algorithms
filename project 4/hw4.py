import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data(filepath):
    loaded = []
    with open(filepath, 'r') as csvfile:
        read = csv.DictReader(csvfile)   
        for row in read:
            loaded.append(dict(row))
    return loaded

def calc_features(row):
    x1 = float(row['child_mort'])
    x2 = float(row['exports'])
    x3 = float(row['health'])
    x4 = float(row['imports'])
    x5 = float(row['income'])
    x6 = float(row['inflation'])
    x7 = float(row['life_expec'])
    x8 = float(row['total_fer'])
    x9 = float(row['gdpp'])
    calc  = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9], dtype = np.float64)
    return calc

def hac(features):
    n = len(features)
    feature = np.array(features)
    Z = np.zeros((n - 1, 4))
    cluster = {i: [i] for i in range(n)}
    dist = np.zeros((feature.shape[0], feature.shape[0]))

    for i in range(feature.shape[0]):
        for j in range(i, feature.shape[0]):
            dist[i, j] = np.linalg.norm(feature[i, :] - feature[j,:])
            dist[j, i] = dist[i, j]
                
    for step in range(n-1):
        closest_i = -1 
        closest_j = -1

        #find closest 2 cluster
        for i in cluster:
            for j in cluster:
                if i >= j:
                    continue
                link_dist = -1
                for el_i in cluster[i]:
                    for el_j in cluster[j]:
                        cur_dist = dist[el_i, el_j]
                        if link_dist < cur_dist or link_dist == -1:
                            link_dist = cur_dist

                if closest_i == -1 or link_dist < closest_j:
                    closest_i = (i, j)
                    closest_j = link_dist
                elif link_dist == closest_j and (i < closest_i[0] or (i == closest_j[0] and j < closest_j[1])):
                    closest_i = (i, j)
                    closest_j = link_dist

        Z[step][0] = closest_i[0]
        Z[step][1] = closest_i[1]
        Z[step][2] = closest_j
        Z[step][3] = len(cluster[closest_i[0]]) + len(cluster[closest_i[1]])

        cluster[n + step] = cluster[closest_i[0]] + cluster[closest_i[1]]
        cluster.pop(closest_i[0])    
        cluster.pop(closest_i[1])
        
    return Z

def fig_hac(Z, names):
    f = plt.figure()
    dendrogram(Z, labels=names, leaf_rotation=90)  
    plt.tight_layout()
    return f

def normalize_features(features):
    average = np.mean(features, axis = 0)
    dev = np.std(features, axis = 0)
    norm = (features - average) / dev
    
    norm = [np.array(row) for row in norm]
    
    return norm