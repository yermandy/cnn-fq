import numpy as np
import os
from itertools import combinations
from datetime import date

n_pairs_per_id = 70

file = np.genfromtxt("resources/casia_boxes_refined.csv", dtype=np.str, delimiter=",")
features = np.load("resources/features_casia_0.5.npy")

templates = file[:, 0]
templates = [t.split("/")[0] for t in templates] 
templates = np.array(templates)

np.set_printoptions(threshold=10)

def find_distances(pairs):
    shape = (pairs.shape[0], 256)
    t1 = np.empty(shape)
    t2 = np.empty(shape)
    for i in range(pairs.shape[0]):
        t1[i, :] = features[pairs[i, 0]]
        t2[i, :] = features[pairs[i, 1]]
    # assume that template descriptors length 1
    distances = 1 - np.einsum("ij,ij->i", t1, t2)
    return distances

def select_n(identities, n):
    pairs = combinations(identities, 2)
    pairs = [*pairs]
    pairs = np.array(pairs)
    
    # swap columns of each second pair 
    pairs[0::2] = pairs[0::2][:, [1, 0]] 

    distances = find_distances(pairs)
    dist_sorted = np.argsort(distances)
    pairs = pairs[dist_sorted]
    pairs = np.vstack((pairs[-5-int(np.ceil(n/2)):], pairs[:int(np.floor(n/2))]))
    return pairs

def populate_triplets(triplets, templates, u):
    same_idx = np.flatnonzero(templates == u) # A_i, B_i
    diff_idx = np.flatnonzero(templates != u) # C_i

    n_chosen = len(same_idx) if len(same_idx) <= n_pairs_per_id * 2 else n_pairs_per_id * 2
    n_chosen = n_chosen if n_chosen % 2 == 0 else n_chosen - 1
    half = int(n_chosen / 2)
    # A_and_B = np.random.choice(same_idx, n_chosen, replace=False)
    A_and_B = select_n(same_idx, half)
    A, B = A_and_B[:,0], A_and_B[:,1]
    C = np.random.choice(diff_idx, len(A), replace=False)
    triplet = np.vstack((A, B, C)).T
    if triplets is None:
        triplets = triplet
    else:
        triplets = np.vstack((triplets, triplet))
    return triplets

def create_triplets():    
    # find such x that divides the dataset into 20% for validation and 80% for training
    unique, counts = np.unique(templates, return_counts=True)
    cumulative_counts = np.cumsum(counts)
    n_files = len(file[:, 0])
    x = np.searchsorted(cumulative_counts, n_files * 0.2)
    x = cumulative_counts[x]

    # separate validation from training
    val_templates = templates[:x]
    trn_templates = templates[x:]    

    # populate val_triplets and trn_triplets
    trn_triplets = None
    val_triplets = None
    for u in np.unique(val_templates):
        val_triplets = populate_triplets(val_triplets, val_templates, u)
    for u in np.unique(trn_templates):
        trn_triplets = populate_triplets(trn_triplets, trn_templates, u)

    return val_triplets, trn_triplets, 

def find_labels(triplets):
    pairs_A_B = triplets[:, [0, 1]]
    pairs_A_C = triplets[:, [0, 2]]
    pairs_B_C = triplets[:, [1, 2]]

    distances_A_B = find_distances(pairs_A_B)
    distances_A_C = find_distances(pairs_A_C)
    distances_B_C = find_distances(pairs_B_C)

    labels = distances_A_B < np.min((distances_A_C, distances_B_C), axis=0)
    labels = np.atleast_2d(labels).T

    print(f'triplets num: {triplets.shape[0]}')
    print(f'0\'s triplets: {labels[labels == 0].shape[0]}')
    print(f'1\'s triplets: {labels[labels == 1].shape[0]}')

    triplets = np.concatenate((triplets, labels), axis=1)

    # shuffle triptets to get random order of idenities
    np.random.shuffle(triplets) 

    return triplets
 
val_triplets, trn_triplets = create_triplets()

print('validation')
val_triplets = find_labels(val_triplets)
print('\ntraining')
trn_triplets = find_labels(trn_triplets)

np.savetxt(f'resources/casia_val_{date.today()}.csv', val_triplets, delimiter=",", fmt="%s")
np.savetxt(f'resources/casia_trn_{date.today()}.csv', trn_triplets, delimiter=",", fmt="%s")
