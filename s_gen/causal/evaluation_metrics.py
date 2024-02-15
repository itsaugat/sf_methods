import numpy as np
import scipy as sp
from math import sqrt
from sklearn.neighbors import KDTree
from causal_helper import sgen_causal_lips

'''
Nearest Hit -> Instance that is most similar to query (Q) and has the same class as Q
'''
def get_nearest_hit_index(pred, indices, y_train):
    for index in indices[0]:
        # if the actual label of the query is same as the prediction then return it as the nearest hit
        if (y_train[index] == pred):
            return index


'''
NMOTB -> Nearest miss that lies over the decision boundary.
1. has different class from Q
2. most similar to Q
3. is not located, according to similarity, between Q and NH [Sim(Q,NH) > Sim(NMOTB,NH)]
'''
def get_nmotb_index(pred, indices, distances, nearest_hit_idx, y_train):
    # get distance between Q and NH
    dist_q_nh = distances[0][nearest_hit_idx]
    # loop through indices
    for index in indices[0]:
        if (y_train[index] != pred):
            # get distance between neighbor and NH
            dist_n_nh = distances[0][index]
            if dist_q_nh < dist_n_nh:
                return index


'''
Calculate k-ratio ->  (num of k from sf to nh / num of k from sf to nmotb)
Higher the value -> better the sf computed
Parameters : 
nh_idx -> index of nearest hit instance
nmotb_idx -> index of nmotb instance
nbrs -> nearest neighbor model fitted on training data
'''
def calculate_k(nearest_hit_idx, nmotb_idx, nbrs, sf):
    # get distances and indices for neighbors of the computed SF
    distances, indices = nbrs.kneighbors(np.reshape(sf, (1, -1)))
    # get the index of query_idx in the indices list and add 1 to get the value of k
    num_k_sf_nh = indices[0].tolist().index(nearest_hit_idx) + 1
    # get the index of nmotb_idx in the indices list and add 1 to get the value of k
    num_k_sf_nmotb = indices[0].tolist().index(nmotb_idx) + 1
    # calculate the ratio
    #k_ratio = float(num_k_sf_nh / num_k_sf_nmotb)
    return num_k_sf_nh, num_k_sf_nmotb


'''
Calculate the L2-norm distance
'''
def calculate_l2_dist(x1, x2):
    return np.linalg.norm(x1 - x2, 2)


'''
Calculate Raw Sparsity : Number of non-similar elements(feature differences) between two arrays (SF and Query)
'''
def calculate_sparsity(x1, x2):
    non_zero_threshold_sparsity = 1e-5
    diff = x1 - x2
    return np.sum(np.abs(diff[i]) > non_zero_threshold_sparsity for i in range(diff.shape[0]))


'''
Calculate Mahalanobis distance between a point and the distribution
'''
def compute_mahalanobis_dist(x=None, data=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution.
    """
    x_minus_mu = x - np.mean(data, axis=0)
    cov = np.cov(np.array(data).T)
    inv_covmat = sp.linalg.inv(cov)
    right_term = np.dot(x_minus_mu, inv_covmat)
    mahal_square = np.dot(right_term, x_minus_mu.T)
    return sqrt(mahal_square)


'''
Calculate Mahalanobis distance between SF and a class distribution
'''
def calculate_mahalanobis(x_train, y_train, pred, sf):

    train = np.column_stack([x_train, y_train])
    # get positive class from training set
    if pred == 0:
        pos_fltr = 0
        neg_fltr = 1
    elif pred == 1:
        pos_fltr = 1
        neg_fltr = 0
    pos = np.asarray([pos_fltr])
    pos_class = train[np.in1d(train[:, -1], pos)]
    pos_class = pos_class[:, :-1]
    # get negative class from training set
    neg = np.asarray([neg_fltr])
    neg_class = train[np.in1d(train[:, -1], neg)]
    neg_class = neg_class[:, :-1]

    # get mahalanobis distance for positive and negative class
    maha_pos_dist = compute_mahalanobis_dist(sf, pos_class.astype(float))
    maha_neg_dist = compute_mahalanobis_dist(sf, neg_class.astype(float))

    return maha_pos_dist, maha_neg_dist


'''
Calculate OOD Distribution measure using E Kenny's SF paper
- Distance from the SF to the nearest training datapoint
'''
def calculate_ood(nbrs, y_train, pred, sf):
    # check if nearest neighbor is the same class as the SF
    nn = True
    # get distances and indices for neighbors of the computed SF
    distances, indices = nbrs.kneighbors(np.reshape(sf, (1, -1)))

    if nn:
        idxs = indices[0][1:] #do not consider itself
        dists = distances[0][1:] #do not consifer itself
        for i in range(len(idxs)):
            if (y_train[idxs[i]] == pred):
                return dists[i], idxs[i]
    else:
        return distances[0][1], indices[0][1]


'''
Trust Score based on https://github.com/google/TrustScore/blob/master/trustscore/trustscore.py
'''


class TrustScore:
    """
    Trust Score: a measure of classifier uncertainty based on nearest neighbors.
  """

    def __init__(self, k=10, alpha=0.1, filtering="density", min_dist=1e-12):
        """
        k and alpha are the tuning parameters for the filtering,
        filtering: method of filtering. option are "none", "density",
        "uncertainty"
        min_dist: some small number to mitigate possible division by 0.
    """
        self.k = k
        self.filtering = filtering
        self.alpha = alpha
        self.min_dist = min_dist

    def filter_by_density(self, X: np.array):
        """Filter out points with low kNN density.

    Args:
    X: an array of sample points.

    Returns:
    A subset of the array without points in the bottom alpha-fraction of
    original points of kNN density.
    """
        kdtree = KDTree(X)
        knn_radii = kdtree.query(X, k=self.k)[0][:, -1]
        eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
        return X[np.where(knn_radii <= eps)[0], :]

    def fit(self, X: np.array, y: np.array):
        """Initialize trust score precomputations with training data.

    WARNING: assumes that the labels are 0-indexed (i.e.
    0, 1,..., n_labels-1).

    Args:
    X: an array of sample points.
    y: corresponding labels.
    """

        self.n_labels = np.max(y) + 1
        self.kdtrees = [None] * self.n_labels
        for label in range(self.n_labels):
            if self.filtering == "none":
                X_to_use = X[np.where(y == label)[0]]
                self.kdtrees[label] = KDTree(X_to_use)
            elif self.filtering == "density":
                X_to_use = self.filter_by_density(X[np.where(y == label)[0]])
                self.kdtrees[label] = KDTree(X_to_use)

    def get_score(self, X: np.array, y_pred: np.array):
        """Compute the trust scores.

    Given a set of points, determines the distance to each class.

    Args:
    X: an array of sample points.
    y_pred: The predicted labels for these points.

    Returns:
    The trust score, which is ratio of distance to closest class that was not
    the predicted class to the distance to the predicted class.
    """
        d = np.tile(None, (X.shape[0], self.n_labels))
        for label_idx in range(self.n_labels):
            d[:, label_idx] = self.kdtrees[label_idx].query(X, k=2)[0][:, -1]

        sorted_d = np.sort(d, axis=1)
        d_to_pred = d[range(d.shape[0]), y_pred]
        d_to_closest_not_pred = np.where(
            sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1]
        )
        return d_to_closest_not_pred / (d_to_pred + self.min_dist)



'''
Generate Perturbations centered around given query(X) in radius of epsilon
'''
def generate_perturbations(X, N):
    epsilon = 0.1
    # Initialize an array to store perturbations
    perturbations = np.zeros((N, len(X)))

    # Generate N perturbations
    for i in range(N):
        # Generate random noise within the specified radius (epsilon)
        noise = np.random.uniform(low=-epsilon, high=epsilon, size=len(X))

        # Add noise to the original array to create a perturbation
        perturbation = X + noise

        # Store the perturbation in the array
        perturbations[i, :] = perturbation

    return perturbations


'''
Calculate Lipschitz constant
'''
def calculate_lipschitz(query, sf, model, scmm, trainer, constraints, num_perturbations):
    # get perturbations
    perturbations = generate_perturbations(query, num_perturbations)

    _, perturb_sf_list = sgen_causal_lips(perturbations, constraints, model, scmm, trainer)
    perturb_sfs = []

    for i in range(perturb_sf_list.shape[0]):
        # Select the first SF from the list of SFs for each perturbation
        sf_p = perturb_sf_list[i][0]
        perturb_sfs.append(sf_p)

    amplification = np.linalg.norm(sf - perturb_sfs, axis=1) / np.linalg.norm(query - perturbations, axis=1)
    L = np.max(amplification)

    return L