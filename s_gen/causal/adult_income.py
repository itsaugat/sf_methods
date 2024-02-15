import numpy as np
import torch
import utils
import data_utils
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from causal_helper import sgen_causal
from multiprocessing import Pool
from evaluation_metrics import *
from tqdm import tqdm
import pickle
import warnings
warnings.simplefilter('ignore', UserWarning)


def sgen_loop(X_train, X_test, Y_train, Y_test, constraints, dataset, model_type, trainer, seed, lambd, N_explain, num_perturbations):

    # fit nearest neighbors on training data
    nbrs = NearestNeighbors(n_neighbors=len(X_train)).fit(X_train)

    # initialization for trust score
    trust_model = TrustScore()
    Y_train_int = Y_train.copy()
    Y_train_int = Y_train_int.astype(int)
    trust_model.fit(X_train, Y_train_int)

    # create dictionary to hold results
    results = {}
    # Results/Statistics
    sf_query = []
    sf_nh_knn = []
    sf_nun_knn = []
    sf_nun = []
    mahalanobis = []
    sparsity = []
    ood_distance = []
    trust_score = []
    lipschitz = []

    query_arr, sf_arr, model, scmm = sgen_causal(X_train, X_test, Y_train, Y_test, constraints, dataset, model_type, trainer, seed, lambd, N_explain)

    label = 1

    # loop through each test item
    for i in range(query_arr.shape[0]):
        query = query_arr[i][0]
        sf_list = sf_arr[i]
        # loop through each sf for the query
        for sf in sf_list:

            # get the neighbors of the query
            dist_q, ind_q = nbrs.kneighbors(np.reshape(query, (1, -1)))
            # get nearest hit and nearest miss
            nearest_hit_idx = get_nearest_hit_index(label, ind_q, Y_train_int)
            # get nmotb idx of the query
            nmotb_idx = get_nmotb_index(label, ind_q, dist_q, nearest_hit_idx, Y_train_int)

            # Sparsity
            sparse = calculate_sparsity(sf, query)
            sparsity.append(sparse)
            # SF-Query Distance
            sf_query.append(calculate_l2_dist(sf, query))
            # SF-NMOTB Distance
            if nmotb_idx is not None:
                nmotb = X_train[nmotb_idx]
                # print(nmotb)
                sf_nun.append(calculate_l2_dist(sf, nmotb))
                # print(calculate_l2_dist(mdn_sf, nmotb))
                # SF-kNN(%)
                num_k_sf_nh, num_k_sf_nmotb = calculate_k(nearest_hit_idx, nmotb_idx, nbrs, sf)
                sf_nh_knn.append(num_k_sf_nh)
                sf_nun_knn.append(num_k_sf_nmotb)
            # Mahalanobis Distance
            maha_pos, maha_neg = calculate_mahalanobis(X_train, Y_train_int, label, sf)
            # print(maha_pos)
            mahalanobis.append(maha_pos)
            # OOD Distance
            ood_dist, _ = calculate_ood(nbrs, Y_train_int, label, sf)
            # print(ood_dist)
            ood_distance.append(ood_dist)
            # Trust score
            trust = trust_model.get_score(np.reshape(sf, (1, -1)), np.reshape(label, (1, -1)))
            # print(trust[0][0])
            trust_score.append(trust[0][0])
            # Lipschitz constant
            lips = calculate_lipschitz(query, sf, model, scmm, trainer, constraints, num_perturbations)
            lipschitz.append(lips)

    results['sf_query'] = sf_query
    results['sf_nun'] = sf_nun
    results['sf_nh_knn'] = sf_nh_knn
    results['sf_nun_knn'] = sf_nun_knn
    results['mahalanobis'] = mahalanobis
    results['sparsity'] = sparsity
    results['ood_distance'] = ood_distance
    results['trust_score'] = trust_score
    results['lipschitz'] = lipschitz

    return results



def run_parallel(arguments):

    with Pool() as pool, tqdm(total=len(arguments), desc="Processing") as pbar:
        results = list(tqdm(pool.starmap(sgen_loop, arguments), total=len(arguments), desc="Processing"))
        pbar.update(len(arguments))  # Manually update progress bar to ensure completion

    return results


if __name__ == "__main__":

    # Specifications
    dataset = 'adult'
    algo = 'sgen'

    model_type = 'mlp'
    method = 'S-GEN'
    trainer = 'ERM'

    # No. of random test cases for which explanation is found
    N_explain = 100

    # for lipschitz constant
    num_perturbations = 10  # 50(100)

    # num of K-fold loop
    n_folds = 5  #5


    # Set the random seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    lambd = utils.get_lambdas(dataset, model_type, trainer)

    # Load the dataset
    X, Y, constraints = data_utils.process_data(dataset)
    X, Y = X.to_numpy(), Y.to_numpy()

    # Generate fold indices for k-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=None)

    arguments = []

    for train_index, test_index in kfold.split(X, Y):

        X_train = X[train_index]
        X_test = X[test_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index]

        arguments.append((X_train, X_test, Y_train, Y_test, constraints, dataset, model_type, trainer, seed, lambd, N_explain, num_perturbations))

    parallel_results = run_parallel(arguments)

    # write the result dictionary to pickle file
    with open('./results/adult_income_sgen.pickle', 'wb') as f:
        pickle.dump(parallel_results, f)