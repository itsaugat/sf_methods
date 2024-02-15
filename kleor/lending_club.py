from sklearn.model_selection import KFold
from kleor_helper import get_kleor_attr_sim
from sklearn.neighbors import NearestNeighbors
from evaluation_metrics import *
from utils import load_custom_data
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import warnings
warnings.simplefilter('ignore', UserWarning)


def kleor_loop(x, y, train_index, test_index, cat_idx, num_idx, k_neighbors, num_perturbations):

    x_train = x[train_index]
    y_train = y[train_index]

    # fit nearest neighbors on training data
    # print(len(X_train))
    nbrs = NearestNeighbors(n_neighbors=len(x_train)).fit(x_train)

    # initialization for trust score
    trust_model = TrustScore()
    trust_model.fit(x_train, y_train)

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

    # for each query (instance) in the test set
    for i in test_index:

        query = x[i]
        query = np.reshape(query, (1, -1))
        # get the distances and indices for all the training data (ascending order of distance)
        distances, indices = nbrs.kneighbors(query)

        labels = [y_train[index] for index in indices[0][:k_neighbors]]
        pred = max(set(labels), key=labels.count)

        if pred == y[i]:

            # get nearest hit idx of the query
            nearest_hit_idx = get_nearest_hit_index(pred, indices, y_train)
            # print(nearest_hit_idx)
            # get nmotb idx of the query
            nmotb_idx = get_nmotb_index(pred, indices, distances, nearest_hit_idx, y_train)
            # print(nmotb_idx)

            if nearest_hit_idx is not None and nmotb_idx is not None:

                # get nearest hit and nmotb
                nmotb_sc = x_train[nmotb_idx]

                train = np.column_stack([x_train, y_train])
                pred_train = train[np.in1d(train[:, -1], pred)]

                # get kleor based semi-factuals
                #kleor_sim_miss, kleor_sim_miss_idx = get_kleor_sim_miss(pred_train, nmotb_sc)
                #kleor_global_sim, kleor_global_sim_idx = get_kleor_global_sim(pred_train, nmotb_sc, query)
                kleor_attr_sim, kleor_attr_sim_idx = get_kleor_attr_sim(pred_train, nmotb_sc, query, num_idx, cat_idx)

                # compute metrics for kleor_attr_sim
                if kleor_attr_sim is not None:

                    # Sparsity
                    sparse = calculate_sparsity(kleor_attr_sim, query[0])
                    sparsity.append(sparse)

                    # SF-Query Distance
                    sf_query.append(calculate_l2_dist(kleor_attr_sim, query))

                    # SF-NMOTB Distance
                    nmotb = x_train[nmotb_idx]
                    # print(nmotb)
                    sf_nun.append(calculate_l2_dist(kleor_attr_sim, nmotb))
                    # print(calculate_l2_dist(x_sf, nmotb))
                    # SF-kNN(%)
                    num_k_sf_nh, num_k_sf_nmotb = calculate_k(nearest_hit_idx, nmotb_idx, nbrs, kleor_attr_sim)
                    sf_nh_knn.append(num_k_sf_nh)
                    sf_nun_knn.append(num_k_sf_nmotb)

                    # Mahalanobis Distance
                    maha_pos, maha_neg = calculate_mahalanobis(x_train, y_train, pred, kleor_attr_sim)
                    mahalanobis.append(maha_pos)
                    # OOD Distance
                    ood_dist, _ = calculate_ood(nbrs, y_train, pred, kleor_attr_sim)
                    # print(ood_dist)
                    ood_distance.append(ood_dist)
                    # Trust score
                    trust = trust_model.get_score(np.reshape(kleor_attr_sim, (1, -1)), np.reshape(pred, (1, -1)))
                    # print(trust[0][0])
                    trust_score.append(trust[0][0])
                    # Lipschitz constant
                    lips = calculate_lipschitz(query, kleor_attr_sim, num_perturbations, nbrs, pred_train, num_idx, cat_idx, pred, x_train, y_train)
                    # print(lips)
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
        results = list(tqdm(pool.starmap(kleor_loop, arguments), total=len(arguments), desc="Processing"))
        pbar.update(len(arguments))  # Manually update progress bar to ensure completion

    return results


if __name__ == '__main__':

    # Specifications
    data_desc = 'lending_club'
    target = 'loan_status'
    method = 'kleor'
    k_neighbors = 3

    cat_idx = [4, 5, 6, 7]
    num_idx = [0, 1, 2, 3]

    # for lipschitz constant
    num_perturbations = 100

    # Load data
    X, y = load_custom_data(data_desc, target)

    n_folds = 5 #5

    # Generate fold indices for k-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=None)

    arguments = []

    for train_index, test_index in kfold.split(X,y):
        arguments.append((X, y, train_index, test_index, cat_idx, num_idx, k_neighbors, num_perturbations))

    parallel_results = run_parallel(arguments)

    #write the result dictionary to pickle file
    with open('./results/' + data_desc + '_' + method + '.pickle', 'wb') as f:
        pickle.dump(parallel_results, f)