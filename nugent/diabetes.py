from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from evaluation_metrics import *
from utils import load_custom_data
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import warnings
warnings.simplefilter('ignore', UserWarning)


def nugent_loop(x, y, train_index, test_index, k_neighbors, min_num_class, num_perturbations):
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

        # get the distances and indices for query (ascending order of distance)
        dist_q, ind_q = nbrs.kneighbors(query)

        labels = [y_train[index] for index in ind_q[0][:k_neighbors]]
        pred = max(set(labels), key=labels.count)

        if pred == y[i]:

            # get nugent based semi-factual
            nugent_sf = get_nugent_sf(ind_q, dist_q, x_train, y_train, min_num_class, pred, query)
            # print(nugent_sf)

            if nugent_sf is not None:

                nugent_sf = np.reshape(nugent_sf, (1, -1))

                # get nearest hit and nearest miss
                nearest_hit_idx = get_nearest_hit_index(pred, ind_q, y_train)
                # get nmotb idx of the query
                nmotb_idx = get_nmotb_index(pred, ind_q, dist_q, nearest_hit_idx, y_train)

                # Sparsity
                sparse = calculate_sparsity(nugent_sf[0], query[0])
                sparsity.append(sparse)
                # SF-Query Distance
                sf_query.append(calculate_l2_dist(nugent_sf, query))
                # SF-NMOTB Distance
                if nmotb_idx is not None:
                    nmotb = x_train[nmotb_idx]
                    # print(nmotb)
                    sf_nun.append(calculate_l2_dist(nugent_sf, nmotb))
                    # print(calculate_l2_dist(x_sf, nmotb))
                    # SF-kNN(%)
                    num_k_sf_nh, num_k_sf_nmotb = calculate_k(nearest_hit_idx, nmotb_idx, nbrs, nugent_sf)
                    sf_nh_knn.append(num_k_sf_nh)
                    sf_nun_knn.append(num_k_sf_nmotb)
                # Mahalanobis Distance
                maha_pos, maha_neg = calculate_mahalanobis(x_train, y_train, pred, nugent_sf)
                mahalanobis.append(maha_pos)
                # OOD Distance
                ood_dist, _ = calculate_ood(nbrs, y_train, pred, nugent_sf)
                # print(ood_dist)
                ood_distance.append(ood_dist)
                # Trust score
                trust = trust_model.get_score(np.reshape(nugent_sf, (1, -1)), np.reshape(pred, (1, -1)))
                # print(trust[0][0])
                trust_score.append(trust[0][0])
                # Lipschitz constant
                lips = calculate_lipschitz(query, nugent_sf, num_perturbations, nbrs, x_train, y_train, min_num_class,
                                           pred)
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
        results = list(tqdm(pool.starmap(nugent_loop, arguments), total=len(arguments), desc="Processing"))
        pbar.update(len(arguments))  # Manually update progress bar to ensure completion

    return results


if __name__ == '__main__':

    # Specifications
    data_desc = 'diabetes'
    target = 'Outcome'
    method = 'nugent'
    k_neighbors = 3
    # minimum number of instances for each class in the local case base
    min_num_class = 200  #200

    # for lipschitz constant
    num_perturbations = 100

    # Load data
    X, y = load_custom_data(data_desc, target)

    n_folds = 5 #5

    # Generate fold indices for k-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=None)

    arguments = []

    for train_index, test_index in kfold.split(X, y):
        arguments.append((X, y, train_index, test_index, k_neighbors, min_num_class, num_perturbations))

    parallel_results = run_parallel(arguments)

    # write the result dictionary to pickle file
    with open('./results/' + data_desc + '_' + method + '.pickle', 'wb') as f:
        pickle.dump(parallel_results, f)