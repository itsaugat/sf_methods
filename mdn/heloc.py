from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from evaluation_metrics import *
from utils import load_custom_data, get_numerical_std, categorical_embed
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import warnings
warnings.simplefilter('ignore', UserWarning)


def mdn_loop(x, y, x_orig, train_index, test_index, cat_idx, num_idx, target, features, cat_cols, num_cols, cat_unique, cat_embed, std_dict, df_sc, df_embed, num_perturbations, transformer):

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
        query_orig = x_orig[i]
        #print(query_orig)
        label = y[i]

        # get the neighbors of the query
        dist_q, ind_q = nbrs.kneighbors(np.reshape(query, (1, -1)))
        # get nearest hit and nearest miss
        nearest_hit_idx = get_nearest_hit_index(label, ind_q, y_train)
        # get nmotb idx of the query
        nmotb_idx = get_nmotb_index(label, ind_q, dist_q, nearest_hit_idx, y_train)

        mdn_sf, _ = get_mdn_sf(query_orig, label, target, features, num_idx, cat_idx, cat_cols, cat_embed, df_embed, df_sc, std_dict)

        # Sparsity
        sparse = calculate_sparsity(mdn_sf, query)
        sparsity.append(sparse)

        # SF-Query Distance
        sf_query.append(calculate_l2_dist(mdn_sf, query))

        # SF-NMOTB Distance
        if nmotb_idx is not None:
            nmotb = x_train[nmotb_idx]
            # print(nmotb)
            sf_nun.append(calculate_l2_dist(mdn_sf, nmotb))
            # print(calculate_l2_dist(mdn_sf, nmotb))
            # SF-kNN(%)
            num_k_sf_nh, num_k_sf_nmotb = calculate_k(nearest_hit_idx, nmotb_idx, nbrs, mdn_sf)
            sf_nh_knn.append(num_k_sf_nh)
            sf_nun_knn.append(num_k_sf_nmotb)

        # Mahalanobis Distance
        maha_pos, maha_neg = calculate_mahalanobis(x_train, y_train, label, mdn_sf)
        mahalanobis.append(maha_pos)
        # OOD Distance
        ood_dist, _ = calculate_ood(nbrs, y_train, label, mdn_sf)
        # print(ood_dist)
        ood_distance.append(ood_dist)
        # Trust score
        trust = trust_model.get_score(np.reshape(mdn_sf, (1, -1)), np.reshape(label, (1, -1)))
        # print(trust[0][0])
        trust_score.append(trust[0][0])
        # Lipschitz constant
        lips = calculate_lipschitz(query, mdn_sf, num_perturbations, query_orig, label, target, features, num_idx, cat_idx, cat_cols, num_cols, cat_unique, cat_embed, df_embed, df_sc, std_dict, transformer)
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
        results = list(tqdm(pool.starmap(mdn_loop, arguments), total=len(arguments), desc="Processing"))
        pbar.update(len(arguments))  # Manually update progress bar to ensure completion

    return results


if __name__ == '__main__':

    # Specifications
    data_desc = 'heloc'
    target = 'class'
    method = 'mdn'

    cat_idx = []
    num_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    # for lipschitz constant
    num_perturbations = 100

    # Load data
    X, y, x_orig, features, cat_cols, num_cols, cat_unique, data_sc, data_embed, transformer = load_custom_data(data_desc, target)

    # get standard deviation of the numerical features
    std_dict = get_numerical_std(data_desc, target)

    # get categorical embeddings
    cat_embed = categorical_embed[data_desc]

    n_folds = 5 #5

    # Generate fold indices for k-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=None)

    arguments = []

    for train_index, test_index in kfold.split(X,y):
        # df_sc with training data
        df_sc = data_sc.iloc[train_index]
        # df_embed with training data
        df_embed = data_embed.iloc[train_index]
        # create arguments list
        arguments.append((X, y, x_orig, train_index, test_index, cat_idx, num_idx, target, features, cat_cols, num_cols, cat_unique, cat_embed, std_dict, df_sc, df_embed, num_perturbations, transformer))

    parallel_results = run_parallel(arguments)

    #write the result dictionary to pickle file
    with open('./results/' + data_desc + '_' + method + '.pickle', 'wb') as f:
        pickle.dump(parallel_results, f)