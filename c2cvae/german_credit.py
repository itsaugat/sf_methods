import torch
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from utils import load_custom_data, categorical_embed
from evaluation_metrics import *
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import random
import warnings
warnings.simplefilter('ignore', UserWarning)
from models.german_vae import VariationalAutoencoder
from models.german_c2c_vae import c2c_VariationalAutoencoder, c2c_latent_dims


def c2c_vae_loop(x, y, train_index, test_index, cat_embed, device, data_desc, num_features, num_perturbations):

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

    # load vae and c2c_vae models
    vae = VariationalAutoencoder(input_dim=num_features, output_dim=num_features)
    vae = vae.to(device)
    vae_file = data_desc + '_vae.pth'
    vae.load_state_dict(torch.load(vae_file, map_location=device))
    vae.eval()

    c2c_vae = c2c_VariationalAutoencoder()
    c2c_vae = c2c_vae.to(device)
    c2c_vae_file = data_desc + '_c2c.pth'
    c2c_vae.load_state_dict(torch.load(c2c_vae_file, map_location=device))
    c2c_vae.eval()

    # for each query (instance) in the test set
    for i in test_index:

        query = x[i]
        label = y[i]

        sf = get_c2c_sf(query, label, cat_embed, device, vae, c2c_vae, c2c_latent_dims)

        # get the neighbors of the query
        dist_q, ind_q = nbrs.kneighbors(np.reshape(query, (1, -1)))
        # get nearest hit and nearest miss
        nearest_hit_idx = get_nearest_hit_index(label, ind_q, y_train)
        # get nmotb idx of the query
        nmotb_idx = get_nmotb_index(label, ind_q, dist_q, nearest_hit_idx, y_train)

        # Sparsity
        sparse = calculate_sparsity(sf, query)
        sparsity.append(sparse)
        # SF-Query Distance
        sf_query.append(calculate_l2_dist(sf, query))
        # SF-NMOTB Distance
        if nmotb_idx is not None:
            nmotb = x_train[nmotb_idx]
            # print(nmotb)
            sf_nun.append(calculate_l2_dist(sf, nmotb))
            # print(calculate_l2_dist(x_sf, nmotb))
            # SF-kNN(%)
            num_k_sf_nh, num_k_sf_nmotb = calculate_k(nearest_hit_idx, nmotb_idx, nbrs, sf)
            sf_nh_knn.append(num_k_sf_nh)
            sf_nun_knn.append(num_k_sf_nmotb)
        # Mahalanobis Distance
        maha_pos, maha_neg = calculate_mahalanobis(x_train, y_train, label, sf)
        mahalanobis.append(maha_pos)
        # OOD Distance
        ood_dist, _ = calculate_ood(nbrs, y_train, label, sf)
        # print(ood_dist)
        ood_distance.append(ood_dist)
        # Trust score
        trust = trust_model.get_score(np.reshape(sf, (1, -1)), np.reshape(label, (1, -1)))
        # print(trust[0][0])
        trust_score.append(trust[0][0])
        # Lipschitz constant
        lips = calculate_lipschitz(query, sf, label, cat_embed, device, vae, c2c_vae, c2c_latent_dims, num_perturbations)
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
        results = list(tqdm(pool.starmap(c2c_vae_loop, arguments), total=len(arguments), desc="Processing"))
        pbar.update(len(arguments))  # Manually update progress bar to ensure completion

    return results


if __name__ == '__main__':

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Specifications
    data_desc = 'german_credit'
    target = 'class'
    num_features = 20

    method = 'c2c'

    # for lipschitz constant
    num_perturbations = 100
    # no. of k-folds
    n_folds = 5  # 5

    # Load data
    X, y = load_custom_data(data_desc, target)

    # categorical embeddings
    cat_embed = categorical_embed[data_desc]

    # Generate fold indices for k-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=None)

    arguments = []

    for train_index, test_index in kfold.split(X,y):
        arguments.append((X, y, train_index, test_index, cat_embed, device, data_desc, num_features, num_perturbations))

    parallel_results = run_parallel(arguments)

    #write the result dictionary to pickle file
    with open('./results/' + data_desc + '_' + method + '.pickle', 'wb') as f:
        pickle.dump(parallel_results, f)