import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from evaluation_metrics import *
from data_analysis import actionability_constraints, default_credit_card_dict
from utils import *
from genetic_helper import *
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import warnings
warnings.simplefilter('ignore', UserWarning)
import random

def sgen_loop(action_meta, cat_idxs, cat_feats_idx, cont_feats_idx, categorical_features, continuous_features,
              actionable_idxs, X_train, X_train_sc, y_train, y_test, X_test, X_test_sc, clf, enc, num_perturbations, DIVERSITY_SIZE, LIPS_DIV):

    REACH_KNN = KNeighborsClassifier(p=2).fit(X_train, y_train)

    # fit nearest neighbors on training data
    # print(len(X_train))
    nbrs = NearestNeighbors(n_neighbors=len(X_train_sc)).fit(X_train_sc)

    # initialization for trust score
    trust_model = TrustScore()
    trust_model.fit(X_train_sc, y_train)

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

    for test_idx in range(len(X_test)):
        x = X_test[test_idx]
        query_sc = X_test_sc[test_idx]
        label = y_test[test_idx]

        sf_list = sgen_genetic(clf, x, X_train, continuous_features, categorical_features, action_meta, cat_idxs, actionable_idxs, REACH_KNN, label, DIVERSITY_SIZE)

        # if sf exists and list is not null
        if len(sf_list) > 0:
            # get the neighbors of the query
            dist_q, ind_q = nbrs.kneighbors(np.reshape(query_sc, (1, -1)))
            # get nearest hit and nearest miss
            nearest_hit_idx = get_nearest_hit_index(label, ind_q, y_train)
            # get nmotb idx of the query
            nmotb_idx = get_nmotb_index(label, ind_q, dist_q, nearest_hit_idx, y_train)

            # loop through each sf in the list
            for sf in sf_list:

                sf = np.array(sf)

                # Extract the continuous features
                cont_feats_len = len(cont_feats_idx)
                cont_feats = sf[:cont_feats_len]
                # print(cont_feats)
                # Extract categorical features
                cat_feats = sf[cont_feats_len:]
                # print(cat_feats)
                # Inverse transform the categorical features
                cat_feats = enc.inverse_transform(cat_feats.reshape(1, -1))
                # print(cat_feats[0])
                cat_feats = [mapping_dict[val] for val, mapping_dict in zip(cat_feats[0], default_credit_card_dict)]
                # print(cat_feats)

                index_list = sorted(cont_feats_idx + cat_feats_idx)
                # print(index_list)
                sf_sc = [cont_feats[cont_feats_idx.index(i)] if i in cont_feats_idx else cat_feats[cat_feats_idx.index(i)]
                         for i in index_list]

                # Sparsity
                sparse = calculate_sparsity(sf_sc, query_sc)
                sparsity.append(sparse)
                # SF-Query Distance
                sf_query.append(calculate_l2_dist(sf_sc, query_sc))
                # SF-NMOTB Distance
                if nmotb_idx is not None:
                    nmotb = X_train_sc[nmotb_idx]
                    # print(nmotb)
                    sf_nun.append(calculate_l2_dist(sf_sc, nmotb))
                    # print(calculate_l2_dist(mdn_sf, nmotb))
                    # SF-kNN(%)
                    num_k_sf_nh, num_k_sf_nmotb = calculate_k(nearest_hit_idx, nmotb_idx, nbrs, sf_sc)
                    sf_nh_knn.append(num_k_sf_nh)
                    sf_nun_knn.append(num_k_sf_nmotb)
                # Mahalanobis Distance
                maha_pos, maha_neg = calculate_mahalanobis(X_train_sc, y_train, label, sf_sc)
                #print(maha_pos)
                mahalanobis.append(maha_pos)
                # OOD Distance
                ood_dist, _ = calculate_ood(nbrs, y_train, label, sf_sc)
                #print(ood_dist)
                ood_distance.append(ood_dist)
                # Trust score
                trust = trust_model.get_score(np.reshape(sf_sc, (1, -1)), np.reshape(label, (1, -1)))
                #print(trust[0][0])
                trust_score.append(trust[0][0])
                # Lipschitz constant
                lips = calculate_lipschitz(x, sf, clf, X_train, continuous_features, categorical_features,
                                           action_meta, cat_idxs, actionable_idxs, REACH_KNN, label, num_perturbations, cont_feats_len, LIPS_DIV)
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

    #print(results)
    return results

def run_parallel(arguments):

    with Pool() as pool, tqdm(total=len(arguments), desc="Processing") as pbar:
        results = list(tqdm(pool.starmap(sgen_loop, arguments), total=len(arguments), desc="Processing"))
        pbar.update(len(arguments))  # Manually update progress bar to ensure completion

    return results


if __name__ == '__main__':

    # Specifications
    dataset = 'default_credit_card'
    method = 'sgen'

    # for lipschitz constant
    num_perturbations = 100

    # num of K-fold loop
    n_folds = 5  #5

    # no. of SF explanations
    DIVERSITY_SIZE = 3
    # no. of SF explanations for lipschitz
    LIPS_DIV = 1

    TARGET_NAME = 'Default'
    cont_feats = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                  'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    cat_feats = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    cont_feats_idx = [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    cat_feats_idx = [1, 2, 3, 5, 6, 7, 8, 9, 10]

    # load data
    df, df_sc = get_dataset(dataset, cont_feats, cat_feats)

    target = df[TARGET_NAME].values
    del df[TARGET_NAME]
    del df_sc[TARGET_NAME]
    X = df.values
    X_sc = np.array(df_sc)

    continuous_features = df[cont_feats]
    categorical_features = df[cat_feats]
    enc = OneHotEncoder().fit(categorical_features)
    categorical_features_enc = enc.transform(categorical_features).toarray()
    data = np.concatenate((continuous_features.values, categorical_features_enc), axis=1)

    action_meta = actionability_constraints(dataset)
    cat_idxs = generate_cat_idxs(cont_feats, enc)
    actionable_idxs = get_actionable_feature_idxs(action_meta, continuous_features, categorical_features)

    ###################### K-FOLD #####################################

    # Generate fold indices for k-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=None)

    arguments = []

    for train_index, test_index in kfold.split(X, target):

        X_train_sc = X_sc[train_index]
        X_test_sc = X_sc[test_index]

        training = np.zeros(df.shape[0])
        training[train_index] = 1
        df['training'] = training

        df_train = df[df.training == 1]
        df_test = df[df.training == 0]
        df_train = df_train.reset_index(inplace=False, drop=True)
        df_test = df_test.reset_index(inplace=False, drop=True)
        del df_train['training']
        del df_test['training']
        X_train = data[(df.training == 1).values]
        X_test = data[(df.training == 0).values]

        y_train = target[(df.training == 1).values]
        y_test = target[(df.training == 0).values]

        clf = LogisticRegression(max_iter=1000, fit_intercept=False, class_weight='balanced')
        clf.fit(X_train, y_train)

        arguments.append((action_meta, cat_idxs, cat_feats_idx, cont_feats_idx, categorical_features, continuous_features, actionable_idxs, X_train,
                  X_train_sc, y_train, y_test, X_test, X_test_sc, clf, enc, num_perturbations, DIVERSITY_SIZE, LIPS_DIV))

    parallel_results = run_parallel(arguments)

    # write the result dictionary to pickle file
    with open('./results/' + dataset + '_' + method + '.pickle', 'wb') as f:
        pickle.dump(parallel_results, f)