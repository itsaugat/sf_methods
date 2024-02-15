import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from evaluation_metrics import *
from data_analysis import actionability_constraints
from utils import get_dataset
from piece_helper import *
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import warnings
warnings.simplefilter('ignore', UserWarning)
import random

def piece_loop(X_test, X_test_sc, y_test, X_train, X_train_sc, y_train, df_train, df_test, clf, enc, train_preds,
               continuous_feature_names, categorical_feature_names, cont_feats_idx, cat_feats_idx, dataset, num_perturbations):

    action_meta = actionability_constraints(dataset)

    # # Make Counterfactual
    cf_df = df_train[(df_train.preds == 0)]
    #preds = clf.predict(X_test)

    cat_idxs = generate_cat_idxs(continuous_feature_names, enc)

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

        query = X_test[test_idx]
        test_q = X_test[test_idx]
        # get dictionary of categorical features in label encoding format
        cat_feats_q = test_q[len(cont_feats_idx):]
        if len(cat_feats_q) == 0:
            test_q_dict = {}
        else:
            cat_feats_q = enc.inverse_transform(cat_feats_q.reshape(1, -1))
            pair = zip(categorical_feature_names, cat_feats_q[0])
            test_q_dict = dict(pair)

        sf, _, _ = get_counterfactual(test_q, X_train, cf_df, continuous_feature_names, categorical_feature_names,
                                      clf, action_meta, cat_idxs, test_q_dict, df_train, enc, train_preds)

        # Extract the continuous features
        cont_feats_len = len(cont_feats_idx)
        # cont_feats = sf[:cont_feats_len]
        # # print(cont_feats)
        # # Extract categorical features
        # cat_feats = sf[cont_feats_len:]
        # # Inverse transform the categorical features
        # cat_feats = enc.inverse_transform(cat_feats.reshape(1, -1))
        # # print(cat_feats[0])
        # cat_feats = [mapping_dict[val] for val, mapping_dict in zip(cat_feats[0], blood_alcohol_dict)]
        # # print(cat_feats)
        #
        # index_list = sorted(cont_feats_idx + cat_feats_idx)
        # sf_sc = [cont_feats[cont_feats_idx.index(i)] if i in cont_feats_idx else cat_feats[cat_feats_idx.index(i)] for i
        #          in index_list]
        sf_sc = sf

        query_sc = X_test_sc[test_idx]

        label = y_test[test_idx]
        # get the neighbors of the query
        dist_q, ind_q = nbrs.kneighbors(np.reshape(query_sc, (1, -1)))
        # get nearest hit and nearest miss
        nearest_hit_idx = get_nearest_hit_index(label, ind_q, y_train)
        # get nmotb idx of the query
        nmotb_idx = get_nmotb_index(label, ind_q, dist_q, nearest_hit_idx, y_train)

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
        sf_sc = np.array(sf_sc)
        #print(sf_sc)
        maha_pos, maha_neg = calculate_mahalanobis(X_train_sc, y_train, label, sf_sc)
        #print(maha_pos)
        #print('----------')
        mahalanobis.append(maha_pos)
        # OOD Distance
        ood_dist, _ = calculate_ood(nbrs, y_train, label, sf_sc)
        # print(ood_dist)
        ood_distance.append(ood_dist)
        # Trust score
        trust = trust_model.get_score(np.reshape(sf_sc, (1, -1)), np.reshape(label, (1, -1)))
        # print(trust[0][0])
        trust_score.append(trust[0][0])
        # Lipschitz constant
        lips = calculate_lipschitz(query, query_sc, sf, sf_sc, X_train, cf_df, continuous_feature_names,
                                   categorical_feature_names, clf, action_meta, cat_idxs, df_train, enc, train_preds, num_perturbations,
                                   cont_feats_len, cont_feats_idx, cat_feats_idx)
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
        results = list(tqdm(pool.starmap(piece_loop, arguments), total=len(arguments), desc="Processing"))
        pbar.update(len(arguments))  # Manually update progress bar to ensure completion

    return results


if __name__ == '__main__':

    # Specifications
    dataset = 'heloc'
    method = 'piece'

    # for lipschitz constant
    num_perturbations = 100

    TARGET_NAME = 'class'
    cont_feats = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
    cat_feats = []

    cont_feats_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    cat_feats_idx = []

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

    n_folds = 5  #5

    # Generate fold indices for k-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=None)

    arguments = []

    for train_index, test_index in kfold.split(X, target):

        X_train_sc = X_sc[train_index]
        X_test_sc = X_sc[test_index]

        training = np.zeros(df.shape[0])
        training[train_index] = 1
        df['training'] = training

        data = np.concatenate((continuous_features.values, categorical_features_enc), axis=1)
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

        # ## Normalization
        # scaler = MinMaxScaler().fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, fit_intercept=False, class_weight='balanced')
        clf.fit(X_train, y_train)

        test_preds = clf.predict(X_test)
        train_preds = clf.predict(X_train)

        test_probs = clf.predict_proba(X_test)
        train_probs = clf.predict_proba(X_train)

        df_test['preds'] = test_preds
        df_test['probs'] = test_probs.T[1]

        df_train['preds'] = train_preds
        df_train['probs'] = train_probs.T[1]

        arguments.append((X_test, X_test_sc, y_test, X_train, X_train_sc, y_train, df_train, df_test, clf, enc, train_preds,
                          cont_feats, cat_feats, cont_feats_idx, cat_feats_idx, dataset, num_perturbations))

    parallel_results = run_parallel(arguments)

    # write the result dictionary to pickle file
    with open('./results/' + dataset + '_' + method + '.pickle', 'wb') as f:
        pickle.dump(parallel_results, f)





