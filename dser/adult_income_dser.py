from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import RandomUnderSampler
from modelselection_conformal import ConformalRejectOptionGridSearchCV
from utils import *
from conformalprediction import ConformalPredictionClassifier, ConformalPredictionClassifierRejectOption, MyClassifierSklearnWrapper
from semifactual import SemifactualExplanation
from evaluation_metrics import *
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import warnings
warnings.simplefilter('ignore', UserWarning)
import random

def dser_sf_explanation(X, y, train_index, test_index, n_sf_explanations, num_perturbations, num_perturb_explanations):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # fit nearest neighbors on training data
    # print(len(X_train))
    nbrs = NearestNeighbors(n_neighbors=len(X_train)).fit(X_train)

    # initialization for trust score
    trust_model = TrustScore()
    trust_model.fit(X_train, y_train)

    # Hyperparameter tuning
    model_search = ConformalRejectOptionGridSearchCV(model_class=KNeighborsClassifier, parameter_grid=knn_parameters,
                                                     rejection_thresholds=reject_thresholds)
    best_params = model_search.fit(X_train, y_train)

    # Split training set into train and calibtration set (calibration set is needed for conformal prediction)
    X_train_calib, X_calib, y_train_calib, y_calib = train_test_split(X_train, y_train, test_size=0.2)

    # Fit & evaluate model and reject option
    model = KNeighborsClassifier(**best_params["model_params"])
    model.fit(X_train_calib, y_train_calib)
    # print(f"Model score: {model.score(X_train, y_train)}, {model.score(X_test, y_test)}")

    conformal_model = ConformalPredictionClassifier(MyClassifierSklearnWrapper(model))
    conformal_model.fit(X_calib, y_calib)
    # print(f"Conformal predictor score: {conformal_model.score(X_train, y_train)}, {conformal_model.score(X_test, y_test)}")

    # print(f'Rejection threshold: {best_params["rejection_threshold"]}')
    reject_option = ConformalPredictionClassifierRejectOption(conformal_model,
                                                              threshold=best_params["rejection_threshold"])

    explanator = SemifactualExplanation(reject_option)

    # For each sample in the test set, check if it is rejected
    y_rejects = []
    for i in range(X_test.shape[0]):
        x = X_test[i, :]
        if reject_option(x):
            y_rejects.append(i)
    # print(f"{len(y_rejects)}/{X_test.shape[0]} are rejected")

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

    # Compute explanations for all rejected test samples
    for idx in y_rejects:
        #try:
        x_orig = X_test[idx, :]
        # print(x_orig)
        pred = y_test[idx]
        # print(pred)

        X_sf = explanator.compute_diverse_explanations(x_orig, n_explanations=n_sf_explanations)

        # get the neighbors of the query
        dist_q, ind_q = nbrs.kneighbors(np.reshape(x_orig, (1, -1)))
        # get nearest hit and nearest miss
        nearest_hit_idx = get_nearest_hit_index(pred, ind_q, y_train)
        # get nmotb idx of the query
        nmotb_idx = get_nmotb_index(pred, ind_q, dist_q, nearest_hit_idx, y_train)

        for x_sf in X_sf:
            ################# evaluation metrics ########################
            # Sparsity
            sparse = calculate_sparsity(x_sf, x_orig)
            sparsity.append(sparse)
            # SF-Query Distance
            sf_query.append(calculate_l2_dist(x_sf, x_orig))
            # SF-NMOTB Distance
            if nmotb_idx is not None:
                nmotb = X_train[nmotb_idx]
                # print(nmotb)
                sf_nun.append(calculate_l2_dist(x_sf, nmotb))
                # print(calculate_l2_dist(x_sf, nmotb))
                # SF-kNN(%)
                num_k_sf_nh, num_k_sf_nmotb = calculate_k(nearest_hit_idx, nmotb_idx, nbrs, x_sf)
                sf_nh_knn.append(num_k_sf_nh)
                sf_nun_knn.append(num_k_sf_nmotb)
            # Mahalanobis Distance
            maha_pos, maha_neg = calculate_mahalanobis(X_train, y_train, pred, x_sf)
            mahalanobis.append(maha_pos)
            # OOD Distance
            ood_dist, _ = calculate_ood(nbrs, y_train, pred, x_sf)
            # print(ood_dist)
            ood_distance.append(ood_dist)
            # Trust score
            trust = trust_model.get_score(np.reshape(x_sf, (1, -1)), np.reshape(pred, (1, -1)))
            # print(trust[0][0])
            trust_score.append(trust[0][0])
            # Lipschitz constant
            lips = calculate_lipschitz(x_orig, x_sf, num_perturbations, num_perturb_explanations, explanator)
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

        # except Exception as ex:
        #     print(ex)

    return results


def run_parallel(arguments):

    with Pool() as pool, tqdm(total=len(arguments), desc="Processing") as pbar:
        results = list(tqdm(pool.starmap(dser_sf_explanation, arguments), total=len(arguments), desc="Processing"))
        pbar.update(len(arguments))  # Manually update progress bar to ensure completion

    return results


if __name__ == "__main__":

    # Specifications
    data_desc = 'adult_income'
    target = 'income'
    model_desc = 'knn'
    n_sf_explanations = 3
    n_folds = 5
    # for lipschitz constant
    num_perturbations = 100
    num_perturb_explanations = 1  # only 1 SF explanation for simplicity

    # Load data
    X, y = load_custom_data(data_desc, target)

    # Generate fold indices for k-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=None)

    arguments = []

    for train_index, test_index in kfold.split(X, y):
        arguments.append(
            (X, y, train_index, test_index, n_sf_explanations, num_perturbations, num_perturb_explanations))

    parallel_results = run_parallel(arguments)

    # write the result dictionary to pickle file
    with open('./results/' + data_desc + '_dser.pickle', 'wb') as f:
        pickle.dump(parallel_results, f)

