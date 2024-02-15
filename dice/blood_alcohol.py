from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import dice_ml
from sklearn.neighbors import NearestNeighbors
from utils import load_custom_data, categorical_embed, actionability_constraints
from evaluation_metrics import *
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import random
import warnings
warnings.simplefilter('ignore', UserWarning)

def dice_loop(data, label, data_sc, train_index, test_index, cols, numerical, categorical, cat_idx, cont_idx, target, transformer, cat_embed, features_to_vary, num_perturbations):

    train_dataset = data.iloc[train_index]
    train_dataset_sc = data_sc.iloc[train_index]
    x_train = train_dataset.drop(target, axis=1)
    x_train_sc = np.array(train_dataset_sc.drop(target, axis=1))
    y_train = label[train_index]
    y_train_sc = np.array(y_train)

    test_dataset = data.iloc[test_index]
    test_dataset_sc = data_sc.iloc[test_index]
    x_test = test_dataset.drop(target, axis=1)
    x_test_sc = np.array(test_dataset_sc.drop(target, axis=1))
    y_test = label[test_index]
    y_test = y_test.tolist()

    d = dice_ml.Data(dataframe=train_dataset, continuous_features=numerical, outcome_name=target)

    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical),
            ('cat', categorical_transformer, categorical)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', transformations),
                          ('classifier', RandomForestClassifier())])
    model = clf.fit(x_train, y_train)

    # # pre-requistes for evaluation metrics
    # fit nearest neighbors on training data
    nbrs = NearestNeighbors(n_neighbors=len(x_train_sc)).fit(x_train_sc)

    # # initialization for trust score
    trust_model = TrustScore()
    trust_model.fit(x_train_sc, y_train_sc)

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

    for idx in range(len(x_test)):

        test = x_test[idx:idx+1]
        label = y_test[idx]
        # no. of dice sfs
        num = 3
        query, query_pred, sfs = get_dice_sf(test, label, d, model, num, features_to_vary)
        # remove the target (label) value from the query
        query = query[:-1]

        # check the prediction of the query is the same as the label and the desired class
        if (query_pred == label):

            # get query in scaled form
            query_sc = x_test_sc[idx]
            # get the neighbors of the query
            dist_q, ind_q = nbrs.kneighbors(np.reshape(query_sc, (1, -1)))
            # get nearest hit and nearest miss
            nearest_hit_idx = get_nearest_hit_index(label, ind_q, y_train_sc)
            # get nmotb idx of the query
            nmotb_idx = get_nmotb_index(label, ind_q, dist_q, nearest_hit_idx, y_train_sc)

            # loop through each sf
            for sf in sfs:
                # remove target (label) from the end
                sf = sf[:-1]
                # create a fresh copy of the sf in the original format
                sf_orig = sf.copy()

                # need to convert sf to scaled form
                for idx in cat_idx:
                    sf[idx] = float(cat_embed[idx][sf[idx]])

                sf_df = pd.DataFrame([sf], columns=cols)
                sf_df[numerical] = transformer.transform(sf_df[numerical])
                sf_sc = sf_df.to_numpy()[0]

                # Sparsity
                sparse = calculate_sparsity(sf_orig, query)
                sparsity.append(sparse)
                # SF-Query Distance
                sf_query.append(calculate_l2_dist(sf_sc, query_sc))
                # SF-NMOTB Distance
                if nmotb_idx is not None:
                    nmotb = x_train_sc[nmotb_idx]
                    # print(nmotb)
                    sf_nun.append(calculate_l2_dist(sf_sc, nmotb))
                    # print(calculate_l2_dist(x_sf, nmotb))
                    # SF-kNN(%)
                    num_k_sf_nh, num_k_sf_nmotb = calculate_k(nearest_hit_idx, nmotb_idx, nbrs, sf_sc)
                    sf_nh_knn.append(num_k_sf_nh)
                    sf_nun_knn.append(num_k_sf_nmotb)
                # Mahalanobis Distance
                maha_pos, maha_neg = calculate_mahalanobis(x_train_sc, y_train_sc, label, sf_sc)
                mahalanobis.append(maha_pos)
                # OOD Distance
                ood_dist, _ = calculate_ood(nbrs, y_train_sc, label, sf_sc)
                # print(ood_dist)
                ood_distance.append(ood_dist)
                # Trust score
                trust = trust_model.get_score(np.reshape(sf_sc, (1, -1)), np.reshape(label, (1, -1)))
                # print(trust[0][0])
                trust_score.append(trust[0][0])
                # Lipschitz constant
                lips = calculate_lipschitz(query_sc, sf_sc, label, d, model, transformer, cat_embed, cols, numerical, cat_idx, features_to_vary, num_perturbations)
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
        results = list(tqdm(pool.starmap(dice_loop, arguments), total=len(arguments), desc="Processing"))
        pbar.update(len(arguments))  # Manually update progress bar to ensure completion

    return results


if __name__ == '__main__':

    # Specifications
    data_desc = 'blood_alcohol'
    target = 'class'
    method = 'dice'

    # for lipschitz constant
    num_perturbations = 100
    # no. of k-folds
    n_folds = 5  # 5

    # Load data
    data, data_sc = load_custom_data(data_desc)
    label = data[target]

    # columns
    cols = list(data.columns[:-1])

    # get actionable features
    action_meta = actionability_constraints(data_desc)
    features_to_vary = []
    for col in cols:
        if action_meta[col]['actionable']:
            features_to_vary.append(col)


    numerical = ['units_consumed', 'weight', 'duration']
    categorical = ['gender', 'meal']

    cont_idx = [2, 3, 4]
    cat_idx = [0, 1]

    # scaler for nnumerical columns
    transformer = MinMaxScaler()
    df = data.copy()
    df = df.iloc[:, :-1]
    df[numerical] = transformer.fit_transform(df[numerical])

    # categorical embeddings
    cat_embed = categorical_embed[data_desc]

    # Generate fold indices for k-fold cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=None)

    arguments = []

    for train_index, test_index in kfold.split(data,label):
        arguments.append((data, label, data_sc, train_index, test_index, cols, numerical, categorical, cat_idx, cont_idx, target, transformer, cat_embed, features_to_vary, num_perturbations))

    parallel_results = run_parallel(arguments)

    # write the result dictionary to pickle file
    with open('./results/' + data_desc + '_' + method + '.pickle', 'wb') as f:
        pickle.dump(parallel_results, f)