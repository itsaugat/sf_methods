import numpy as np
import pandas as pd

reject_thresholds = [0.001, 0.005, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 1, 2, 3, 4, 5, 7, 10, 100, 1000]

knn_parameters = {'n_neighbors': [3, 5, 7, 10, 15]}
random_forest_parameters = {
    'n_estimators': [20, 50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy'],
    'random_state': [444]
}

non_zero_threshold = 1e-5
non_zero_threshold_sparsity = 1e-5


def load_custom_data(data_desc, target):
    data = pd.read_csv('../datasets/'+data_desc+'_sc.csv')
    y = np.array(data[target])
    df_sub = data.loc[:, data.columns != target]
    x = np.array(df_sub)

    return x, y
