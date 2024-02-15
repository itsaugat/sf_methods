import pandas as pd
import numpy as np

def load_custom_data(data_desc, target):
    data = pd.read_csv('../datasets/'+data_desc+'_sc.csv')
    y = np.array(data[target])
    df_sub = data.loc[:, data.columns != target]
    x = np.array(df_sub)

    return x, y