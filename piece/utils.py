import pandas as pd

def get_dataset(data, cont_feats, cat_feats):
    """
    Assumes target class is binary 0 1, and that 1 is the semi-factual class
    """

    df = pd.read_csv('../datasets/sf/'+data+'.csv')
    df_sc = pd.read_csv('../datasets/sf/'+data+'_sc.csv')

    for cat in cat_feats:
        df[cat] = df[cat].astype('int')
    for cat in cont_feats:
        df[cat] = df[cat].astype('float')

    return df, df_sc