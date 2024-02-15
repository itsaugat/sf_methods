import pandas as pd

POP_SIZE = 12
MAX_GENERATIONS = 25
LAMBDA1 = 30  # robustness e-neighborhood
LAMBDA2 = 10  # robustness instance
GAMMA = 1  # diversity
POSITIVE_CLASS = 1  # the semi-factual positive class
CONT_PERTURB_STD = 0.05 # perturb continuous features by 5% STD
MUTATION_RATE = 0.05
ELITIST = 4  # how many of the "best" to save
MAX_MC = 100

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