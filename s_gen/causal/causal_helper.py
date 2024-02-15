import utils
import data_utils
import trainers
import numpy as np
import torch
import recourse

def find_recourse_mlp(model, trainer, scmm, X_explain, constraints):

    hyperparams = utils.get_recourse_hyperparams(trainer)

    explain = recourse.DifferentiableRecourseSGEN(model, hyperparams)

    x, counterfacs = recourse.causal_recourse(X_explain, explain, constraints, scm=scmm)

    return x, counterfacs

def sgen_causal(X_train, X_test, Y_train, Y_test, constraints, dataset, model_type, trainer, random_seed, lambd, N_explain):

    # Load the relevant dataset
    #X, Y, constraints = data_utils.process_data(dataset)
    #X_train, Y_train, X_test, Y_test = data_utils.train_test_split(X, Y)

    # Load the relevant model
    model_dir = utils.get_model_save_dir(dataset, trainer, model_type, random_seed, lambd) + '.pth'
    model = trainers.MLP
    model = model(X_train.shape[-1], actionable_features=constraints['actionable'], actionable_mask=trainer == 'AF')
    model.load_state_dict(torch.load(model_dir))
    model.set_max_mcc_threshold(X_train, Y_train)

    # Load the SCM
    scmm = utils.get_scm(model_type, dataset)

    # get the test ids where the model predicts and the true label is also 1 (i.e positive outcome)
    id_neg = (model.predict(X_test) == 1) & (Y_test == 1)
    X_neg = X_test[id_neg]
    #Y_neg = Y_test[id_neg]
    N_Explain = min(N_explain, len(X_neg))

    # Different seed here
    id_explain = np.random.choice(np.arange(X_neg.shape[0]), size=N_Explain, replace=False)
    X_explain = X_neg[id_explain]
    #labels = Y_neg[id_explain]

    # Find recourse
    query, sfs = find_recourse_mlp(model, trainer, scmm, X_explain, constraints)

    return query, sfs, model, scmm


'''
define sgen_causal function to compute lipschitz value
'''
def sgen_causal_lips(X_explain, constraints, model, scmm, trainer):

    # Find recourse
    query, sfs = find_recourse_mlp(model, trainer, scmm, X_explain, constraints)

    return query, sfs