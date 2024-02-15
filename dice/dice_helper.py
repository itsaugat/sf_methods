'''
Get DICE-based SF
'''

import dice_ml

def get_dice_sf(test, label, d, model, num, features_to_vary):

    # Using sklearn backend
    m = dice_ml.Model(model=model, backend="sklearn")
    # Using method=random for generating CFs
    exp = dice_ml.Dice(d, m, method="random")

    e1 = exp.generate_counterfactuals(test, total_CFs=num, desired_class=label, features_to_vary=features_to_vary)

    query, query_pred, sfs = e1.visualize_as_list()

    return query, query_pred, sfs