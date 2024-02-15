import numpy as np
from sklearn.linear_model import LogisticRegression

'''
Get nugent-based semi-factual
'''


def get_nugent_sf(indices, distances, x_train, y_train, min_num_class, pred, query):
    # create local case base
    num_class_0 = 0
    num_class_1 = 0
    last_sim = None
    local_case_x = []
    local_case_y = []
    query_class_list = []

    indices = indices[0]
    distances = distances[0]

    # iterate through the neighbors
    for i, (idx, dist) in enumerate(zip(indices, distances)):
        # get the class of the neighbor
        idx_class = y_train[idx]

        if (num_class_0 > min_num_class) and (num_class_1 > min_num_class) and (dist != last_sim):
            break
        else:
            local_case_x.append(x_train[idx])
            local_case_y.append(idx_class)

            if idx_class == pred:
                query_class_list.append((x_train[idx], i, idx))

            if idx_class == 0:
                num_class_0 += 1
            elif idx_class == 1:
                num_class_1 += 1

            last_sim = dist

    # extract the idx not present in list
    idx_present = [x[2] for x in query_class_list]
    idx_absent = list(set(indices) - set(idx_present))
    # add rest of query_class_instances to query_class_lst
    for idx in idx_absent:
        # check if same class as query (pred)
        if y_train[idx] == pred:
            query_class_list.append((x_train[idx], None, idx))

    # fit logistic regression model on local case base
    lr = LogisticRegression(random_state=0).fit(local_case_x, local_case_y)

    # get the probability based on logistic regression for being in the same predicted class
    query_prob = lr.predict_proba(query)[0][list(lr.classes_).index(pred)]

    sf = None

    '''filter only query whose probability (predicted by logistic regression model) 
    of being in the actual class is >=0.5'''
    if query_prob >= 0.5:

        '''
        Each case in the local case base (alternatively, the k-neighbors) is considered as the candidate explanation case.
        Get the case which has the least probability of being in the same class as the query. 
        '''
        prob_list = []
        for idx in range(len(local_case_x)):
            label = local_case_y[idx]
            probabs = lr.predict_proba(np.reshape(local_case_x[idx], (1, -1)))
            class_prob = probabs[0][list(lr.classes_).index(pred)]
            if class_prob >= 0.5 and label == pred:
                prob_list.append((idx, label, class_prob))

        # sort in ascending order of probability (lowest to highest)
        sorted_prob_list = sorted(prob_list, key=lambda t: t[2])

        # select the first instance as the semi-factual
        sf = local_case_x[sorted_prob_list[0][0]]

    return sf