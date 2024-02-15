import numpy as np
import scipy
from copy import deepcopy


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_prob_cat(cf_df, x, continuous_feature_names, categorical_feature_names, df_train, enc):
    cat_probabilities = list()
    expected_values = list()
    index_current = len(continuous_feature_names)

    for i, cat in enumerate(categorical_feature_names):
        temp0 = df_train[df_train.preds == 0][cat]
        temp1 = df_train[df_train.preds == 1][cat]

        # Expected value
        probs = list()
        for cat2 in enc.categories_[i]:
            probs.append((temp0 == cat2).sum() / ((temp1 == cat2).sum() + 0.0001))

        probs = np.array(probs) / sum(probs)

        expected_values.append(np.argmax(np.array(probs)))

        # Feature prob
        feature_rep = x[index_current: index_current + enc.categories_[i].shape[0]]
        feature_prob = (feature_rep * probs).sum()
        cat_probabilities.append(feature_prob)
        actual_feature_value_idx = np.argmax(np.array(feature_rep))

        index_current += enc.categories_[i].shape[0]

    return cat_probabilities, expected_values


def get_prob_cont(x, continuous_feature_names, X_train, train_preds):
    """
    		Returns probability of values from normal class, expected value
    		"""

    cont_probs = list()
    cont_expected = list()

    for i, cat in enumerate(continuous_feature_names):

        # pick continuous feature (i.e., i), and positive prediction (i.e., 1)
        temp = X_train.T[i][train_preds == 1]
        rv = scipy.stats.gamma

        try:
            params = rv.fit(temp)
        except:
            params = (0.5, 0.5, 0.5)

        prob = rv.cdf(x[i], *params)
        if prob < 0.5:
            cont_probs.append(prob)

            # project mean to nearest recorded value (to allow ordinal variables to work)
            mean = find_nearest(temp, rv.mean(*params))
            cont_expected.append(mean)
        else:
            cont_probs.append(1 - prob)

            # project mean to nearest recorded value (to allow ordinal variables to work)
            mean = find_nearest(temp, rv.mean(*params))
            cont_expected.append(mean)

    return cont_probs, cont_expected


def get_feature_probabilities(cf_df, x, test_q_dict, test_q, X_train, train_preds, continuous_feature_names, categorical_feature_names, df_train, enc):
    cont_probs, cont_expected = get_prob_cont(test_q, continuous_feature_names, X_train, train_preds)
    cat_probs, expected_cat = get_prob_cat(cf_df, test_q, continuous_feature_names, categorical_feature_names, df_train, enc)
    return cont_probs, cont_expected, cat_probs, expected_cat


def flip_category(x, categorical_feature_names, cat_idxs, cat_name='menopaus', change_to=1):
    for i, cat in enumerate(categorical_feature_names):
        if cat == cat_name:
            feature_rep = deepcopy(x[cat_idxs[i][0]: cat_idxs[i][1]])
            feature_rep *= 0.
            feature_rep[int(change_to) - 1] = 1.
            x[cat_idxs[i][0]: cat_idxs[i][1]] = feature_rep
    return x


def clip_expected_values(test_q, expected_values, feature_names, action_meta, continuous_feature_names, test_q_dict):
    # iterate each actionable feature
    for idx, f in enumerate(feature_names):
        if action_meta[f]['actionable']:

            if f in continuous_feature_names:
                current_value = test_q[idx]
            else:
                current_value = test_q_dict[f]

            # current_value = df_test.iloc[test_idx].values[idx]
            e_value = expected_values[idx]

            # if expected value is lower than actionable range and you can't go down
            if e_value < current_value and not action_meta[f]['can_decrease']:
                expected_values[idx] = current_value

            # opposite
            if e_value > current_value and not action_meta[f]['can_increase']:
                expected_values[idx] = current_value

    return expected_values


def get_counterfactual(test_q, X_train, cf_df, continuous_feature_names, categorical_feature_names, clf, action_meta, cat_idxs, test_q_dict, df_train, enc, train_preds):
    # Totally normalized (0-1)
    #x = deepcopy(X_test[test_idx])
    #original_query = deepcopy(X_test[test_idx])

    # Get feature probabilities
    cont_probs, expected_conts, cat_probs, expected_cat = get_feature_probabilities(cf_df, test_q, test_q_dict, test_q, X_train, train_preds, continuous_feature_names, categorical_feature_names, df_train, enc)

    feature_probs = np.array(cont_probs + cat_probs)
    feature_expected = np.array(expected_conts + expected_cat)
    features = continuous_feature_names + categorical_feature_names
    feature_expected = clip_expected_values(test_q, feature_expected, features, action_meta, continuous_feature_names, test_q_dict)
    feature_order = np.argsort(feature_probs)
    original_prob = clf.predict_proba(test_q.reshape(1, -1))[0][1]
    current_prob = clf.predict_proba(test_q.reshape(1, -1))[0][1]
    original_pred = clf.predict(test_q.reshape(1, -1)).item()

    # Flip the excpetional feature(s) one at a time:
    #print(feature_order)
    for i in range(len(feature_order)):

        if action_meta[features[feature_order[i]]]['actionable']:

            temp = deepcopy(test_q)
            tempx = deepcopy(test_q)

            if features[feature_order[i]] in continuous_feature_names:
                temp[feature_order[i]] = expected_conts[feature_order[i]]
            else:
                temp = flip_category(temp, categorical_feature_names, cat_idxs, cat_name=features[feature_order[i]],
                                     change_to=feature_expected[feature_order[i]])

            new_prob = clf.predict_proba(temp.reshape(1, -1))[0][1]
            new_pred = clf.predict(temp.reshape(1, -1)).item()

            if new_pred != original_pred:
                return temp, original_prob, current_prob

            if new_prob < current_prob:
                x = temp
                current_prob = new_prob

    return temp, original_prob, current_prob


def generate_cat_idxs(continuous_feature_names, enc):
    """
    Get indexes for all categorical features that are one hot encoded
    """

    cat_idxs = list()
    start_idx = len(continuous_feature_names)
    for cat in enc.categories_:
        cat_idxs.append([start_idx, start_idx + cat.shape[0]])
        start_idx = start_idx + cat.shape[0]
    return cat_idxs