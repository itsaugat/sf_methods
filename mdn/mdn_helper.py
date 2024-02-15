import pandas as pd
import numpy as np
from most_distant_neighbor import get_most_distant_neighbor_v2

'''
compute tolerance level to check equality of numerical feature values based on std 
'''
def num_tolerance(std):
    level = 0.2
    tol = level * std
    return tol



'''
compute number of common numerical features
'''
def count_num_com(num_idx, mdn, query, key_feat_idx, std_dict):
    count = 0
    # ignore the key feature in consideration
    num_idx_diff = set(num_idx) - set([key_feat_idx])
    for idx in list(num_idx_diff):
        # get std of the feature
        std = std_dict[idx]
        # get tolerance level for the feature
        tol = num_tolerance(std)
        # check if the query value lies within the range of tolerance
        if mdn[idx]+tol > mdn[idx]:
            if float(mdn[idx]-tol) <= float(query[idx]) <= float(mdn[idx]+tol):
                count += 1
        elif mdn[idx]+tol < mdn[idx]:
            if float(mdn[idx]+tol) <= float(query[idx]) <= float(mdn[idx]-tol):
                count += 1
    return count



'''
compute number of common categorical features
'''
def count_cat_com(cat_idx, mdn, query, key_feat_idx):
    count = 0
    # ignore the key feature in consideration
    cat_idx_diff = set(cat_idx) - set([key_feat_idx])
    for idx in list(cat_idx_diff):
        if mdn[idx] == query[idx]:
            count += 1

    return count

def compute_sf_value_continuous_v2(tot_com_count, feat_diff, max_feat_diff, total_feats):
    m1 = (tot_com_count / total_feats)
    m2 = (feat_diff / max_feat_diff)

    val = (1 / (total_feats - tot_com_count)) * (m1 + m2)

    return val


def compute_sf_value_categorical_v2(tot_com_count, feat_diff, max_feat_diff, total_feats):
    # compute the overall value
    m1 = (tot_com_count / total_feats)
    m2 = (feat_diff / max_feat_diff)
    val = (1/(total_feats - tot_com_count))*(m1 + m2)

    return val


def get_mdn_sf(query, label, target, features, num_idx, cat_idx, cat_cols, cat_embed, df_embed, df_sc, std_dict):
    # create dataframe to store the mdn's for each feature
    mdn_df = pd.DataFrame(columns=['mdn', 'mdn_sc', 'sf_val', 'sim', 'feature'])

    for key_feature in features:

        # convert into dictionary
        row = {}
        for a, b in zip(features, query):
            row[a] = b

        # add target column to query at the end
        row[target] = label

        # make a copy of the query to use it later
        # row_cp/row is query in dictionary form in un-normalized form
        row_cp = row.copy()

        key_feat_idx = df_embed.columns.get_loc(key_feature)
        if key_feat_idx in num_idx:
            key_feature_type = 'continuous'
        else:
            key_feature_type = 'categorical'

        # get the list of mdn's
        high_mdn_list, high_mdn_list_sc, low_mdn_list, low_mdn_list_sc = get_most_distant_neighbor_v2(row_cp,
                                                                                                      target,
                                                                                                      df_embed,
                                                                                                      df_sc,
                                                                                                      cat_cols,
                                                                                                      cat_idx,
                                                                                                      cat_embed,
                                                                                                      key_feature)

        # remove the target column/value from the query_dict since it is at the end
        row.popitem()

        # get the query values as list
        row = list(row.values())

        # initialize high/low set lists
        high_sf_list = []
        high_sf_sim_list = []
        low_sf_list = []
        low_sf_sim_list = []

        total_feats = len(num_idx) + len(cat_idx)

        ###################################### High MDN #####################################
        for i in range(len(high_mdn_list)):
            mdn = high_mdn_list[i][1][0]
            mdn_sc = high_mdn_list_sc[i]

            cat_count = count_cat_com(cat_idx, mdn, row, key_feat_idx)
            num_count = count_num_com(num_idx, mdn, row, key_feat_idx, std_dict)
            tot_com_count = num_count + cat_count
            toal_feat_diff = total_feats - tot_com_count

            if key_feature_type == 'continuous':
                max_feat_diff = abs(high_mdn_list[0][1][0][key_feat_idx] - query[key_feat_idx])
                if (max_feat_diff == 0):
                    max_feat_diff = 1
                    feat_diff = 0
                else:
                    feat_diff = abs(mdn[key_feat_idx] - query[key_feat_idx])

                sf_val = compute_sf_value_continuous_v2(tot_com_count, feat_diff, max_feat_diff, total_feats)

            elif key_feature_type == 'categorical':
                # encode max mdn
                max_mdn_cat_encode = high_mdn_list[0][1][0].copy()
                for idx in cat_idx:
                    max_mdn_cat_encode[idx] = float(cat_embed[idx][max_mdn_cat_encode[idx]])

                # encode categorical features of mdn
                mdn_cat_encode = mdn.copy()
                for idx in cat_idx:
                    mdn_cat_encode[idx] = float(cat_embed[idx][mdn_cat_encode[idx]])

                # encode query
                query_cat_encode = query.copy()
                for idx in cat_idx:
                    query_cat_encode[idx] = float(cat_embed[idx][query_cat_encode[idx]])

                max_feat_diff = abs(max_mdn_cat_encode[key_feat_idx] - query_cat_encode[key_feat_idx])

                if (max_feat_diff == 0):
                    max_feat_diff = 1
                    feat_diff = 0
                else:
                    feat_diff = abs(mdn_cat_encode[key_feat_idx] - query_cat_encode[key_feat_idx])

                sf_val = compute_sf_value_categorical_v2(tot_com_count, feat_diff, max_feat_diff, total_feats)

            high_sf_list.append((i, mdn, sf_val, mdn_sc))
            high_sf_sim_list.append((i, toal_feat_diff))

        ######################################### Low MDN ########################################
        for i in range(len(low_mdn_list)):
            mdn = low_mdn_list[i][1][0]
            mdn_sc = low_mdn_list_sc[i]

            cat_count = count_cat_com(cat_idx, mdn, row, key_feat_idx)
            num_count = count_num_com(num_idx, mdn, row, key_feat_idx, std_dict)
            tot_com_count = num_count + cat_count
            toal_feat_diff = total_feats - tot_com_count

            if key_feature_type == 'continuous':
                max_feat_diff = abs(low_mdn_list[0][1][0][key_feat_idx] - query[key_feat_idx])

                if (max_feat_diff == 0):
                    max_feat_diff = 1
                    feat_diff = 0
                else:
                    feat_diff = abs(mdn[key_feat_idx] - query[key_feat_idx])

                sf_val = compute_sf_value_continuous_v2(tot_com_count, feat_diff, max_feat_diff, total_feats)

            elif key_feature_type == 'categorical':
                # encode max mdn
                max_mdn_cat_encode = low_mdn_list[0][1][0].copy()
                for idx in cat_idx:
                    max_mdn_cat_encode[idx] = float(cat_embed[idx][max_mdn_cat_encode[idx]])

                # encode categorical features of mdn
                mdn_cat_encode = mdn.copy()
                for idx in cat_idx:
                    mdn_cat_encode[idx] = float(cat_embed[idx][mdn_cat_encode[idx]])

                # encode query
                query_cat_encode = query.copy()
                for idx in cat_idx:
                    query_cat_encode[idx] = float(cat_embed[idx][query_cat_encode[idx]])

                max_feat_diff = abs(max_mdn_cat_encode[key_feat_idx] - query_cat_encode[key_feat_idx])

                if (max_feat_diff == 0):
                    max_feat_diff = 1
                    feat_diff = 0
                else:
                    feat_diff = abs(mdn_cat_encode[key_feat_idx] - query_cat_encode[key_feat_idx])

                sf_val = compute_sf_value_categorical_v2(tot_com_count, feat_diff, max_feat_diff, total_feats)

            low_sf_list.append((i, mdn, sf_val, mdn_sc))
            low_sf_sim_list.append((i, toal_feat_diff))

        ########################### Get best MDN from High/Low list ###########################

        high_sim_dict = dict(high_sf_sim_list)
        low_sim_dict = dict(low_sf_sim_list)

        if len(low_sf_list) != 0 and len(high_sf_list) != 0:

            # get the highest sfs score from both list
            high_sf_val = max(high_sf_list, key=lambda item: item[2])
            low_sf_val = max(low_sf_list, key=lambda item: item[2])

            # select sf with highest sfs score (from high and low set)
            if high_sf_val[2] > low_sf_val[2]:
                sf_idx = high_sf_val[0]
                sf = high_sf_val[1]
                sf_sc = high_sf_val[3]
                sf_val = high_sf_val[2]
                sim = high_sim_dict[sf_idx]

            elif low_sf_val[2] > high_sf_val[2]:
                sf_idx = low_sf_val[0]
                sf = low_sf_val[1]
                sf_sc = low_sf_val[3]
                sf_val = low_sf_val[2]
                sim = low_sim_dict[sf_idx]

            # if the sf val is same for high and low then select the one with the highest similarity
            elif high_sf_val[2] == low_sf_val[2]:
                if high_sim_dict[high_sf_val[0]] >= low_sim_dict[low_sf_val[0]]:
                    sf = high_sf_val[1]
                    sf_sc = high_sf_val[3]
                    sf_val = high_sf_val[2]
                    sim = high_sim_dict[high_sf_val[0]]

                elif low_sim_dict[low_sf_val[0]] > high_sim_dict[high_sf_val[0]]:
                    sf = low_sf_val[1]
                    sf_sc = low_sf_val[3]
                    sf_val = low_sf_val[2]
                    sim = low_sim_dict[low_sf_val[0]]

            feat_sf = np.array(sf)
            feat_sf_sc = np.array(sf_sc)

        elif len(low_sf_list) == 0:
            high_sf_val = max(high_sf_list, key=lambda item: item[2])
            sf_idx = high_sf_val[0]
            sf = high_sf_val[1]
            sf_sc = high_sf_val[3]
            sf_val = high_sf_val[2]
            sim = high_sim_dict[sf_idx]
            feat_sf = np.array(sf)
            feat_sf_sc = np.array(sf_sc)

        elif len(high_sf_list) == 0:
            low_sf_val = max(low_sf_list, key=lambda item: item[2])
            sf_idx = low_sf_val[0]
            sf = low_sf_val[1]
            sf_sc = low_sf_val[3]
            sf_val = low_sf_val[2]
            sim = low_sim_dict[sf_idx]
            feat_sf = np.array(sf)
            feat_sf_sc = np.array(sf_sc)

        # feat_sf is the best mdn(sf) of the featuure in numpy array format in un-normalized form
        # feat_sf_sc is the best mdn(sf) of the featuure in numpy array format in normalized form

        # insert the best mdn of each feature to the dataframe
        new_df = pd.DataFrame([[feat_sf, feat_sf_sc, sf_val, sim, key_feature]],
                              columns=['mdn', 'mdn_sc', 'sf_val', 'sim', 'feature'])
        mdn_df = pd.concat([mdn_df, new_df])

    # Get the best of the best MDN from all the features
    mdn_df = mdn_df.sort_values(by=['sf_val', 'sim'], ascending=[False, False])
    temp = mdn_df.values.tolist()
    # mdn_sf is in list un-normalized form
    mdn_sf = temp[0][0]
    mdn_sf_sc = temp[0][1]

    return mdn_sf_sc, mdn_sf