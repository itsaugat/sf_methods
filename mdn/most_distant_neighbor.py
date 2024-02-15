import pandas as pd

'''
get the categorical feature based on similar or closest value
'''


def get_categorical(val, cat_list):
    diff = lambda list_value: abs(list_value - val)
    closest_value = min(cat_list, key=diff)
    return closest_value


'''
get mdn for a particular feature and type 
'''


def get_most_distant_neighbor_v2(query, target_col, dataset_embed, df_sc, categorical_columns, categoric_attr_idx,
                                 cat_embed, feature):
    # get query label
    query_label = query[target_col]

    # remove target key from query
    query.popitem()

    # filter dataset based on the label of the query
    df_filter = dataset_embed[dataset_embed[target_col] == query_label]
    df_filter = df_filter.drop(columns=[target_col], axis=1)

    df_filter_sc = df_sc[df_sc[target_col] == query_label]
    df_filter_sc = df_filter_sc.drop(columns=[target_col], axis=1)

    # index of the key feature
    num_idx = df_filter.columns.get_loc(feature)

    # sort values in ascending order
    filter_df = df_filter.sort_values(by=[feature])
    filter_df_sc = df_filter_sc.sort_values(by=[feature])

    # convert df to dict to iterate faster
    filter_df_dict = filter_df.to_dict('records')

    # normalize categorical feature values in query
    for cat_feature in categorical_columns:
        cat_idx = df_filter.columns.get_loc(cat_feature)
        query[cat_feature] = float(cat_embed[cat_idx][query[cat_feature]])

    #print(query)

    # find higher_set and lower_set
    row_idx = 0
    threshold = None
    for row in filter_df_dict:
        #print(row[feature])
        #print(query[feature])
        if row[feature] >= query[feature]:
            threshold = row_idx
            break
        row_idx += 1

    # if threshold is still 'None', take the last index as threshold
    if threshold is None:
        threshold = row_idx

    ###################################### get higher set #######################################
    # (:) inclusive of the threshold index -> contains the query as well
    higher_set_sc = filter_df_sc[threshold:].to_numpy()
    higher_set = filter_df[threshold:].to_numpy()

    high_dict_list = {}
    for i in range(len(higher_set)):
        # create dictionary of feature values
        high_dict_list[i] = higher_set[i][num_idx]

    high_df = pd.DataFrame(high_dict_list.items(), columns=['index', feature])

    high_df_sort = high_df.sort_values([feature], ascending=[False])

    # remove unnecessary columns from df
    high_df_sort = high_df_sort.drop([feature], axis=1)

    high_mdn_df_list = high_df_sort.values.tolist()

    high_df_sc_list = []

    for item in high_mdn_df_list:
        high_df_sc_list.append(higher_set_sc[int(item[0])].tolist())
        item[0] = higher_set[int(item[0])].tolist()

    # get the categorical variables from the numerical embeddings
    for mdn in high_mdn_df_list:
        for idx in categoric_attr_idx:
            val = mdn[0][idx]
            cat_list = list(cat_embed[idx].values())
            # get the index of the closest categorical value
            cat_idx = list(cat_embed[idx].values()).index(get_categorical(val, cat_list))
            mdn[0][idx] = list(cat_embed[idx])[cat_idx]

    ###################################### get lower set #######################################
    # ':' is not inclusive of the threshold index, so adding '1'
    lower_set_sc = filter_df_sc[:(threshold + 1)].to_numpy()
    lower_set = filter_df[:(threshold + 1)].to_numpy()

    low_dict_list = {}
    for i in range(len(lower_set)):
        # create dictionary of feature values
        low_dict_list[i] = lower_set[i][num_idx]

    low_df = pd.DataFrame(low_dict_list.items(), columns=['index', feature])

    low_df_sort = low_df.sort_values([feature], ascending=[True])

    # remove other columns from df
    low_df_sort = low_df_sort.drop([feature], axis=1)

    low_mdn_df_list = low_df_sort.values.tolist()

    low_df_sc_list = []

    for item in low_mdn_df_list:
        low_df_sc_list.append(lower_set_sc[int(item[0])].tolist())
        item[0] = lower_set[int(item[0])].tolist()

    # get the categorical variables from the numerical embeddings
    for mdn in low_mdn_df_list:
        for idx in categoric_attr_idx:
            val = mdn[0][idx]
            cat_list = list(cat_embed[idx].values())
            # get the index of the closest categorical value
            cat_idx = list(cat_embed[idx].values()).index(get_categorical(val, cat_list))
            mdn[0][idx] = list(cat_embed[idx])[cat_idx]

    #######################################################################################

    # assign index to each item in both list
    high_mdn_df_list_idx = [(idx, item) for idx, item in enumerate(high_mdn_df_list)]

    low_mdn_df_list_idx = [(idx, item) for idx, item in enumerate(low_mdn_df_list, start=len(high_mdn_df_list))]

    return high_mdn_df_list_idx, high_df_sc_list, low_mdn_df_list_idx, low_df_sc_list
