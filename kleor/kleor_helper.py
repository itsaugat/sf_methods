import numpy as np

'''
Calculate L2 distance (Euclidean)
'''
def calculate_l2_distance(v1, v2):
    return np.linalg.norm(v1 - v2, 2)


'''
KLEOR-Sim-Miss -> A case that 
1. has the same class as Q;
2. is most similar to NM(NMOTB).
'''
def get_kleor_sim_miss(pred_train, nmotb):
    # filter instances with same class as pred
    #pred_train = train[np.in1d(train[:, -1], pred)]

    # compute distance of each instance with nmotb
    idx_dist_list = []
    for i in range(len(pred_train)):
        dist = calculate_l2_distance(pred_train[i][:(len(pred_train[i]) - 1)], nmotb)
        idx_dist_list.append((i, dist))

    # sort the tuples in list in ascending order of the distance
    idx_dist_list_sorted = sorted(idx_dist_list, key=lambda t: t[1])

    # first instance is the kleor_sim_miss -> closest distance to nmotb
    kleor_sim_miss_idx = idx_dist_list_sorted[0][0]
    kleor_sim_miss = pred_train[kleor_sim_miss_idx]

    kleor_sim_miss = kleor_sim_miss[:-1]

    return kleor_sim_miss, kleor_sim_miss_idx


'''
KLEOR-Global-Sim -> A case that 
1. has the same class as Q;
2. is most similar to NM(NMOTB).
3. is located, according to similarity, between Q and NM(NMOTB): Sim(Q, EC)>Sim(Q, NMOTB)
'''
def get_kleor_global_sim(pred_train, nmotb, query):
    # filter instances with same class as pred
    #pred_train = train[np.in1d(train[:, -1], pred)]

    # compute the distance between query and nmotb
    dist_q_nmotb = calculate_l2_distance(query[0], nmotb)

    idx_dist_list = []
    # iterate through each instance
    for i in range(len(pred_train)):
        # compute the distance between the instance and query
        dist_q_n = calculate_l2_distance(pred_train[i][:(len(pred_train[i]) - 1)], query[0])
        # check if dist(query, instance) <= dist(query, nmotb) [following the criteria]
        if dist_q_n <= dist_q_nmotb:
            # if true, then compute the distance between the instance and nmotb
            dist_n_nmotb = calculate_l2_distance(pred_train[i][:(len(pred_train[i]) - 1)], nmotb)
            idx_dist_list.append((i, dist_n_nmotb))

    # sort the tuple in ascending order of the distance
    idx_dist_list_sorted = sorted(idx_dist_list, key=lambda t: t[1])

    # if the list is empty then return None
    if len(idx_dist_list_sorted) == 0:
        kleor_global_sim = None
        kleor_global_sim_idx = None
    else:
        # first instance is the kleor_global_sim -> closest to nmotb
        kleor_global_sim_idx = idx_dist_list_sorted[0][0]
        kleor_global_sim = pred_train[kleor_global_sim_idx]
        kleor_global_sim = kleor_global_sim[:-1]


    return kleor_global_sim, kleor_global_sim_idx



'''
Get the number of attributes which matches the condition 2 of KLEOR-Attr-Sim
'''
def get_attribute_count(query, neighbor, nmotb, numerical_attr_index, cat_attr_index):
    count = 0

    ################ numeric attribute count ##########################
    # check if the condition matches
    for attr in numerical_attr_index:
        if (query[attr] < nmotb[attr]):
            if (query[attr] < neighbor[attr]) and (neighbor[attr] < nmotb[attr]):
                count += 1
        elif (query[attr] > nmotb[attr]):
            if (nmotb[attr] < neighbor[attr]) and (neighbor[attr] < query[attr]):
                count += 1

    ################## categorical attribute count #####################
    # if the categorical values are same then the similarity is 1 or else 0
    for att in cat_attr_index:
        query_a = query[att]
        neighbor_a = neighbor[att]
        nmotb_a = nmotb[att]
        if (query_a == neighbor_a) and (query_a != nmotb_a):
            count += 1

    return count


'''
KLEOR-Attr-Sim -> A case that 
1. has the same class as Q;
2. is most similar to NM(NMOTB).
3. has the most attributes a for which: sim(Q.a, EC.a) > sim(Q.a, NMOTB.a)
'''
def get_kleor_attr_sim(pred_train, nmotb, query, num_idx, cat_idx):
    # filter instances with same class as pred
    #pred_train = train[np.in1d(train[:, -1], pred)]

    # print(pred_train[np.in1d(train[:, -1], 1)])

    # compute distance with nmotb and get similar attribute count for each instance
    idx_dist_count_list = []
    for i in range(len(pred_train)):
        dist = calculate_l2_distance(pred_train[i][:(len(pred_train[i]) - 1)], nmotb)
        count = get_attribute_count(query[0], pred_train[i], nmotb, num_idx, cat_idx)
        idx_dist_count_list.append((i, dist, count))

    # sort the tuples in list in ascending order of the distance
    idx_dist_count_list_sorted = sorted(idx_dist_count_list, key=lambda t: t[1])

    # sort the tuples in descending order of count
    idx_dist_count_list_sorted = sorted(idx_dist_count_list_sorted, key=lambda t: t[2], reverse=True)

    # first instance is the kleor_sim_miss -> closest distance to nmotb
    kleor_attr_sim_idx = idx_dist_count_list_sorted[0][0]
    kleor_attr_sim = pred_train[idx_dist_count_list_sorted[0][0]]

    kleor_attr_sim = kleor_attr_sim[:-1]

    return kleor_attr_sim, kleor_attr_sim_idx