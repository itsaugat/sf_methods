import pickle
from statistics import mean
import collections
from sklearn.preprocessing import MinMaxScaler


data = 'blood_alcohol'
method = 'dser'

# function to scale the values
def scale_values(input_list, feature_range=(0, 1)):
    # Reshape the input list to a 2D array for MinMaxScaler
    values_2d = [[value] for value in input_list]

    # Initialize the MinMaxScaler with the specified feature range
    scaler = MinMaxScaler(feature_range=feature_range)

    # Fit and transform the data using the scaler
    scaled_values = scaler.fit_transform(values_2d)

    # Extract the scaled values from the 2D array and return as a list
    scaled_list = [scaled_value[0] for scaled_value in scaled_values]

    return scaled_list


with open(data + '_' + method + '.pickle', 'rb') as f:
    results = pickle.load(f)

print(len(results))

# print(results[0])

sf_query = []
sf_nh_knn = []
sf_nun_knn = []
sf_nun = []
mahalanobis = []
sparsity = []
ood_distance = []
trust_score = []
lipschitz = []

for item in results:
    for val in item:
        sf_query.append(val['sf_query'])
        sf_nun.append(val['sf_nun'])
        mahalanobis.append(val['mahalanobis'])
        sparsity.append(val['sparsity'])
        ood_distance.append(val['ood_distance'])
        trust_score.append(val['trust_score'])
        lipschitz.append(val['lipschitz'])

sf_query = [x for xs in sf_query for x in xs]
sf_nun = [x for xs in sf_nun for x in xs]
mahalanobis = [x for xs in mahalanobis for x in xs]
sparsity = [x for xs in sparsity for x in xs]
ood_distance = [x for xs in ood_distance for x in xs]
trust_score = [x for xs in trust_score for x in xs]
trust_score_sc = scale_values(trust_score)
lipschitz = [x for xs in lipschitz for x in xs]
lipschitz_sc = scale_values(lipschitz)

metric_zip = zip(sf_query, sf_nun, sparsity, ood_distance, trust_score_sc, lipschitz_sc)

unitary = []

# create unitary metric by combining the metrics in a relevant way
for sf_q, sf_n, sp, ood, ts, lips in metric_zip:

    if (sp == 0):
        val = sf_q + sf_n + 0 + ts - ood - lips
    else:
        val = sf_q + sf_n + (1 / sp) + ts - ood - lips

    unitary.append(val)

unitary = mean(unitary)
print('unitary :'+ str(unitary))

sf_query = mean(sf_query)
print('sf_query :'+ str(sf_query))

sf_nun = mean(sf_nun)
print('sf_nun :'+ str(sf_nun))

mahalanobis = mean(mahalanobis)
print('mahalanobis :'+ str(mahalanobis))

ood_distance = mean(ood_distance)
print('ood_distance :'+ str(ood_distance))

trust_score = mean(trust_score)
print('trust_score :'+ str(trust_score))

trust_score_sc = mean(trust_score_sc)
#trust_score_sc

lipschitz = mean(lipschitz)
print('lipschitz :'+ str(lipschitz))

lipschitz_sc = mean(lipschitz_sc)
#lipschitz_sc

sparse = collections.Counter(sparsity)
#sparse

new_dict = {}
three = 0
one = 0
two = 0
for key in sparse:
    if key >= 3:
        three += sparse[key]
    elif key == 1:
        one = sparse[key]
    elif key == 2:
        two = sparse[key]

new_dict['1'] = one
new_dict['2'] = two
new_dict['3+'] = three

#new_dict

new_dict_p = {}

new_dict_p['1'] = (new_dict['1'] / (new_dict['1'] + new_dict['2'] + new_dict['3+'])) * 100
new_dict_p['2'] = (new_dict['2'] / (new_dict['1'] + new_dict['2'] + new_dict['3+'])) * 100
new_dict_p['3+'] = (new_dict['3+'] / (new_dict['1'] + new_dict['2'] + new_dict['3+'])) * 100

print('sparse : '+str(new_dict_p))