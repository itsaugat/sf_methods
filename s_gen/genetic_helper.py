import numpy as np
from copy import deepcopy
from random import gauss
from scipy.spatial import distance_matrix
from utils import *

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


def get_actionable_feature_idxs(action_meta, continuous_features, categorical_features):
    """
    sample a random actionable feature index
    """

    feature_names = continuous_features.columns.tolist() + categorical_features.columns.tolist()
    actionable_idxs = list()

    for i, f in enumerate(feature_names):
        if action_meta[f]['actionable']:
            actionable_idxs.append([i, action_meta[f]['can_increase'], action_meta[f]['can_decrease']])

    return actionable_idxs


def generate_category(x, x_prime, idx, cat_idxs, categorical_features, action_meta, replace=True):
    """
    Randomly generate a value for a OHE categorical feature using actionability constraints
    replace: this gives the option if the generation should generate the original
    value for the feature that is present in x, or if it should only generate
    different x_primes with different values for the feature

    """

    original_rep = x[cat_idxs[idx][0]: cat_idxs[idx][1]]  # To constrain with initial datapoint
    new_rep = x_prime[cat_idxs[idx][0]: cat_idxs[idx][1]]  # to make sure we modify based on new datapoint

    cat_name = categorical_features.columns[idx]

    if replace:  # just for population initialisation

        # If you can generate new feature anywhere
        if action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
            new = np.eye(len(original_rep))[np.random.choice(len(original_rep))]

        # if you can only increase
        elif action_meta[cat_name]['can_increase'] and not action_meta[cat_name]['can_decrease']:
            try:
                # To account for when it's the last value in the scale of categories
                new = np.eye(len(original_rep) - (np.argmax(original_rep)))[
                    np.random.choice(len(original_rep) - (np.argmax(original_rep)))]
                new = np.append(np.zeros((np.argmax(original_rep))), new)
            except:
                new = new_rep

        # If you can only decrease
        elif not action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
            try:
                # To account for when it's the first value in the scale of categories
                new = np.eye(np.argmax(original_rep) + 1)[np.random.choice(np.argmax(original_rep) + 1)]
                new = np.append(new, np.zeros((len(original_rep) - np.argmax(original_rep)) - 1))
            except:
                new = new_rep

        else:
            new = new_rep

    else:  # For MC sampling, and mutation

        # If you can generate new feature anywhere
        if action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
            new = np.eye(len(original_rep) - 1)[np.random.choice(len(original_rep) - 1)]
            new = np.insert(new, np.argmax(new_rep), 0)

        # if you can only increase
        elif action_meta[cat_name]['can_increase'] and not action_meta[cat_name]['can_decrease']:
            try:
                # To account for when it's the last value in the scale of categories
                new = np.eye(len(original_rep) - np.argmax(original_rep) - 1)[
                    np.random.choice(len(original_rep) - np.argmax(original_rep) - 1)]
                new = np.insert(new, np.argmax(new_rep) - (np.argmax(original_rep)), 0)
                new = np.concatenate(
                    (np.zeros((len(original_rep) - (len(original_rep) - np.argmax(original_rep)))), new))
            except:
                new = new_rep

        # If you can only decrease
        elif not action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:

            try:  # To account for when it's the first value in the scale of categories
                new = np.eye(np.argmax(original_rep))[np.random.choice(np.argmax(original_rep))]
                new = np.insert(new, np.argmax(new_rep), 0)
                new = np.concatenate((new, np.zeros((len(original_rep) - np.argmax(original_rep) - 1))))

            except:
                new = new_rep
        else:
            new = new_rep

    return new


def init_population(x, X_train, continuous_features, categorical_features, action_meta, cat_idxs, DIVERSITY_SIZE, replace=True):
    num_features = X_train.shape[1]
    population = np.zeros((POP_SIZE, DIVERSITY_SIZE, num_features))

    # iterate continous features
    for i in range(len(continuous_features.columns)):

        cat_name = continuous_features.columns[i]
        value = x[i]

        # If the continuous feature can take any value
        if action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
            f_range = action_meta[cat_name]['max'] - action_meta[cat_name]['min']
            temp = value + np.random.normal(0, CONT_PERTURB_STD, (POP_SIZE, DIVERSITY_SIZE, 1))
            temp *= f_range
            population[:, :, i:i + 1] = temp

        # If the continous feature can only go up
        elif action_meta[cat_name]['can_increase'] and not action_meta[cat_name]['can_decrease']:
            f_range = action_meta[cat_name]['max'] - value
            temp = value + abs(np.random.normal(0, CONT_PERTURB_STD, (POP_SIZE, DIVERSITY_SIZE, 1)))
            temp *= f_range
            population[:, :, i:i + 1] = temp

        # if the continuous features can only go down
        elif not action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
            f_range = value
            temp = value - abs(np.random.normal(0, CONT_PERTURB_STD, (POP_SIZE, DIVERSITY_SIZE, 1)))
            temp *= f_range
            population[:, :, i:i + 1] = temp

        # If it's not actionable
        else:
            temp = np.zeros((POP_SIZE, DIVERSITY_SIZE, 1)) + value
            population[:, :, i:i + 1] = temp

    # iterate categorical features
    current_idx = len(continuous_features.columns)
    for i in range(len(categorical_features.columns)):
        cat_len = len(x[cat_idxs[i][0]: cat_idxs[i][1]])
        temp = list()

        for j in range(POP_SIZE):
            temp2 = list()
            for k in range(DIVERSITY_SIZE):
                x_prime = deepcopy(x)  # to keep x the same
                temp3 = generate_category(x, x_prime, i, cat_idxs, categorical_features, action_meta, replace=True)
                temp2.append(temp3.tolist())
            temp.append(temp2)

        temp = np.array(temp)
        population[:, :, current_idx:current_idx + cat_len] = temp
        current_idx += cat_len

    return population


def get_rand_actionable_feature_idx(x, actionable_idxs, cat_idxs):
    """
    sample a random actionable feature index
    """

    instance_specific_actionable_indexes = deepcopy(actionable_idxs)

    # Get starting index of categories in actionable index list
    for i in range(len(actionable_idxs)):
        if actionable_idxs[i][0] == cat_idxs[0][0]:
            break
    starting_index = i

    for idx, i in enumerate(list(range(starting_index, len(actionable_idxs)))):

        sl = x[cat_idxs[idx][0]: cat_idxs[idx][1]]

        at_top = sl[-1] == 1
        can_only_go_up = actionable_idxs[i][1]

        at_bottom = sl[0] == 1
        can_only_go_down = actionable_idxs[i][2]

        if can_only_go_up and at_top:
            instance_specific_actionable_indexes.remove(actionable_idxs[i])

        if can_only_go_down and at_bottom:
            instance_specific_actionable_indexes.remove(actionable_idxs[i])

    rand = np.random.randint(len(instance_specific_actionable_indexes))
    return instance_specific_actionable_indexes[rand]


def perturb_continuous(x, x_prime, idx, continuous_features, categorical_features, action_meta):
    """
    slightly perturb continuous feature with actionability constraints
    """

    # Get feature max and min -- and clip it to these
    feature_names = continuous_features.columns.tolist() + categorical_features.columns.tolist()
    cat_name = feature_names[idx]

    if action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
        max_value = action_meta[cat_name]['max']
        min_value = action_meta[cat_name]['min']

    elif action_meta[cat_name]['can_increase'] and not action_meta[cat_name]['can_decrease']:
        max_value = action_meta[cat_name]['max']
        min_value = x[idx]

    elif not action_meta[cat_name]['can_increase'] and action_meta[cat_name]['can_decrease']:
        max_value = x[idx]
        min_value = action_meta[cat_name]['min']

    else:  # not actionable
        max_value = x[idx]
        min_value = x[idx]

    perturb = gauss(0, ((max_value - min_value) * CONT_PERTURB_STD))
    x_prime[idx] += perturb

    if x_prime[idx] > max_value:
        x_prime[idx] = max_value
    if x_prime[idx] < min_value:
        x_prime[idx] = min_value

    return x_prime


def perturb_one_random_feature(x, x_prime, continuous_features, categorical_features, action_meta, cat_idxs,
                               actionable_idxs):
    """
    perturb one actionable feature for MC robustness optimization
    """

    feature_names = continuous_features.columns.tolist() + categorical_features.columns.tolist()
    change_idx = get_rand_actionable_feature_idx(x, actionable_idxs, cat_idxs)[0]
    feature_num = len(feature_names)

    # if categorical feature
    if feature_names[change_idx] in categorical_features.columns:
        perturbed_feature = generate_category(x,
                                              x_prime,
                                              change_idx - len(continuous_features.columns),
                                              cat_idxs,
                                              categorical_features,
                                              action_meta,
                                              replace=False)

        x_prime[cat_idxs[change_idx - len(continuous_features.columns)][0]:
                cat_idxs[change_idx - len(continuous_features.columns)][1]] = perturbed_feature

    # if continuous feature
    else:
        x_prime = perturb_continuous(x,
                                     x_prime,
                                     change_idx,
                                     continuous_features,
                                     categorical_features,
                                     action_meta)

    return x_prime


def get_reachability(REACH_KNN, solution):
    """
    OOD Check using NN-dist metric
    """

    l2s, _ = REACH_KNN.kneighbors(X=solution, n_neighbors=1, return_distance=True)
    l2s = 1 / (l2s ** 2 + 0.1)
    return l2s


def get_gain(x, solution):
    """
    Return mean distance between query and semifactuals
    """

    scores = np.sqrt(((x - solution) ** 2).sum(axis=1))
    return scores


def get_robustness(x, solution, clf, cat_idxs, actionable_idxs, action_meta, continuous_features, categorical_features):
    """
    Monte Carlo Approximation of e-neighborhood robustness
    """

    perturbation_preds = list()
    for x_prime in solution:
        instance_perturbations = list()
        for _ in range(MAX_MC):
            x_prime_clone = deepcopy(x_prime)
            perturbed_instance = perturb_one_random_feature(x,
                                                            x_prime_clone,
                                                            continuous_features,
                                                            categorical_features,
                                                            action_meta,
                                                            cat_idxs,
                                                            actionable_idxs)

            instance_perturbations.append(perturbed_instance.tolist())
        predictions = clf.predict(instance_perturbations) == POSITIVE_CLASS
        perturbation_preds.append(predictions.tolist())
    return np.array(perturbation_preds).mean(axis=1)


def get_diversity(solution, DIVERSITY_SIZE):
    """
    Return L2 distance between all vectors (the mean)
    """

    if DIVERSITY_SIZE == 1:
        return 0

    # Take average distance
    score = distance_matrix(solution, solution).sum() / (DIVERSITY_SIZE ** 2 - DIVERSITY_SIZE)
    return score

def force_sf(clf, result):
    result_preds = clf.predict(result)
    keep = np.where(result_preds==abs(POSITIVE_CLASS))[0]
    sf = result[keep[0]]
    replace_these_idxs = np.where(result_preds==abs(POSITIVE_CLASS-1))[0]
    for idx in replace_these_idxs:
        result[idx] = sf
    return result, replace_these_idxs

## Genetic Algorithm
def fitness(REACH_KNN, x, population, cat_idxs, actionable_idxs, clf, action_meta, continuous_features, categorical_features, DIVERSITY_SIZE):

    fitness_scores = list()
    meta_fitness = list()

    for solution in population:
        reachability = get_reachability(REACH_KNN, solution)
        gain = get_gain(x, solution)
        robustness_1 = get_robustness(x, solution, clf, cat_idxs,
                                      actionable_idxs, action_meta,
                                      continuous_features, categorical_features) * 1

        robustness_2 = (clf.predict(solution) == POSITIVE_CLASS) * 1
        diversity = get_diversity(solution, DIVERSITY_SIZE)

        term1 = np.array(reachability.flatten() * gain)
        robustness_1 = np.array(robustness_1)
        robustness_2 = np.array(robustness_2)

        robustness_1 *= LAMBDA1
        robustness_2 *= LAMBDA2
        diversity *= GAMMA

        term1 = (term1 + robustness_1 + robustness_2).mean()

        correctness = clf.predict(solution).mean()  # hard constraint that the solution MUST contain SF
        fitness_scores.append((term1 + diversity).item() * correctness)
        meta_fitness.append([reachability.mean(), gain.mean(), robustness_1.mean(), robustness_2.mean(), diversity])

    return np.array(fitness_scores), np.array(meta_fitness)


def natural_selection(population, fitness_scores):
    """
    Save the top solutions
    """

    tournamet_winner_idxs = list()
    for i in range(POP_SIZE - ELITIST):
        knights = np.random.randint(0, population.shape[0], 2)
        winner_idx = knights[np.argmax(fitness_scores[knights])]
        tournamet_winner_idxs.append(winner_idx)
    return population[tournamet_winner_idxs], population[(-fitness_scores).argsort()[:ELITIST]]


def crossover(population, actionable_idxs, continuous_features, cat_idxs, DIVERSITY_SIZE):
    """
    mix up the population
    """

    children = list()

    for i in range(0, population.shape[0], 2):

        parent1, parent2 = population[i:i + 2]
        child1, child2 = deepcopy(parent1), deepcopy(parent2)

        crossover_idxs = np.random.randint(low=0,
                                           high=2,
                                           size=DIVERSITY_SIZE * len(actionable_idxs)).reshape(DIVERSITY_SIZE,
                                                                                               len(actionable_idxs))

        # Crossover Children
        for j in range(DIVERSITY_SIZE):
            for k in range(len(actionable_idxs)):

                # Child 1
                if crossover_idxs[j][k] == 0:

                    # if continuous
                    if actionable_idxs[k][0] < len(continuous_features.columns):
                        child1[j][actionable_idxs[k][0]] = parent2[j][actionable_idxs[k][0]]

                    # if categorical
                    else:
                        cat_idx = actionable_idxs[k][0] - len(continuous_features.columns)
                        child1[j][cat_idxs[cat_idx][0]: cat_idxs[cat_idx][1]] = parent2[j][
                                                                                cat_idxs[cat_idx][0]: cat_idxs[cat_idx][
                                                                                    1]]


                # Child 2
                else:
                    # if continuous
                    if actionable_idxs[k][0] < len(continuous_features.columns):
                        child2[j][actionable_idxs[k][0]] = parent1[j][actionable_idxs[k][0]]

                    # if categorical
                    else:
                        cat_idx = actionable_idxs[k][0] - len(continuous_features.columns)
                        child2[j][cat_idxs[cat_idx][0]: cat_idxs[cat_idx][1]] = parent1[j][
                                                                                cat_idxs[cat_idx][0]: cat_idxs[cat_idx][
                                                                                    1]]

        children.append(child1.tolist())
        children.append(child2.tolist())

    return np.array(children)


def mutation(population, continuous_features, categorical_features, x, actionable_idxs, cat_idxs, action_meta, DIVERSITY_SIZE):
    """
    Iterate all features and randomly perturb them
    """

    feature_names = continuous_features.columns.tolist() + categorical_features.columns.tolist()

    for i in range(len(population)):
        for j in range(DIVERSITY_SIZE):
            x_prime = population[i][j]
            for k in range(len(actionable_idxs)):
                if np.random.rand() < MUTATION_RATE:
                    change_idx = actionable_idxs[k][0]
                    # if categorical feature
                    if feature_names[change_idx] in categorical_features.columns:
                        perturbed_feature = generate_category(x,
                                                              x_prime,
                                                              change_idx - len(continuous_features.columns),
                                                              # index of category for function
                                                              cat_idxs,
                                                              categorical_features,
                                                              action_meta,
                                                              replace=False)
                        x_prime[cat_idxs[change_idx - len(continuous_features.columns)][0]:
                                cat_idxs[change_idx - len(continuous_features.columns)][1]] = perturbed_feature

                    # if continuous feature
                    else:
                        x_prime = perturb_continuous(x,
                                                     x_prime,
                                                     change_idx,
                                                     continuous_features,
                                                     categorical_features,
                                                     action_meta)
    return population


def sgen_genetic(clf, x, X_train, continuous_features, categorical_features, action_meta, cat_idxs, actionable_idxs, REACH_KNN, label, DIVERSITY_SIZE):

    sf_data = list()
    found_sfs = list()
    # check if probability of positive prediction (i.e. class 1) is greater than 0.6 and label of query is also 1
    if clf.predict_proba(x.reshape(1, -1))[0][1] > 0.6 and label==1:

        # this while loop exists so that the initial population has at least one semifactual
        avg_preds = 0.0
        counter_xxx = 0
        while avg_preds < 0.3:
            counter_xxx += 1
            population = init_population(x, X_train, continuous_features, categorical_features, action_meta, cat_idxs, DIVERSITY_SIZE,
                                         replace=True)
            avg_preds = clf.predict(population.reshape(-1, x.shape[0])).mean()
            if counter_xxx == 100:
                break

        if counter_xxx != 100:
            # Start GA
            for generation in range(MAX_GENERATIONS):
                # Evaluate fitness (meta = reachability, gain, robustness, diversity)
                fitness_scores, meta_fitness = fitness(REACH_KNN, x, population, cat_idxs,
                                                       actionable_idxs, clf, action_meta,
                                                       continuous_features, categorical_features, DIVERSITY_SIZE)

                # Selection
                population, elites = natural_selection(population, fitness_scores)

                # Crossover
                population = crossover(population, actionable_idxs, continuous_features, cat_idxs, DIVERSITY_SIZE)

                # Mutate
                population = mutation(population, continuous_features, categorical_features, x, actionable_idxs,
                                      cat_idxs, action_meta, DIVERSITY_SIZE)

                # Carry over elite solutions
                population = np.concatenate((population, elites), axis=0)

                # Evaluate fitness (meta = reachability, gain, robustness, diversity)
                fitness_scores, meta_fitness = fitness(REACH_KNN, x, population, cat_idxs,
                                                       actionable_idxs, clf, action_meta,
                                                       continuous_features, categorical_features, DIVERSITY_SIZE)

            result = population[np.argmax(fitness_scores)]

            if sum(fitness_scores * (meta_fitness.T[-2] == LAMBDA2)) > 0:
                for d in result:
                    sf_data.append(d.tolist())
                    #found_sfs.append([test_idx, True])

            else:
                result, replaced_these_idxs = force_sf(clf, result)
                for idx, d in enumerate(result):
                    sf_data.append(d.tolist())

                    # if idx in replaced_these_idxs:
                    #     found_sfs.append([test_idx, False])
                    # else:
                    #     found_sfs.append([test_idx, True])
    return sf_data