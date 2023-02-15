# This file is copyright (C) 2023 Quantitative Imaging Lab, Charité Universitätsmedizin Berlin, All rights reserved.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

feature_names = # Your list of feature names

'''The data is prepared as follows:
    activations.pickle: pandas DataFrame structured like that:
        {'layer1': […], #list of list containing the layers activations for each lesion,
            #activation lists for the layers 2-11
            'layer12': […],
            'annotation_id': […], #list of unique identifiers for each lesion
            'SHAPE_ROUND': […], #list of specific feature annotation for each lesion,
            #annotation lists for the PI-RADS features
            'T2_ORGANIZEDCHAOS': […],
        }
    feature_names: list of the names of the annotated features
'''

c_list = [10000, 1000, 100, 10, 1, 0.1, 0.05, 0.01]

def balanced_split_binary(x, ratio=0.8):
    if not 0 < ratio < 1:
        print("Ratio must be between 0 and 1. Default 0.8 is chosen")
        ratio = 0.8

    label0 = np.flatnonzero(x == 0)
    label1 = np.flatnonzero(x == 1)
    small_sample = min(label0.size, label1.size)
    n_train = int(ratio * small_sample)
    n_test = int((1 - ratio) * small_sample)
    train_index = np.hstack((label0[:n_train], label1[:n_train]))
    test_index_balanced = np.hstack((label0[n_train:n_train + n_test], label1[n_train:n_test + n_test]))
    test_index = np.hstack((label0[n_train:], label1[n_train:]))
    return train_index, test_index, test_index_balanced

def get_balanced_indexes(x):
    label0 = np.flatnonzero(x == 0)
    label1 = np.flatnonzero(x == 1)
    small_sample = min(label0.size, label1.size)
    index = np.hstack((label0[:small_sample], label1[:small_sample]))
    return index

def get_fold_results(df_dataset, train, test, y, c, balanced_index):
    scores = []
    reports = []
    classifiers = []
    for layer in range(12):
        x = np.vstack(df_dataset[('layer' + str(layer))])[balanced_index]
        l1 = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = c).fit(x[train], y[train])
        scores.append(l1.score(x[test], y[test]))
        reports.append(classification_report(y[test], l1.predict(x[test])))
        classifiers.append(l1)
    return {
        'score_layers' : scores,
        'classification_report_layers' : reports,
        'classifiers' : classifiers,
    }

def get_classification_results(df_dataset, feature_names, c):
    results = {}
    kf = StratifiedKFold()
    for name in feature_names:
        print(name)
        results[name] = {}
        fold_number = 0
        scores_layers = []
        balanced_index =  get_balanced_indexes(np.array(df_dataset[name]))
        y = np.array(df_dataset[name])[balanced_index]
        for train, test in kf.split(X = np.zeros(balanced_index.size), y = y):
            print(fold_number)
            results[name][f"fold_{fold_number}"] = get_fold_results(df_dataset, train, test, y, c, balanced_index)
            scores_layers.append(results[name][f"fold_{fold_number}"]['score_layers'])
            fold_number = fold_number + 1
        scores_layers = np.array(scores_layers)
        mean_layer = []
        std_layer = []
        for i in range(12):
            mean_layer.append(np.mean(scores_layers[:, i]))
            std_layer.append(np.std(scores_layers[:, i]))
        results[name]['mean_score_layers'] = mean_layer
        results[name]['std_score_layers'] = std_layer
    return results

def get_second_iteration(df_dataset, feature_name, best_c):
    results = {}
    promising_c_list = [best_c/4, best_c/2, best_c*0.75, best_c*2.5, best_c*5, best_c*7.5]
    for c in promising_c_list:
        c = int(c)
        print(c)
        classification_result = get_classification_results(df_dataset, [feature_name], c)
        results[c] = classification_result[feature_name]
    return results

def get_promising_features(dict_results, feature_names, c_list):
    promising_features = []
    for name in feature_names:
        for c in c_list:
            if np.max(dict_results[c][name]['mean_score_layers']) >= 0.75:
                promising_features.append(name)
                break
    return promising_features

def get_best_c_for_promising_features(dict_results, promising_features, c_list):
    promising_features_cs = {}
    for name in promising_features:
        best_c = -1
        best_score = 0
        for c in c_list:
            if np.max(dict_results[c][name]['mean_score_layers']) > best_score:
                best_c = c
                best_score = np.max(dict_results[c][name]['mean_score_layers'])
        promising_features_cs[name] = best_c
    return promising_features_cs

def get_best_scores_and_cs(feature_names, dict_results, dict_second_iteration, promising_features, c_list):
    dict_scores = {}
    for name in feature_names:
        dict_scores[name] = {}
    for name in promising_features:
        for c in dict_second_iteration[name].keys():
            dict_scores[name][c] = np.max(dict_second_iteration[name][c]['mean_score_layers'])
    for name in feature_names:
        for c in c_list:
            dict_scores[name][c] = np.max(dict_results[c][name]['mean_score_layers'])
    return dict_scores

def get_best_c_for_each_feature(feature_names, dict_scores):
    dict_c_max = {}
    for name in feature_names:
        mean = 0
        c_max = 0
        for c in dict_scores[name]:
            if dict_scores[name][c] > mean:
                mean = dict_scores[name][c]
                c_max = c
        dict_c_max[name] = c_max
    return dict_c_max

def get_best_layer_and_classifiers(feature_names, dict_c_max, dict_results, dict_second_iteration):
    dict_layer_max = {}
    for name in feature_names:
        if dict_c_max[name] not in c_list:
            scores = dict_second_iteration[name][dict_c_max[name]]['mean_score_layers']
            best_layer = scores.index(np.max(scores))
            classifiers = get_classifiers_for_best_layer(dict_second_iteration, name, dict_c_max, best_layer)
        else:
            scores = dict_results[dict_c_max[name]][name]['mean_score_layers']
            best_layer = scores.index(np.max(scores))
            classifiers = get_classifiers_for_best_layer(dict_results, name, dict_c_max, best_layer)
        dict_layer_max[name] = {'best_layer': best_layer, 'classifiers': classifiers}
    return dict_layer_max

def get_classifiers_for_best_layer(dict_results, name, dict_c_max, best_layer):
    classifiers = []
    for fold in range(5):
        classifiers.append(dict_results[dict_c_max[name]][name][f"fold_{fold}"]['classifiers'][best_layer])
    return classifiers

def get_dict_classifiers(promising_features, dict_layer_max):
    dict_classifiers = {}
    for name in promising_features:
        dict_classifiers[name] = {}
        dict_classifiers[name]['classifiers'] = dict_layer_max[name]['classifiers']
        dict_classifiers[name]['layer'] = dict_layer_max[name]['best_layer']
    return dict_classifiers

def save_dict_classifiers(dict_classifiers):
    with open('./output/classifier_layerwise.pickle', 'wb') as handle:
        pickle.dump(dict_classifiers, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    df_dataset = pd.read_pickle('./data/activations.pkl')
    dict_results = {}
    for c in c_list:
        print(c)
        dict_results[c] = get_classification_results(df_dataset, feature_names, c)
    
    promising_features = get_promising_features(dict_results, feature_names, c_list)
    print('Promising Features:')
    print(promising_features)
    promising_features_cs = get_best_c_for_promising_features(dict_results, promising_features, c_list)

    print('Second iteration for promising features:')
    dict_second_iteration = {}
    for name in promising_features:
        best_c = promising_features_cs[name]
        dict_second_iteration[name] = get_second_iteration(df_dataset, name, best_c)
        
    dict_scores = get_best_scores_and_cs(feature_names, dict_results, dict_second_iteration, promising_features, c_list)
    dict_c_max = get_best_c_for_each_feature(feature_names, dict_scores)
    dict_layer_max = get_best_layer_and_classifiers(feature_names, dict_c_max, dict_results, dict_second_iteration)
    dict_classifiers = get_dict_classifiers(promising_features, dict_layer_max)
    save_dict_classifiers(dict_classifiers) 

