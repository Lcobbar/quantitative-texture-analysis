from imblearn.ensemble import RUSBoostClassifier, BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skrebate import ReliefF, MultiSURF

import utils


def get_algorithms(random_state):
    params_dict = {

        'SVM': {'SVC__C': uniform(1, 10),
                'SVC__kernel': ['linear', 'rbf'],
                'SVC__gamma': ['scale', 'auto', 0.1, 1.0, 10.0]},

        'RANDOM_FOREST': {'BalancedRandomForestClassifier__n_estimators': randint(10, 350),
                          'BalancedRandomForestClassifier__criterion': ['gini', 'entropy'],
                          'BalancedRandomForestClassifier__max_depth': randint(3, 27),
                          'BalancedRandomForestClassifier__min_samples_split': randint(2, 15),
                          'BalancedRandomForestClassifier__min_samples_leaf': [1],
                          'BalancedRandomForestClassifier__max_features': [9, 11, 13, 15],
                          'BalancedRandomForestClassifier__sampling_strategy': ['not minority']},

        'RUSBOOST': {
            'RUSBoostClassifier__n_estimators': [50, 100, 500],
            'RUSBoostClassifier__learning_rate': [0.01, 0.1, 0.5, 1.0],
            'RUSBoostClassifier__sampling_strategy': ['not minority']
        },

        "RELIEFF": {
            "ReliefF__n_neighbors": randint(1, 5),
            "ReliefF__n_features_to_select": randint(5, 10),
        },

        "MULTISURF": {
            "MultiSURF__n_features_to_select": randint(5, 10)
        },

        "VARTH": {
            "VarianceThreshold__threshold": uniform(0, 0.5)
        },

        "KBEST": {
            "SelectKBest__k": randint(5, 10)
        }

    }

    # classifiers
    rus_param = params_dict['RUSBOOST']
    svm_param = params_dict['SVM']
    rf_param = params_dict['RANDOM_FOREST']

    # selectors
    relieff_param = params_dict['RELIEFF']
    multi_surf_param = params_dict["MULTISURF"]
    var_t_param = params_dict["VARTH"]
    kbest_param = params_dict["KBEST"]

    selector_dict = dict()
    selector_dict[VarianceThreshold.__name__] = (VarianceThreshold.__name__, VarianceThreshold()), var_t_param
    selector_dict[ReliefF.__name__] = (ReliefF.__name__, ReliefF()), relieff_param
    selector_dict[MultiSURF.__name__] = (MultiSURF.__name__, MultiSURF()), multi_surf_param
    selector_dict[SelectKBest.__name__] = (SelectKBest.__name__, SelectKBest(mutual_info_classif)), kbest_param

    classifier_dict = dict()
    classifier_dict[RUSBoostClassifier.__name__] = (
    RUSBoostClassifier.__name__, RUSBoostClassifier(random_state=random_state)), rus_param
    # class_weight='balanced' in SVC: weights are adjusted inversely proportional to the class frequencies
    classifier_dict[SVC.__name__] = (
    SVC.__name__, SVC(probability=True, random_state=random_state, class_weight='balanced')), svm_param
    classifier_dict[BalancedRandomForestClassifier.__name__] = (
    BalancedRandomForestClassifier.__name__, BalancedRandomForestClassifier(random_state=random_state)), rf_param

    return selector_dict, classifier_dict

def get_algorithms_smote(random_state):
    params_dict_smote = {

        'SVM': {'SVC__C': uniform(1, 10),
                'SVC__kernel': ['linear', 'rbf'],
                'SVC__gamma': ['scale', 'auto', 0.1, 1.0, 10.0]},

        'RANDOM_FOREST': {'RandomForestClassifier__n_estimators': randint(10, 350),
                          'RandomForestClassifier__criterion': ['gini', 'entropy'],
                          'RandomForestClassifier__max_depth': randint(3, 27),
                          'RandomForestClassifier__min_samples_split': randint(2, 15),
                          'RandomForestClassifier__min_samples_leaf': [1],
                          'RandomForestClassifier__max_features': [9, 11, 13, 15]},

        'KNN': {
            'KNeighborsClassifier__n_neighbors': randint(1, 10),
            'KNeighborsClassifier__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        },

        "RELIEFF": {
            "ReliefF__n_neighbors": randint(1, 5),
            "ReliefF__n_features_to_select": randint(5, 10),
        },

        "MULTISURF": {
            "MultiSURF__n_features_to_select": randint(5, 10)
        },

        "VARTH": {
            "VarianceThreshold__threshold": uniform(0, 0.5)
        },

        "KBEST": {
            "SelectKBest__k": randint(5, 10)
        }

    }

    # classifiers
    knn_param = params_dict_smote['KNN']
    svm_param = params_dict_smote['SVM']
    rf_param = params_dict_smote['RANDOM_FOREST']

    # selectors
    relieff_param = params_dict_smote['RELIEFF']
    multi_surf_param = params_dict_smote["MULTISURF"]
    var_t_param = params_dict_smote["VARTH"]
    kbest_param = params_dict_smote["KBEST"]

    selector_dict = dict()
    selector_dict[VarianceThreshold.__name__] = (VarianceThreshold.__name__, VarianceThreshold()), var_t_param
    selector_dict[ReliefF.__name__] = (ReliefF.__name__, ReliefF()), relieff_param
    selector_dict[MultiSURF.__name__] = (MultiSURF.__name__, MultiSURF()), multi_surf_param
    selector_dict[SelectKBest.__name__] = (SelectKBest.__name__, SelectKBest(mutual_info_classif)), kbest_param

    classifier_dict = dict()
    classifier_dict[KNeighborsClassifier.__name__] = (
    KNeighborsClassifier.__name__, KNeighborsClassifier(random_state=random_state)), knn_param
    classifier_dict[SVC.__name__] = (SVC.__name__, SVC(probability=True, random_state=random_state)), svm_param
    classifier_dict[RandomForestClassifier.__name__] = (
    RandomForestClassifier.__name__, RandomForestClassifier(random_state=random_state)), rf_param

    return selector_dict, classifier_dict


def get_pipeline(combination_name, random_state, smote=False):
    scalar = (StandardScaler.__name__, StandardScaler())

    if smote:
        smote = (SMOTE.__name__, SMOTE(random_state))
        selector_list, classifier_list = get_algorithms_smote(random_state)
        selector = selector_list[combination_name.split("_")[0]]
        classifier = classifier_list[combination_name.split("_")[1]]

        if selector[0][0] == VarianceThreshold.__name__:
            return Pipeline(
                [smote, selector[0], scalar, classifier[0]]
            )
        else:
            return Pipeline(
                [smote, scalar, selector[0], classifier[0]]
            )
    else:
        selector_list, classifier_list = get_algorithms(random_state)
        selector = selector_list[combination_name.split("_")[0]]
        classifier = classifier_list[combination_name.split("_")[1]]

        if selector[0][0] == VarianceThreshold.__name__:
            return Pipeline(
                [selector[0], scalar, classifier[0]]
            )
        else:
            return Pipeline(
                [scalar, selector[0], classifier[0]]
            )


def get_params(combination_name, random_state, smote=False):
    if smote:
        selector_list, classifier_list = get_algorithms_smote(random_state)
    else:
        selector_list, classifier_list = get_algorithms(random_state)
    selector = selector_list[combination_name.split("_")[0]]
    classifier = classifier_list[combination_name.split("_")[1]]

    return utils.merge_dict(selector[1], classifier[1])


def get_combinations(smote=False):
    combinations = []
    if smote:
        selector_list, classifier_list = get_algorithms_smote(32)
    else:
        selector_list, classifier_list = get_algorithms(32)

    for classifier_k, _ in classifier_list.items():
        for selector_k, _ in selector_list.items():
            combinations.append(f'{selector_k}_{classifier_k}')

    return combinations


def get_selected_features(selector_name, selector, features_list):
    selected_features = []
    if selector_name == VarianceThreshold.__name__ or selector_name == SelectKBest.__name__:
        features_selected_mask = selector.get_support()
        for idx, is_selected in enumerate(features_selected_mask):
            if is_selected:
                selected_features.append(features_list[idx])

    else:
        selector_array = selector.top_features_
        num = selector.n_features_to_select

        for i, idx in enumerate(selector_array):
            if i < num:
                selected_features.append(features_list[idx])

    return selected_features
