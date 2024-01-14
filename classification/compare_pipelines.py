import argparse
import configparser
import logging
import os
import time
from collections import OrderedDict
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
from multiprocessing import cpu_count

from joblib import Parallel
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
import utils, algorithms
from joblib import delayed


def _read_config(section, key):
    config = configparser.ConfigParser()
    config.read('config.ini')
    if section in config and key in config[section]:
        return config[section][key]
    else:
        raise ValueError(f"Key '{key}' not found in section '{section}' in config.ini")


class Classification:
    def __init__(self, input_csv, verbose):
        self.verbose = verbose
        # Logger
        self.logger = logging.getLogger('ClassificationLogger')
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler('./log.txt', mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        # Paths

        config_path = os.path.join(os.getcwd(), 'config.ini')
        config = configparser.ConfigParser()
        config.read(config_path)
        self.n_splits = int(config['CV']['n_splits'])
        self.n_iter = int(config['CV']['n_iter'])

        self.dataframe = pd.read_csv(input_csv)
        if (config['CV']['smote']).lower() == 'yes':
            self.smote = True
        else:
            self.smote = False
        if self.smote:
            self.result_path = os.path.join(os.getcwd(), 'results', "smote_"+os.path.basename(input_csv))
        else:
            self.result_path = os.path.join(os.getcwd(), 'results', os.path.basename(input_csv))
        self.NUM_OF_WORKERS = cpu_count() - 1
        if self.NUM_OF_WORKERS < 1:
            self.NUM_OF_WORKERS = 1
        self.logger.info('Workers ' + str(self.NUM_OF_WORKERS))

    # https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
    # https://github.com/ahmedalbuni/biorad/blob/nmbu_ml/experiment/comparison_schemes.py#L192
    def k_fold_cross_validation(self,
                                X,
                                y,
                                groups,
                                pipeline,
                                params,
                                combination_id,
                                n_splits,
                                score_func,
                                n_iter,
                                verbose,
                                random_state,
                                columns_name):

        logging.basicConfig(filename='./results/log.log', filemode='a', level=logging.INFO)
        result = {'random_state': random_state, 'combination_id': combination_id}
        logging.info(f"Combination {result}")
        if verbose > 0:
            logging.info(f"Columns {columns_name}")
            logging.info(f"Pipeline {pipeline}")
            logging.info(f"Params {params}")

        scoring = {
            'f1_score': make_scorer(metrics.f1_score),
            'recall': make_scorer(metrics.recall_score),
            'specificity': make_scorer(utils.specificity_score),
            'precision': make_scorer(metrics.precision_score),
            'average_precision': make_scorer(metrics.average_precision_score),
            'auc_roc': make_scorer(metrics.roc_auc_score, needs_proba=True),
            'auc_pr': make_scorer(score_func, needs_proba=True)
        }

        gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=n_iter,
            scoring=scoring,
            cv=gkf,
            verbose=verbose,
            random_state=random_state,
            error_score='raise',
            refit='auc_pr',
            return_train_score=True,
            n_jobs=-1
        )

        random_search = search.fit(X, y, groups=groups)

        best_pipeline = random_search.best_estimator_
        test_score = random_search.best_score_
        cv_results = random_search.cv_results_

        if verbose > 0:
            outer_results = [{
                'test_score': test_score,
                'best_params': random_search.best_params_
            }]
            logging.info(outer_results)

        selector_name = combination_id.split("_")[0]

        selector = best_pipeline.named_steps[selector_name]
        logging.info(f"selector {selector}")

        selected_features = algorithms.get_selected_features(selector_name, selector, columns_name)

        result.update(
            OrderedDict(
                [
                    ('test_score', np.mean(cv_results['mean_test_auc_pr'])),
                    ('test_score_std', np.mean(cv_results['std_test_auc_pr'])),
                    ('train_score', np.mean(cv_results['mean_train_auc_pr'])),
                    ('train_score_std', np.mean(cv_results['std_train_auc_pr'])),
                    ('test_f1_score', np.mean(cv_results['mean_test_f1_score'])),
                    ('train_f1_score', np.mean(cv_results['mean_train_f1_score'])),
                    ('test_recall', np.mean(cv_results['mean_test_recall'])),
                    ('train_recall', np.mean(cv_results['mean_train_recall'])),
                    ('test_specificity', np.mean(cv_results['mean_test_specificity'])),
                    ('train_specificity', np.mean(cv_results['mean_train_specificity'])),
                    ('test_precision', np.mean(cv_results['mean_test_precision'])),
                    ('train_precision', np.mean(cv_results['mean_train_precision'])),
                    ('test_average_precision', np.mean(cv_results['mean_test_average_precision'])),
                    ('train_average_precision', np.mean(cv_results['mean_train_average_precision'])),
                    ('test_auc_roc', np.mean(cv_results['mean_test_auc_roc'])),
                    ('train_auc_roc', np.mean(cv_results['mean_train_auc_roc'])),
                    ('selected_features', selected_features)
                ]
            )
        )
        result.update(random_search.best_params_)

        return result

    def train_and_evaluate_models(self, smote=False):
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else 1
        np.random.seed(seed=42)
        random_states = np.random.choice(1000, size=5)
        columns_name = self.dataframe.drop(["Record ID", "Label", "Image", "Pixel spacing x", "ROI pixels"],
                                           axis=1).columns

        results = []
        combinations = algorithms.get_combinations(smote)
        start_time = time.time()

        # to numpy array to take advantage of job Parallel
        groups = self.dataframe['Record ID'].values
        X = self.dataframe.drop(["Record ID", "Label", "Image", "Pixel spacing x", "ROI pixels"], axis=1).values
        y = self.dataframe['Label'].values.squeeze()

        for name in combinations:
            self.logger.info("combination " + name)
            print("combination " + name)
            print("randoms " + str(len(random_states)))

            results.extend(
                Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                    delayed(self.k_fold_cross_validation)(
                        X=X,
                        y=y,
                        groups=groups,
                        pipeline=algorithms.get_pipeline(name, random_state, smote),
                        params=algorithms.get_params(name, random_state, smote),
                        n_splits=self.n_splits,
                        combination_id=name,
                        score_func=utils.auc_pr,
                        n_iter=self.n_iter,
                        verbose=self.verbose,
                        random_state=random_state,
                        columns_name=columns_name
                    ) for random_state in random_states
                )
            )

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f"Execution time: {duration} s")

        data = pd.DataFrame([result for result in results])

        data.to_csv(self.result_path, index=False)


def main(input_features=None):
    input_features = input_features or _read_config('PATH', 'input_features')
    clf = Classification(input_features, 1)
    clf.train_and_evaluate_models()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare pipelines')
    parser.add_argument('--input_features', type=str, help='File with the features to train the models.')

    args = parser.parse_args()
    main(args.input_features)
