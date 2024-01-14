import configparser

from sklearn.metrics import precision_recall_curve, auc,  confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from collections import Counter
from sklearn.metrics import roc_curve


def plot_roc_curve(test_y, model_probs):
    fpr, tpr, _ = roc_curve(test_y, model_probs)
    plt.title("ROC CURVE")
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


# PR curve focuses on the minority class, whereas the ROC curve covers both classes.
def plot_pr_rec_curve(y_test, model_probs):
    precision, recall, _ = precision_recall_curve(y_test, model_probs)
    plt.plot(recall, precision)
    plt.title("PRECISION-RECALL CURVE")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


def plot_heat_map(results_path, plot, output_path):
    """
    Plot a heat-map of the test_score (mean of the test_scores in the nestedk-fold cross validation)
    :param results_path: path to csv of results
    :param plot: if True show heat map
    :param output_path: path to save the image
    :return:
    """

    data = pd.read_csv(results_path)

    data[['Selector', 'Classifier']] = data['combination_id'].str.split('_', expand=True)
    max_scores = data.groupby(['Selector', 'Classifier'])['test_score'].max().reset_index()

    pivot_df = max_scores.pivot(index="Selector", columns="Classifier", values="test_score")

    if plot:
        plt.figure(figsize=(5, 7))
        sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlGnBu")
        plt.title("Heatmap of Max Test Scores for Selector-Classifier Combinations")
        plt.show()

        plt.savefig(f"{output_path}/heatmap.jpg")
        plt.close()


def most_common_features(df_results, max_features, plot=False):
    feature_counts = Counter()
    first_row_dict = literal_eval(df_results['selected_features'].iloc[0])

    all_feature_names = list(first_row_dict.keys())

    for run_counts_string in df_results['selected_features']:
        run_counts_dict = literal_eval(run_counts_string)
        for feature in all_feature_names:
            count = run_counts_dict.get(feature, 0)
            feature_counts[feature] += int(count)

    filtered_feature_counts = {feature: count for feature, count in feature_counts.items() if count > 0}
    most_common = Counter(filtered_feature_counts).most_common(max_features)
    if plot:
        plt.figure(figsize=(12, 6))
        features, counts = zip(*most_common)

        plt.bar(features, counts, color='skyblue')
        plt.ylabel('Counts')
        plt.title('Most selected features')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    return most_common


def merge_dict(dict1, dict2):
    merged = dict1.copy()
    merged.update(dict2)

    return merged

def auc_pr(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)