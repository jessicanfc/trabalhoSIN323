import numpy as np
import json
import os
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


def compute_evaluation_measures(y_true: list, y_pred: list, results_dict: dict):
    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    results_dict['all_accuracy'].append(report_dict['accuracy'])
    results_dict['all_macro_avg_p'].append(dict(report_dict['macro avg'])['precision'])
    results_dict['all_macro_avg_r'].append(dict(report_dict['macro avg'])['recall'])
    results_dict['all_macro_avg_f1'].append(dict(report_dict['macro avg'])['f1-score'])
    results_dict['all_weighted_avg_p'].append(dict(report_dict['weighted avg'])['precision'])
    results_dict['all_weighted_avg_r'].append(dict(report_dict['weighted avg'])['recall'])
    results_dict['all_weighted_avg_f1'].append(dict(report_dict['weighted avg'])['f1-score'])


def compute_means_std_eval_measures(clf_name: str, all_y_test: list, all_y_pred: list, results_dict: dict,
                                    results_dir: str):

    new_results_dict = {}

    for measure_name, measure_values in results_dict.items():
        mean_label = measure_name.replace('all_', 'mean_')
        std_label = measure_name.replace('all_', 'std_')
        new_results_dict[mean_label] = np.mean(measure_values)
        new_results_dict[std_label] = np.std(measure_values)

    results_dict.update(new_results_dict)

    all_y_test = [int(y) for y in all_y_test]
    all_y_pred = [int(y) for y in all_y_pred]

    results_dict['all_y_test'] = all_y_test
    results_dict['all_y_pred'] = all_y_pred

    classification_report_file_name = f'{clf_name}_report.json'.lower()

    classification_report_file_path = os.path.join(results_dir, classification_report_file_name)

    with open(classification_report_file_path, 'w') as file:
        json.dump(results_dict, file, indent=4)

    ConfusionMatrixDisplay.from_predictions(all_y_test, all_y_pred)

    confusion_matrix_name = f'{clf_name}_confusion_matrix.pdf'.lower()

    img_path = os.path.join(results_dir, confusion_matrix_name)

    plt.savefig(img_path, dpi=300)
