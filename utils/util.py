import numpy as np
import time
import os
import logging
import sys

import torch
import torch.nn.functional as F
import itertools
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S"
    )
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

    return lg


def get_conditional_probability(pred, x_control, z_value, y=None, y_value=0):
    if y is not None:
        pred = pred[y == y_value]
        x_control = x_control[y == y_value]

    pr_y = sum(pred) / len(pred)

    pred = pred[x_control == z_value]
    pr_con = sum(pred) / len(pred)

    return pr_con, pr_y


def get_confustion_matrix(y_pred, y_true):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    cm = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    rm = {"fpr": fpr, "fnr": fnr, "tpr": tpr, "tnr": tnr}

    return cm, rm


def check_utility(pred, y):
    mean_auc = roc_auc_score(y, pred)

    pred_threshould = (pred > 0.5).astype(int)
    cm, rm = get_confustion_matrix(pred_threshould, y)
    recall = rm["tpr"]
    specificity = rm["tnr"]

    accuracy_dict = {"mean_auc": mean_auc, "recall": recall, "specificity": specificity}

    return accuracy_dict


def check_fairness(pred, y, a_d, threshold=0.5):
    """
    Check fairness metrics both in single attribute comparison and combinations of multiple attributes settings
    Fairness metrics include: FPR gap (EO), FNR gap, AUC gap, ED

    For single attribute settings, plz use the combination dict.

    Args:
        pred:
        y:
        a_d:
        threshold:

    Returns examples:
        fairness_matrix: {
            'single': {
                'Race': {'FPR gap': 0.02, 'FNR gap': 0.03, 'AUC gap': 0.2, 'ED': 0.03, 'EO': 0.02},
                'Age': ...,
                'Sex': ...,
                ...
            },
            'Combination': {'FPR gap': 0.02, 'FNR gap': 0.03, 'AUC gap': 0.2, 'ED': 0.03, 'EO': 0.02}
        }

        meta_data: {
            'single': {
                'Race': {
                    'FPR': [0.02, 0.03],
                    'FNR': ...,
                    'AUC': ...,
                    'product': [0, 1]
                },
                'Age': ...
                ...
            },
            'Combination': {
                'FPR': [0.12, 0.23, ...],
                'FNR': [...],
                'AUC': [...],
                'product': [(0, 0, 0), (0, 0, 1), (0, 1, 0), ..., (1, 1, 1)],
                'attributes': ['Race', 'Age', 'Sex']
            }
        }

    """
    pred = (pred > threshold).astype(int)

    if isinstance(a_d, np.ndarray):
        a_d = {"single": a_d}

    sensitive_attributes = list(a_d.keys())

    # single sensitive attribute
    meta_data_single = {}
    fairness_matrix_single = {}

    for i, sa in enumerate(sensitive_attributes):
        meta_data_single[sa] = {"FPR": [], "FNR": [], "AUC": [], "NUM": [], "product": [0, 1]}
        fairness_matrix_single[sa] = {}

        idx = np.arange(len(pred))
        idx_z_0 = idx[np.where(a_d[sa] == 0)[0]]
        idx_z_1 = idx[np.where(a_d[sa] == 1)[0]]

        # z=0 subgroup
        cm_z_0, rm_z_0 = get_confustion_matrix(pred[idx_z_0], y[idx_z_0])
        auc_z_0 = roc_auc_score(y[idx_z_0], pred[idx_z_0])

        # z=1 subgroup
        cm_z_1, rm_z_1 = get_confustion_matrix(pred[idx_z_1], y[idx_z_1])
        auc_z_1 = roc_auc_score(y[idx_z_1], pred[idx_z_1])

        meta_data_single[sa]["FPR"].extend([rm_z_0["fpr"], rm_z_1["fpr"]])
        meta_data_single[sa]["FNR"].extend([rm_z_0["fnr"], rm_z_1["fnr"]])
        meta_data_single[sa]["AUC"].extend([auc_z_0, auc_z_1])
        meta_data_single[sa]["NUM"].extend([idx_z_0, idx_z_1])

        fairness_matrix_single[sa]["FPR gap"] = abs(rm_z_1["fpr"] - rm_z_0["fpr"])
        fairness_matrix_single[sa]["FNR gap"] = abs(rm_z_1["fnr"] - rm_z_0["fnr"])
        fairness_matrix_single[sa]["AUC gap"] = abs(auc_z_1 - auc_z_0)
        fairness_matrix_single[sa]["ED"] = max(
            fairness_matrix_single[sa]["FPR gap"], fairness_matrix_single[sa]["FNR gap"]
        )
        fairness_matrix_single[sa]["EO"] = fairness_matrix_single[sa]["FPR gap"]

    # combine of multiple sensitive attributes
    a_product = list(itertools.product(range(2), repeat=len(a_d)))

    meta_data_combine = {"FPR": [], "FNR": [], "AUC": [], "NUM": [], "Pos Rate": []}
    fairness_matrix_combine = {}

    for p in a_product:
        idx = np.arange(len(pred))
        for i, sa in enumerate(sensitive_attributes):
            idx = idx[np.where(a_d[sa][idx] == p[i])[0]]

        cm, rm = get_confustion_matrix(pred[idx], y[idx])
        auc = roc_auc_score(y[idx], pred[idx])

        meta_data_combine["FPR"].append(rm["fpr"])
        meta_data_combine["FNR"].append(rm["fnr"])
        meta_data_combine["AUC"].append(auc)
        meta_data_combine["NUM"].append(len(idx))
        meta_data_combine["Pos Rate"].append(sum(y[idx]) / len(idx))

    fairness_matrix_combine["FPR gap"] = max(meta_data_combine["FPR"]) - min(
        meta_data_combine["FPR"]
    )
    fairness_matrix_combine["FNR gap"] = max(meta_data_combine["FNR"]) - min(
        meta_data_combine["FNR"]
    )
    fairness_matrix_combine["AUC gap"] = max(meta_data_combine["AUC"]) - min(
        meta_data_combine["AUC"]
    )
    fairness_matrix_combine["ED"] = max(
        fairness_matrix_combine["FPR gap"], fairness_matrix_combine["FNR gap"]
    )
    fairness_matrix_combine["EO"] = fairness_matrix_combine["FPR gap"]

    meta_data_combine["attributes"] = sensitive_attributes
    meta_data_combine["product"] = a_product

    fairness_matrix = {"single": fairness_matrix_single, "combination": fairness_matrix_combine}
    meta_data = {"single": meta_data_single, "combination": meta_data_combine}

    return fairness_matrix, meta_data
