import math

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score,
)

import logging
import os
import yaml
import random
import shutil

def metrics(labels, probs):
    score = {}
    with torch.no_grad():
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()
        score['AUROC'] = float(roc_auc_score(labels, probs) * 100)
        score['AUPRC'] = float(average_precision_score(labels, probs) * 100)
        labels = np.array(labels)
        k = labels.sum()
    score['RecK'] = float(sum(labels[probs.argsort()[-k:]]) / sum(labels) * 100)
    return score

def metrics1(labels, probs):
    score = {}
    with torch.no_grad():
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()
        score['AUROC'] = float(roc_auc_score(labels, probs) * 100)
        score['AUPRC'] = float(average_precision_score(labels, probs) * 100)
        pred_anomaly = probs > 0.5
        score['Pre'] = float(labels[pred_anomaly].sum() / len(pred_anomaly) * 100)
        labels = np.array(labels)
        k = labels.sum()
    score['RecK'] = float(sum(labels[probs.argsort()[-k:]]) / sum(labels) * 100)
    return score


def compute_metrics_from_probs(probs, y, threshold):
    """Compute the 7 evaluation metrics from positive-class probabilities."""
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    probs = np.asarray(probs).reshape(-1)
    y = np.asarray(y).reshape(-1)

    if len(set(y.tolist())) < 2:
        auc = float("nan")
        pr_auc = float("nan")
    else:
        auc = float(roc_auc_score(y, probs))
        pr_auc = float(average_precision_score(y, probs))

    preds = (probs >= threshold).astype(int)

    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    fraud_f1 = float(f1_score(y, preds, pos_label=1, zero_division=0))
    fraud_precision = float(precision_score(y, preds, pos_label=1, zero_division=0))
    fraud_recall = float(recall_score(y, preds, pos_label=1, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = math.sqrt(sensitivity * specificity)

    return {
        'ROC-AUC': auc * 100,
        'PR-AUC': pr_auc * 100,
        'Macro-F1': macro_f1 * 100,
        'Fraud-F1': fraud_f1 * 100,
        'Fraud-Precision': fraud_precision * 100,
        'Fraud-Recall': fraud_recall * 100,
        'GMean': gmean * 100,
    }


def find_best_threshold(probs, y, metric='GMean'):
    """Select the threshold with the best validation metric."""
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    probs = np.asarray(probs).reshape(-1)

    quantiles = np.linspace(0.0, 1.0, 101)
    adaptive = np.quantile(probs, quantiles).tolist()
    fixed = [i / 20 for i in range(2, 19)]
    thresholds = sorted({float(t) for t in [*fixed, *adaptive] if 0.0 <= float(t) <= 1.0})

    best_threshold = None
    best_score = float("-inf")
    for threshold in thresholds:
        m = compute_metrics_from_probs(probs, y, threshold)
        score = m[metric]
        if not (isinstance(score, float) and math.isfinite(score)):
            continue
        if score > best_score:
            best_score = score
            best_threshold = threshold

    if best_threshold is None:
        raise ValueError(f"Could not select a threshold for metric {metric!r}.")
    return best_threshold


def count_parameters(model):
    """
    count the parameters' number of the input model
    Note: The unit of return value is millions(M) if exceeds 1,000,000.
    :param model: the model instance you want to count
    :return: The number of model parameters, in Million (M).
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = round(num_params / 1e3, 3)
    return num_params


def get_training_config(dataset, config_path='train.conf.yaml'):
    with open(config_path, 'r') as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    default_config = full_config['default']
    dataset_config = full_config[dataset]
    dataset_config = dict(default_config, **dataset_config)
    return dataset_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


def check_readable(path):
    if not os.path.exists(path):
        raise ValueError(f"No such file or directory! {path}")


def get_logger(filename, log_level=1, name=None, mode='a'):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[log_level])

    # Clean logger first to avoid duplicated handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
