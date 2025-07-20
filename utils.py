import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

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
