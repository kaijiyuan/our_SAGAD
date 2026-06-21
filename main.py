import argparse
import os

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import load_data
from model import SAGAD, precompute_Tx, NodeDimGatedFusion, LowFusion, HighFusion, MeanFusion, ConcatFusion, \
    ScalarVectorFusion, AttentionFusion
from mrqsampler import mrqsample
from feature_missing import apply_missing_features, impute_missing_features
from utils import get_training_config, set_seed, get_logger, metrics, compute_metrics_from_probs, find_best_threshold


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch DGL implementation')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--symm', type=bool, default=True,
                        help='whether to symmetric normalize adjacency matrix')
    parser.add_argument('--num_exp', type=int, default=10,
                        help='Repeat how many experiments')
    parser.add_argument('--dataset', type=str, default='reddit',
                        choices=['reddit', 'weibo', 'amazon', 'yelp', 'tfinance',
                                 'elliptic', 'tolokers', 'questions', 'dgraphfin', 'tsocial'])
    parser.add_argument('--fusion', type=str, default='NodeDimGated',
                        choices=['Low', 'High', 'Mean', 'Concat', 'Scalar',
                                 'Vector', 'Attention', 'NodeDimGatedWoReg', 'NodeDimGated'])
    parser.add_argument('--exp_setting', type=str, default='tran',
                        choices=['tran', 'ind'], help='Experiment setting, one of [tran, ind]')

    args = parser.parse_args()
    return args


def run(conf, Tx_list, X, Y, idx_train, idx_val, idx_test):
    """ Set seed """
    set_seed(conf['seed'])

    device = X.device

    Tx_list_train = [Tx[idx_train] for Tx in Tx_list]
    Tx_list_val = [Tx[idx_val] for Tx in Tx_list]
    Tx_list_test = [Tx[idx_test] for Tx in Tx_list]
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    Y_train, Y_val, Y_test = Y[idx_train], Y[idx_val], Y[idx_test]

    w = (1 - Y_train).sum().item() / Y_train.sum().item()

    weights_re = torch.ones_like(Y_train, device=device, dtype=torch.float32)
    weights_re[Y_train == 1] = w
    d_train = Y_train.clone().float()
    d_train[Y_train == 1] = conf['p'][0]
    d_train[Y_train == 0] = conf['p'][1]
    weights_ce = torch.tensor([1., w], device=device)

    in_dim = X_train.shape[1]
    emb_dim = in_dim // 2
    if conf['fusion'] == 'NodeDimGated':
        fusion = NodeDimGatedFusion(in_dim, emb_dim, conf['num_gate_layers'])
    elif conf['fusion'] == 'NodeDimGatedWoReg':
        fusion = NodeDimGatedFusion(in_dim, emb_dim, conf['num_gate_layers'])
    elif conf['fusion'] == 'Low':
        fusion = LowFusion
    elif conf['fusion'] == 'High':
        fusion = HighFusion
    elif conf['fusion'] == 'Mean':
        fusion = MeanFusion
    elif conf['fusion'] == 'Concat':
        fusion = ConcatFusion(in_dim)
    elif conf['fusion'] == 'Scalar':
        fusion = ScalarVectorFusion(in_dim, scalar=True)
    elif conf['fusion'] == 'Scalar':
        fusion = ScalarVectorFusion(in_dim, scalar=False)
    elif conf['fusion'] == 'Attention':
        fusion = AttentionFusion(in_dim)
    else:
        fusion = None

    model = SAGAD(in_dim, emb_dim, hid_dims=conf['hid_dims'] + [2], K=conf['K'], dropout=conf['dropout'],
                  activation=conf['activation'], mlp_norm=conf['mlp_norm'], fusion=fusion).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'], weight_decay=conf['wd'])

    best_epoch = 0
    best_val_score = {'AUROC': 0, 'AUPRC': 0, 'RecK': 0}
    with tqdm(total=conf['epochs'],
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
        for epoch in range(1, 1 + conf['epochs']):
            model.train()
            optimizer.zero_grad()
            if conf['fusion'] == 'NodeDimGated':
                logits, coefs = model(Tx_list_train, X_train, return_coefs=True)
                loss_ce = F.cross_entropy(logits, Y_train, weight=weights_ce)
                loss_re = F.binary_cross_entropy(coefs.mean(1), d_train, weight=weights_re)
                loss = loss_ce + loss_re
            else:
                logits = model(Tx_list_train, X_train, return_coefs=False)
                loss_ce = F.cross_entropy(logits, Y_train, weight=weights_ce)
                loss = loss_ce
            # loss = loss_ce
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                Y_val_pred = model(Tx_list_val, X_val).softmax(dim=1)[:, 1]
                val_score = metrics(Y_val, Y_val_pred)
                if val_score['AUROC'] > best_val_score['AUROC']:
                    best_epoch = epoch
                    best_val_score = val_score
                    torch.save(model.state_dict(), f"snapshots/{conf['dataset']}_snapshot.pkl")
                else:
                    if epoch - best_epoch > conf['patience']:
                        break

            pbar.set_postfix({'Val|AUROC': best_val_score['AUROC'],
                              'AUPRC': best_val_score['AUPRC'],
                              'RecK': best_val_score['RecK']})
            pbar.update()

    model.load_state_dict(torch.load(f"snapshots/{conf['dataset']}_snapshot.pkl", weights_only=False))
    model.eval()
    with torch.no_grad():
        Y_val_prob = model(Tx_list_val, X_val).softmax(dim=1)[:, 1]
        best_threshold = find_best_threshold(Y_val_prob, Y_val, metric='GMean')

        Y_test_prob = model(Tx_list_test, X_test).softmax(dim=1)[:, 1]
        test_score = compute_metrics_from_probs(Y_test_prob, Y_test, best_threshold)
        print(f"Test| ROC-AUC={test_score['ROC-AUC']:.2f}, PR-AUC={test_score['PR-AUC']:.2f}, "
              f"Macro-F1={test_score['Macro-F1']:.2f}, Fraud-F1={test_score['Fraud-F1']:.2f}, "
              f"Fraud-Precision={test_score['Fraud-Precision']:.2f}, "
              f"Fraud-Recall={test_score['Fraud-Recall']:.2f}, GMean={test_score['GMean']:.2f}")
    return test_score


def main():
    args = get_args()

    conf = get_training_config(args.dataset, config_path='semi_train.conf.yaml')
    log_dir = f'./logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists('snapshots'):
        os.makedirs('snapshots')
    logger = get_logger(f'{log_dir}/{args.dataset}.log')
    conf = dict(args.__dict__, **conf)
    logger.info(str(conf))

    device = f'cuda:{conf["device"]}' \
        if torch.cuda.is_available() else 'cpu'

    G, X, Y, train_masks, val_masks, test_masks = (
        load_data(args.dataset, feat_trans=conf['feat_trans']))

    G_mrq = mrqsample(args.dataset, G, X, one_hop=True)

    MISSING_RATIO = 0.995

    AUROCs, AUPRCs, MacroF1s, FraudF1s, FraudPrecs, FraudRecs, GMeans = [], [], [], [], [], [], []
    indices = torch.arange(X.shape[0])
    Y = Y.to(device)
    for i in range(5):
        conf['seed'] = i
        set_seed(conf['seed'])

        X_masked = apply_missing_features(X, MISSING_RATIO, seed=conf['seed'])
        X_masked = impute_missing_features(X_masked, method='zero')

        Tx_list = precompute_Tx(G, X_masked, conf['K'], conv_norm=conf['conv_norm'])
        X_nei = dgl.ops.copy_u_mean(G_mrq, X_masked)
        X_final = torch.concat([X_masked, X_nei], dim=1)

        Tx_list = [Tx.to(device) for Tx in Tx_list]
        X_final = X_final.to(device)

        idx_train = indices[train_masks[:, i]]
        idx_val = indices[val_masks[:, i]]
        idx_test = indices[test_masks[:, i]]
        score = run(conf, Tx_list, X_final, Y, idx_train, idx_val, idx_test)
        AUROCs.append(score['ROC-AUC'])
        AUPRCs.append(score['PR-AUC'])
        MacroF1s.append(score['Macro-F1'])
        FraudF1s.append(score['Fraud-F1'])
        FraudPrecs.append(score['Fraud-Precision'])
        FraudRecs.append(score['Fraud-Recall'])
        GMeans.append(score['GMean'])

    res = (f"Test| ROC-AUC={np.mean(AUROCs):.2f}+-{np.std(AUROCs):.2f}, "
           f"PR-AUC={np.mean(AUPRCs):.2f}+-{np.std(AUPRCs):.2f}, "
           f"Macro-F1={np.mean(MacroF1s):.2f}+-{np.std(MacroF1s):.2f}, "
           f"Fraud-F1={np.mean(FraudF1s):.2f}+-{np.std(FraudF1s):.2f}, "
           f"Fraud-Precision={np.mean(FraudPrecs):.2f}+-{np.std(FraudPrecs):.2f}, "
           f"Fraud-Recall={np.mean(FraudRecs):.2f}+-{np.std(FraudRecs):.2f}, "
           f"GMean={np.mean(GMeans):.2f}+-{np.std(GMeans):.2f}\n")

    logger.info(res)


if __name__ == "__main__":
    main()
