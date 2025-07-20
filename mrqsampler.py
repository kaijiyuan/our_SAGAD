# Refer to https://github.com/lllyyq1121/UniGAD/blob/main/src/utils.py

import os
import time

import dgl
import torch
from dgl import KHopGraph, save_graphs, load_graphs

from dataloader import load_data
from utils import get_training_config

EPS = 1e-12  # for nan


def select_topk_star_debug(xs, xs_ids, x0, x0_id):
    x0 = (x0, x0_id)
    xs = list(zip(xs, xs_ids))

    feature_id = 0

    up = 0
    down = torch.pow(x0[0][feature_id], 2) + EPS  # element wise
    best = up / down
    greedy = lambda xi: - (x0[0][feature_id] - xi[0][feature_id]) ** 2 / (xi[0][feature_id] ** 2)
    xs_sorted = sorted(xs, key=greedy)
    nbs = []
    for i, xi in enumerate(xs_sorted):
        tmp_up = (x0[0][feature_id] - xi[0][feature_id]) ** 2
        tmp_down = xi[0][feature_id] ** 2 + EPS
        if best < tmp_up / tmp_down:
            up += tmp_up
            down += tmp_down
            best = up / down
            nbs.append(xi[1])  # sotre the id
        else:
            break
    return


def select_all_khop(star_khop_graph_big, central_node_id, khop, select_topk):
    pres = star_khop_graph_big.predecessors(central_node_id)  # in edges
    sucs = star_khop_graph_big.successors(central_node_id)  # out edges
    node_ids = torch.unique(torch.cat([pres, sucs], dim=0))
    nbs = torch.unique(node_ids)
    weights = torch.ones(nbs.shape[0], 1)
    weights[:-1, 0] = weights[:-1, 0] / (nbs.shape[0] + EPS)
    return nbs, weights


def select_rand_khop(star_khop_graph_big, central_node_id, khop, select_topk):
    pres = star_khop_graph_big.predecessors(central_node_id)  # in edges
    sucs = star_khop_graph_big.successors(central_node_id)  # out edges
    node_ids = torch.unique(torch.cat([pres, sucs], dim=0))
    idx = torch.randperm(node_ids.shape[0])
    nbs = node_ids[idx[:100]]
    weights = torch.ones(nbs.shape[0], 1)
    weights[:-1, 0] = weights[:-1, 0] / (nbs.shape[0] + EPS)
    return nbs, weights


def select_topk_star_normft(star_khop_graph_big, node_ids, central_node_id):
    h_xs, id_xs, h_x0, id_x0 = star_khop_graph_big.ndata['feature_normed'][node_ids], node_ids, \
    star_khop_graph_big.ndata['feature_normed'][central_node_id], central_node_id

    xs = list(zip(h_xs, id_xs))
    x0 = (h_x0, id_x0)

    up = 0
    down = torch.pow(x0[0], 2) + EPS  # element wise
    best = up / down
    greedy = lambda xi: - (x0[0] - xi[0]) ** 2 / (xi[0] ** 2)
    xs_sorted = sorted(xs, key=greedy)
    nbs = []
    for i, xi in enumerate(xs_sorted):
        tmp_up = (x0[0] - xi[0]) ** 2
        tmp_down = xi[0] ** 2 + EPS
        if best < tmp_up / tmp_down:
            up += tmp_up
            down += tmp_down
            best = up / down
            nbs.append(xi[1])  # sotre the id
        else:
            break
    return nbs


def select_topk_star_unionft(star_khop_graph_big, node_ids, central_node_id):
    h_xs, id_xs, h_x0, id_x0 = star_khop_graph_big.ndata['feature'][node_ids], node_ids, \
    star_khop_graph_big.ndata['feature'][central_node_id], central_node_id

    nbs = set()
    for feature_id in range(h_xs.shape[1]):
        xs = list(zip(h_xs[:, feature_id], id_xs))
        x0 = (h_x0[feature_id], id_x0)

        up = 0
        down = torch.pow(x0[0], 2)  # element wise
        best = up / down
        greedy = lambda xi: - (x0[0] - xi[0]) ** 2 / (xi[0] ** 2)
        xs_sorted = sorted(xs, key=greedy)

        for i, xi in enumerate(xs_sorted):
            tmp_up = (x0[0] - xi[0]) ** 2
            tmp_down = xi[0] ** 2
            if best < tmp_up / tmp_down:
                up += tmp_up
                down += tmp_down
                best = up / down
                nbs.add(xi[1])  # sotre the id
    return list(nbs)


def get_star_topk_nbs(star_khop_graph_big, central_node_id, khop, select_topk):
    star_khop_graph_in = star_khop_graph_big.sample_neighbors([central_node_id], fanout=-1, edge_dir='in')
    star_khop_graph_out = star_khop_graph_big.sample_neighbors([central_node_id], fanout=-1, edge_dir='out')

    node_ids_in = star_khop_graph_in.edges()[0]
    node_ids_out = star_khop_graph_out.edges()[1]
    node_ids = torch.cat([node_ids_in, node_ids_out], dim=0)
    node_ids = torch.unique(node_ids)

    nbs = select_topk(star_khop_graph_big, node_ids, central_node_id)
    nbs.append(torch.tensor(central_node_id).long())  # make sure self is added
    weights = torch.ones(len(nbs), 1) * 0.5
    weights[:-1, 0] = weights[:-1, 0] / (len(nbs) + EPS)

    return nbs, weights


def get_convtree_topk_nbs_norm(graph_whole, xi, khop, select_topk):
    '''
        return topk neighbors weight matrix in Conv Tree graph setting
    '''
    # find all 1st-order neighbours
    pres = graph_whole.predecessors(xi)  # in edges
    sucs = graph_whole.successors(xi)  # out edges
    nbs_xi = torch.unique(torch.cat([pres, sucs], dim=0))  # FIXME: if all bidirected, delete this for performance
    if nbs_xi.shape[0] == 0:
        # no neighbours
        return tuple([xi]), tuple([1.0])
    # some refrences for help
    xf = graph_whole.ndata['feature_normed']
    Pij = {}
    Pik = {}
    Pij_tmp = {}
    Smaxj_list = []
    quant = lambda x: - x[1] / x[2]
    for xj in nbs_xi:
        # clear tmp ik for j
        Pik_tmp = {}
        # add parent edge
        aj = (xf[xj] - xf[xi]) ** 2
        bj = (xf[xj]) ** 2
        Smaxj = aj / bj
        # get xj's neighbours
        pres = graph_whole.predecessors(xj)  # in edges
        sucs = graph_whole.successors(xj)  # out edges
        nbs_xj = torch.unique(torch.cat([pres, sucs], dim=0))
        if nbs_xj.shape[0] == 0:
            # xj no neighbours
            Pij_tmp[xj] = 0.5  # 1/2
        else:
            Pij_tmp[xj] = 0.25  # 1/4, because it has to avg with sons
            num_hop2 = 0  # how many sons has been selected, could be 0?
            ss = [(xk.item(), (xf[xk] - xf[xj]) ** 2, xf[xk] ** 2) for xk in nbs_xj]  # store in (k, ak,bk) form
            ss.sort(key=lambda x: -x[1] / x[2])  # from big to small ak/bk
            # loop to find the optimal value
            for xk, ak, bk in ss:
                if ak / bk > Smaxj:
                    num_hop2 += 1
                    # update the best sons
                    aj += ak
                    bj += bk
                    Smaxj = aj / bj
                    Pik_tmp[xk] = 0.25  # Pik_tmp[xk] = 1/4
                else:
                    # the rest is impossible to make the ans bigger
                    break
            if num_hop2 != 0:
                # update all Pik_tmp
                for xk in Pik_tmp:
                    Pik_tmp[xk] /= num_hop2
                    # add to global Pik
                    if xk in Pik:
                        Pik[xk] += Pik_tmp[xk]
                    else:
                        Pik[xk] = Pik_tmp[xk]
            else:
                Pij_tmp[xj] = 0.5
        Smaxj_list.append((xj, aj, bj))  # j, aj, bj
    Smaxj_list = sorted(Smaxj_list, key=lambda x: -x[1] / x[2])  # from big to small
    ai = Smaxj_list[0][1]
    bi = Smaxj_list[0][2]
    RQ_max = ai / bi  # at least the largest one should be selected
    num_hop1 = 1
    for xj, aj, bj in Smaxj_list[1:]:  # check the rest
        if aj / bj > RQ_max:
            num_hop1 += 1
            ai += aj
            bi += bj
            RQ_max = ai / bi
            Pij[xj] = Pij_tmp[xj]  # select xj
        else:
            break

    for xj in Pij:
        # update all Pij
        Pij[xj] /= num_hop1
    Pij[xi] = 0.5  # self loop

    Pfinal = {k: v for k, v in Pij.items()}
    for k, v in Pik.items():
        if k in Pfinal:
            Pfinal[k] += v
        else:
            Pfinal[k] = v

    adj_list, weight_list = tuple(Pfinal.keys()), tuple(Pfinal.values())

    return adj_list, weight_list


def mrqsample(dataset, graph, features, one_hop=True):
    sp_matrix_graphs_filename = f"./mrqs/{dataset}.{1 if one_hop else 2}hop.sp_matrix"
    if os.path.exists(sp_matrix_graphs_filename):
        sp_matrix_graph_list, _ = load_graphs(sp_matrix_graphs_filename)
    else:
        print(f'MRQSampling {dataset} ...')
        st = time.time()
        if one_hop:
            k_hop = 1
            get_sp_adj_list = get_star_topk_nbs
            select_topk_fn = select_topk_star_normft
        else:
            k_hop = 2
            get_sp_adj_list = get_convtree_topk_nbs_norm
            select_topk_fn = None

        graph.ndata['feature_normed'] = features
        # norm it
        graph.ndata['feature_normed'] -= graph.ndata['feature_normed'].min(0, keepdim=True)[0]
        graph.ndata['feature_normed'] /= graph.ndata['feature_normed'].max(0, keepdim=True)[0] + EPS
        graph.ndata['feature_normed'] = torch.norm(graph.ndata['feature_normed'], dim=1)

        sp_matrix_graph = dgl.graph(([], []))
        sp_matrix_graph.add_nodes(graph.num_nodes())  # keep the node num same

        transform = KHopGraph(k_hop)
        tmp_graph = transform(graph)
        tmp_graph = tmp_graph.to_simple()
        tmp_graph = tmp_graph.remove_self_loop()

        for central_node_id in graph.nodes():
            adj_list, weight_list = get_sp_adj_list(tmp_graph, central_node_id.item(), k_hop, select_topk_fn)
            sp_matrix_graph.add_edges(adj_list, central_node_id.long())
            # sp_matrix_graph.add_edges(adj_list, central_node_id.long(),
            #                           {'pw': torch.tensor(weight_list)})  # adj_list->node_id, edata['pw'] = weights
        print(f'Done in {time.time()-st:.2f} seconds.')
        sp_matrix_graph_list = [sp_matrix_graph]
        save_graphs(sp_matrix_graphs_filename, sp_matrix_graph_list)
    return sp_matrix_graph_list[0]


if __name__ == '__main__':
    # for dataset in ['reddit', 'weibo', 'amazon', 'yelp', 'tfinance',
    #                              'elliptic', 'tolokers', 'questions', 'dgraphfin', 'tsocial']:
    for dataset in ['reddit']:
        conf = get_training_config(dataset, config_path='train.conf.yaml')
        G, X, Y, train_masks, val_masks, test_masks = load_data(dataset, semi=True, feat_trans=conf['feat_trans'])
        mrqsample(dataset, G, X, one_hop=True)