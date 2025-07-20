import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn


# Refer to https://github.com/ChenJY-Count/PolyGCL/blob/master/ChebnetII_pro.py
def precompute_Tx(g, x, K, conv_norm='none'):
    """
    Precompute the Chebyshev polynomial propagation results T_0(x), ..., T_K(x)

    Args:
        g (dgl.DGLGraph): Input graph
        x (torch.Tensor): Node features of shape [N, F]
        K (int): Order of the Chebyshev polynomial
        conv_norm (str): Normalization type ('none', 'batch', or 'feature')
        lap_edge_weight (Tensor, optional): Precomputed Laplacian edge weights

    Returns:
        Tx_list (List[Tensor]): A list of length K+1 with each T_k(x)
        lap_edge_weight (Tensor): The Laplacian edge weights used
    """

    def conv_normalize(x, conv_norm):
        if conv_norm != 'none':
            dim = 0 if conv_norm == 'batch' else 1
            x = (x - x.mean(dim, keepdim=True)) / (x.std(dim, unbiased=False, keepdim=True) + 1e-5)
        return x

    g = g.local_var()

    deg = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    g.ndata['norm'] = norm

    def norm_edge_weight(edges):
        return {'lap_edge_weight': edges.src['norm'] * edges.dst['norm']}

    g.apply_edges(norm_edge_weight)
    g.edata['lap_edge_weight'] = -g.edata['lap_edge_weight']

    def cheb_propagate(g, x):
        g.ndata['h'] = x
        g.update_all(fn.u_mul_e('h', 'lap_edge_weight', 'm'), fn.sum('m', 'h'))
        return conv_normalize(g.ndata['h'], conv_norm)

    Tx_list = []
    Tx_0 = x
    Tx_list.append(Tx_0)

    if K == 0:
        return Tx_list

    Tx_1 = cheb_propagate(g, x)
    Tx_list.append(Tx_1)

    for _ in range(2, K + 1):
        Tx_2 = cheb_propagate(g, Tx_1)
        Tx_2 = 2 * Tx_2 - Tx_0
        Tx_list.append(Tx_2)
        Tx_0, Tx_1 = Tx_1, Tx_2

    return Tx_list


class SAGAD(nn.Module):
    def __init__(self, in_dim, hid_dims, K, dropout, activation, mlp_norm, fusion):
        super(SAGAD, self).__init__()
        self.chebnet = ChebnetIIPropDGL(K)
        self.fusion = fusion
        self.mlp = MLP(in_dim, hid_dims, dropout, activation, mlp_norm)

    def forward(self, Tx_list, X, return_coefs=False):
        Z_l = self.chebnet(Tx_list, lowpass=True)
        Z_h = self.chebnet(Tx_list, lowpass=False)
        if return_coefs:
            Z, C = self.fusion(Z_l, Z_h, X, return_coefs=True)
            return self.mlp(Z), C
        else:
            Z = self.fusion(Z_l, Z_h, X)
            return self.mlp(Z)


def cheby(i, x):
    if i == 0:
        return 1.0
    elif i == 1:
        return x
    else:
        T0 = 1.0
        T1 = x
        for ii in range(2,i+1):
            T2 = 2 * x * T1-T0
            T0, T1 = T1, T2
        return T2


def presum_tensor(h, initial_val):
    length = len(h) + 1
    temp = torch.zeros(length, device=h.device, dtype=h.dtype)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx - 1] + h[idx - 1]
    return temp


def preminus_tensor(h, initial_val):
    length = len(h) + 1
    temp = torch.zeros(length, device=h.device, dtype=h.dtype)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx - 1] - h[idx - 1]
    return temp


def reverse_tensor(h):
    temp = torch.zeros_like(h)
    length = len(temp)
    for idx in range(length):
        temp[idx] = h[length - 1 - idx]
    return temp


class ChebnetIIPropDGL(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.register_buffer('initial_val_low', torch.tensor(2.0))
        self.temp_low = nn.Parameter(torch.Tensor(self.K))
        self.temp_high = nn.Parameter(torch.Tensor(self.K))
        self.register_buffer('initial_val_high', torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.temp_low, 2.0 / self.K)
        nn.init.constant_(self.temp_high, 2.0 / self.K)

    def compute_coeffs(self, lowpass):
        if lowpass:
            TEMP = F.relu(self.temp_low)
            coe_tmp = preminus_tensor(TEMP, self.initial_val_low)
        else:
            TEMP = F.relu(self.temp_high)
            coe_tmp = presum_tensor(TEMP, self.initial_val_high)

        coe = coe_tmp.clone()
        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] += coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)
        return coe
    
    def forward(self, Tx_list, lowpass=True):
        coe = self.compute_coeffs(lowpass)

        out = coe[0] / 2 * Tx_list[0]
        for i in range(1, self.K + 1):
            out = out + coe[i] * Tx_list[i]
        return out


class MLP(nn.Module):
    def __init__(
            self,
            in_dim,
            hid_dims,
            dropout,
            activation,
            norm_type='none',
            last_activate=False,
    ):
        super(MLP, self).__init__()
        self.norm_type = norm_type
        self.last_activate = last_activate

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

        hid_dims = [in_dim] + hid_dims
        for i in range(len(hid_dims) - 1):
            self.layers.append(nn.Linear(hid_dims[i], hid_dims[i + 1]))

        for i in range(len(hid_dims) - 2):
            if self.norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(hid_dims[i + 1]))
            elif self.norm_type == 'layer':
                self.norms.append(nn.LayerNorm(hid_dims[i + 1]))

    def forward(self, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(h)
            if l != len(self.layers) - 1:
                if self.norm_type != 'none':
                    h = self.norms[l](h)
                h = self.activation(h)
        if self.last_activate:
            h = self.activation(h)
        return h
