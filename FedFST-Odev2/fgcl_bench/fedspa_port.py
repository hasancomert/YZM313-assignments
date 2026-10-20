"""FedSPA (CVPR 2025; OakleyTan/FedSPA @ d2b3ddb): SFPD + HBDA'nın harness'e port'u.

Orijinal kodun ilgili parçaları buraya kopyalandı; değişen yerler "[port]" ile işaretli.
Kaynak dosyalar (Code/ altında):
  benchmark001/base.py            : Seq, MLP_spa                                   (aynen)
  benchmark001/config.py          : freeze_parameters, normalize_tensor, mp_func_spa,
                                    agg_func_spa, GraphConvolution, ACM, get_spa
  benchmark001/model/default_model.py : Model (get_spa(in, 128, out, 2))
  flgo/utils/fmodule.py           : FModule.has_nan                                (aynen)
  flgo/algorithm/fedbase.py       : calculate_shigh                                (aynen)
  flgo/algorithm/fedspa.py        : fish_cal, calculate_fisher_average, normalize_list,
                                    sum_of_layer_fisher_differences, normalize,
                                    calculate_weights (aynen), Server.aggregate, Client.train
  flgo/experiment/logger/simple_logger.py : adj_low_un / adj_low / adj_high (log_once)
  flgo/benchmark/toolkits/graph/node_classification/__init__.py : GeneralCalculator.compute_loss
Değişiklikler:
  1. DGL: graph.update_all(mp_func_spa, agg_func_spa) -> spa_update_all (index_add_).
     Kenar türü (0: iki uç da etiketli ve aynı sınıf, 1: etiketli ve farklı sınıf,
     2: en az bir uç etiketsiz) mp_func_spa'daki kuralın aynısı; her hedef düğümde
     gelen kenarların kaynak öznitelikleri türe göre toplanır (agg_func_spa).
  2. DGL: g.adjacency_matrix() -> yoğun A[src, dst] = 1 (DGL 2.1'de satır = kaynak).
  3. Sabit 'cuda' / 'cuda:0' -> device parametresi; sabit boyutlar (1703, 5) -> veri setinden.
  4. att_struc_low orijinalde ilklenmemiş bellek (torch.FloatTensor) -> sıfır. İleri geçişte
     kullanılmıyor; yalnızca rastgele bellek içeriğinin (NaN olabilir) has_nan kontrolünü
     tetiklemesini önler. RNG tüketimi değişmez.
  5. Kenarı olmayan grafikte orijinal convert_to_dglgraph boş tensörün max'ını alıp çöküyor;
     burada düğüm sayısı doğrudan x'ten alınır.
"""
import collections
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.nn.norm import GraphNorm


# =============================================================== benchmark001/base.py (aynen)
class Seq(nn.Module):
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


class MLP_spa(nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 dropout=0,
                 tail_activation=False,
                 activation=nn.ReLU(inplace=True),
                 gn=False):
        super().__init__()
        modlist = []
        self.seq = None
        if num_layers == 1:
            modlist.append(nn.Linear(input_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)
        else:
            modlist.append(nn.Linear(input_channels, hidden_channels))
            for _ in range(num_layers - 2):
                if gn:
                    modlist.append(GraphNorm(hidden_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
                modlist.append(nn.Linear(hidden_channels, hidden_channels))
            if gn:
                modlist.append(GraphNorm(hidden_channels))
            if dropout > 0:
                modlist.append(nn.Dropout(p=dropout, inplace=True))
            modlist.append(activation)
            modlist.append(nn.Linear(hidden_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)

    def forward(self, x):
        return self.seq(x)


# =============================================================== benchmark001/config.py
def freeze_parameters(parameters):
    for param in parameters:
        if isinstance(param, nn.Parameter):
            param.requires_grad = False


def normalize_tensor(mx, eqvar=None):
    rowsum = torch.sum(mx, 1)
    if eqvar:
        r_inv = torch.pow(rowsum, -1 / eqvar).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx


class SPAGraph:
    """[port] ACM.preprocess'teki DGL bloğunun yerine: kenar listesi + label_unk + kenar türü.

    Orijinal: g = dgl.graph((src, dst)); g.ndata['label_unk'] = -1 (eğitim düğümlerinde etiket);
    block = dgl.to_block(g, dst_nodes=arange(N)). Kenar türü mp_func_spa'daki kuralla hesaplanır.
    """

    def __init__(self, edge_index, label_unk, num_nodes):
        self.src = edge_index[0]
        self.dst = edge_index[1]
        self.label_unk = label_unk
        self.num_nodes = num_nodes
        # mp_func_spa (aynı kural)
        src_label = label_unk[self.src]
        dst_label = label_unk[self.dst]
        src_fake_label = torch.zeros_like(src_label)
        src_fake_label[(src_label < 0) | (dst_label < 0)] = 2
        condition = (src_label != dst_label) & (src_label >= 0) & (dst_label >= 0)
        src_fake_label[condition] = 1
        self.src_fake_label = src_fake_label


def spa_update_all(graph, h_low, h_high):
    """[port] graph.update_all(mp_func_spa, agg_func_spa) eşdeğeri.

    mp_func_spa her kenar için kaynağın h_low / h_high'ını mesaj yapar (DGL bunu tüm kenarlar
    için index_select ile toplar). agg_func_spa her hedef düğümde gelen mesajları kenar türü
    maskesiyle çarpıp toplar: neigh_hete_* (tür 1), neigh_homo_* (tür 0), neigh_unk_* (tür 2).
    DGL gelen kenarları kenar numarası sırasıyla dizer; index_add_ de kenarları aynı sırayla
    toplar. Gelen mesajı olmayan düğüm DGL'de sıfır alır; index_add_ de sıfırdan başlar."""
    m_low = torch.index_select(h_low, 0, graph.src)
    m_high = torch.index_select(h_high, 0, graph.src)
    out = {}
    for name, code in (("hete", 1), ("homo", 0), ("unk", 2)):
        mask = (graph.src_fake_label == code).unsqueeze(-1)
        for tag, m in (("low", m_low), ("high", m_high)):
            agg = torch.zeros(graph.num_nodes, m.size(1), dtype=m.dtype, device=m.device)
            out[f"neigh_{name}_{tag}"] = agg.index_add(0, graph.dst, m * mask)
    return out


def get_spa(nfeat,
        nhid,
        nclass,
        nlayers,
        device=None):
    return ACM(nfeat,
        nhid,
        nclass,
        nlayers,
        dropout=0.1,
        model_type='acm_spa',
        variant=False,
        init_layers_X=1,
        device=device)


class GraphConvolution(Module):
    def __init__(
        self,
        in_features,
        out_features,
        model_type,
        output_layer=0,
        variant=False,
        device=None,  # [port] orijinalde modül düzeyinde "cuda:0 if available else cpu"
    ):
        super(GraphConvolution, self).__init__()
        device = device if device is not None else torch.device("cpu")
        self.device = device
        (
            self.in_features,
            self.out_features,
            self.output_layer,
            self.model_type,
            self.variant,
        ) = (
            in_features,
            out_features,
            output_layer,
            model_type,
            variant,
        )
        self.bn_low = nn.BatchNorm1d(out_features)
        self.bn_high = nn.BatchNorm1d(out_features)
        self.bn_mlp = nn.BatchNorm1d(out_features)
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        self.weight_low, self.weight_high, self.weight_mlp = (
            Parameter(init.xavier_uniform_(torch.FloatTensor(in_features, out_features).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(in_features, out_features).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(in_features, out_features).to(device))),
        )
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
        )
        self.layer_norm_low, self.layer_norm_high, self.layer_norm_mlp = (
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
        )
        self.layer_norm_struc_low, self.layer_norm_struc_high = nn.LayerNorm(
            out_features
        ), nn.LayerNorm(out_features)
        self.att_struc_low = Parameter(
            torch.zeros(1 * out_features, 1).to(device)  # [port] orijinal: ilklenmemiş torch.FloatTensor
        )
        self.att_vec_3 = init.xavier_uniform_(Parameter(torch.FloatTensor(3, 3).to(device)))
        self.att_vec_low_fr, self.att_vec_high_fr, self.att_vec_low_be, self.att_vec_high_be, self.att_vec_low_unk, self.att_vec_high_unk = (
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
        )

        self.att_vec = init.xavier_uniform_(Parameter(torch.FloatTensor(7, 7).to(device)))

        self.fc_balance = MLP_spa(out_features, hidden_channels=128, output_channels=1,
                              num_layers=1)
        self.balance_w = nn.Sigmoid()
        self.fr_low_g = None
        self.fr_high_g = None
        self.mlp_g = None

    def attention7(self, neigh_hete_low, neigh_hete_high, neigh_homo_low, neigh_homo_high, neigh_unk_low, neigh_unk_high,
                   output_mlp, balance_1=None, balance_2=None, balance_3=None):
        T = 7
        if balance_1 != None:
            logits = (
                    torch.mm(
                        torch.sigmoid(
                            torch.cat(
                                [
                                    torch.mm(neigh_hete_low, self.att_vec_low_fr),
                                    torch.mm(neigh_hete_high, self.att_vec_high_fr),
                                    torch.mm(neigh_homo_low, self.att_vec_low_be),
                                    torch.mm(neigh_homo_high, self.att_vec_high_be),
                                    torch.mm(neigh_unk_low, self.att_vec_low_fr) * balance_1 + torch.mm(neigh_unk_low,
                                                                                                        self.att_vec_low_be) * (
                                                1 - balance_1),
                                    torch.mm(neigh_unk_high, self.att_vec_high_fr) * balance_2 + torch.mm(neigh_hete_high,
                                                                                                          self.att_vec_high_be) * (
                                                1 - balance_2),
                                    torch.mm(output_mlp, self.att_vec_mlp),
                                ],
                                1,
                            )
                        ),
                        self.att_vec,
                    )
                    / T
            )
        else:
            logits = (
                    torch.mm(
                        torch.sigmoid(
                            torch.cat(
                                [
                                    torch.mm(neigh_hete_low, self.att_vec_low_fr),
                                    torch.mm(neigh_hete_high, self.att_vec_high_fr),
                                    torch.mm(neigh_homo_low, self.att_vec_low_be),
                                    torch.mm(neigh_homo_high, self.att_vec_high_be),
                                    torch.mm(neigh_unk_low, self.att_vec_low_unk),
                                    torch.mm(neigh_unk_high, self.att_vec_high_unk),
                                    torch.mm(output_mlp, self.att_vec_mlp),
                                ],
                                1,
                            )
                        ),
                        self.att_vec,
                    )
                    / T
            )
        att = torch.softmax(logits, 1)
        return (att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None],
                att[:, 3][:, None], att[:, 4][:, None], att[:, 5][:, None], att[:, 6][:, None])

    def forward(self, input, adj_low, adj_high, adj_low_unnormalized, graph=None, fr_high_g=None, fr_low_g=None, mlp_g=None):
        device = self.device  # [port]

        output_low = F.relu(torch.spmm(adj_low, (torch.mm(input, self.weight_low))))
        output_high = F.relu(torch.spmm(adj_high, (torch.mm(input, self.weight_high))))
        output_mlp = F.relu(torch.mm(input, self.weight_mlp))

        # (yalnızca model_type == "acm_spa" yolu kullanılıyor)
        dstdata = spa_update_all(graph, output_low, output_high)  # [port] graph.update_all(mp_func_spa, agg_func_spa)

        neigh_hete_low = dstdata["neigh_hete_low"].to(device)
        neigh_hete_high = dstdata["neigh_hete_high"].to(device)
        neigh_homo_low = dstdata["neigh_homo_low"].to(device)
        neigh_homo_high = dstdata["neigh_homo_high"].to(device)
        neigh_unk_low = dstdata["neigh_unk_low"].to(device)
        neigh_unk_high = dstdata["neigh_unk_high"].to(device)

        if fr_high_g != None:
            balance_1 = self.balance_w(self.fc_balance(neigh_unk_low)).to(device)
            balance_2 = self.balance_w(self.fc_balance(neigh_unk_high)).to(device)
            self.att_fr_low, self.att_fr_high, self.att_be_low, self.att_be_high, self.att_unk_low, self.att_unk_high, self.att_mlp = self.attention7(
                neigh_hete_low, neigh_hete_high, neigh_homo_low, neigh_homo_high, neigh_unk_low, neigh_unk_high, output_mlp, balance_1, balance_2
            )

            return 7 * (
                    self.att_fr_low * neigh_hete_low +
                    self.att_fr_high * neigh_hete_high +
                    self.att_be_low * neigh_homo_low +
                    self.att_be_high * neigh_homo_high +
                    self.att_unk_low * neigh_unk_low +
                    self.att_unk_high * neigh_unk_high +
                    self.att_mlp * output_mlp
            )

        self.att_fr_low, self.att_fr_high, self.att_be_low, self.att_be_high, self.att_unk_low, self.att_unk_high, self.att_mlp = self.attention7(
            neigh_hete_low, neigh_hete_high, neigh_homo_low, neigh_homo_high, neigh_unk_low, neigh_unk_high, output_mlp
        )

        return 7 * (
                self.att_fr_low * neigh_hete_low +
                self.att_fr_high * neigh_hete_high +
                self.att_be_low * neigh_homo_low +
                self.att_be_high * neigh_homo_high +
                self.att_unk_low * neigh_unk_low +
                self.att_unk_high * neigh_unk_high +
                self.att_mlp * output_mlp
        )


class ACM(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        nlayers,
        dropout,
        model_type,
        variant=False,
        init_layers_X=1,
        device=None,  # [port]
    ):
        super(ACM, self).__init__()
        device = device if device is not None else torch.device("cpu")
        self.bn = nn.BatchNorm1d(nhid)
        self.preprocessed = False
        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.model_type, self.nlayers = (model_type, nlayers)
        self.gcns.append(
            GraphConvolution(
                nfeat,
                nhid,
                model_type=model_type,
                variant=variant,
                device=device,
            )
        )
        self.gcns.append(
            GraphConvolution(
                1 * nhid,
                nclass,
                model_type=model_type,
                output_layer=1,
                variant=variant,
                device=device,
            )
        )

        self.dropout = dropout
        self.fea_param, self.xX_param = init.xavier_uniform_(Parameter(
            torch.FloatTensor(1, 1).to(device))
        ), init.xavier_uniform_(Parameter(torch.FloatTensor(1, 1).to(device)))

    def preprocess(self, data):
        # [port] convert_to_dglgraph + assign_pseudo_labels_to_dglgraph + create_block_from_dglgraph
        num_nodes = data.x.size(0)
        if data.edge_index.numel() > 0:
            num_nodes = max(num_nodes, int(data.edge_index.max()) + 1)
        train_nid = torch.nonzero(data.train_mask, as_tuple=False).squeeze()
        labels = data.y[train_nid]
        label_unk = (torch.ones(num_nodes, device=data.y.device) * -1).long()
        label_unk[train_nid] = labels.long()
        self.block = SPAGraph(data.edge_index, label_unk, num_nodes)
        self.h = data.x

    def forward(self, data, adj_low_un, adj_low, adj_high, fr_low_g=None, fr_high_g=None, mlp_g=None):
        if fr_low_g !=None:
            freeze_parameters(fr_low_g)
            freeze_parameters(fr_low_g)
            mlp_g
        if self.preprocessed:
            block = self.block
            h = self.h
        else:
            self.preprocess(data)
            block = self.block
        x = data.x
        x = F.dropout(x, self.dropout, training=self.training)

        if fr_high_g != None:
            fea1 = self.gcns[0](x, adj_low, adj_high, adj_low_un, block, fr_high_g[0], fr_low_g[0], mlp_g[0])
        else:
            fea1 = self.gcns[0](x, adj_low, adj_high, adj_low_un, block)

        fea1 = F.relu(fea1)
        fea1 = F.dropout(fea1, self.dropout, training=self.training)
        if fr_high_g != None:
            fea2 = self.gcns[1](fea1, adj_low, adj_high, adj_low_un, block, fr_high_g[1], fr_low_g[1], mlp_g[1])
        else:
            fea2 = self.gcns[1](fea1, adj_low, adj_high, adj_low_un, block)
        return F.log_softmax(fea2, dim=1)


class Model(nn.Module):
    """benchmark001/model/default_model.py: Model (FModule). [port] in_size/out_size veri setinden."""

    def __init__(self, in_size, out_size, hid_size=128, device=None):
        super().__init__()
        self.model = get_spa(in_size, hid_size, out_size, 2, device=device)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def has_nan(self):  # flgo/utils/fmodule.py: FModule.has_nan (aynen)
        for p in self.parameters():
            if torch.any(torch.isnan(p)).item():
                return True
        return False


# =============================================================== simple_logger.py: log_once
def compute_adjacencies(data, device):
    """[port] SimpleLogger.log_once'taki adj_low_un / adj_low / adj_high hesabı (DGL'siz).

    Orijinal:
        g = convert_to_dglgraph(data); adj_low_unnormalized = g.adjacency_matrix()
        adj_low = normalize_tensor(I + adj_low_unnormalized.to_dense())
        adj_high = (I - adj_low).to_sparse()
    DGL 2.1'de adjacency_matrix() satırı kaynak, sütunu hedef olan seyrek matris döndürür
    (A[src, dst] = 1; yerel ortamda DGL 2.1.0 ile doğrulandı)."""
    num_nodes = data.x.size(0)
    if data.edge_index.numel() > 0:
        num_nodes = max(num_nodes, int(data.edge_index.max()) + 1)
    edge_index = data.edge_index.to(device)
    values = torch.ones(edge_index.size(1), device=device)
    adj_low_unnormalized = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()
    eye_matrix = torch.eye(num_nodes).to(device)
    adj_low = (normalize_tensor(eye_matrix + adj_low_unnormalized.to_dense())).to(device)
    adj_high = (torch.eye(num_nodes).to(device) - adj_low).to(device).to_sparse()
    return adj_low_unnormalized, adj_low, adj_high


# =============================================================== flgo/algorithm/fedbase.py (aynen)
def calculate_shigh(data, sampling_rate=0.2):
    features = data.x
    num_nodes, num_features = features.shape
    features = features.cpu().numpy()
    edge_index = data.edge_index.cpu().numpy()
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i, j in zip(edge_index[0], edge_index[1]):
        adj_matrix[i, j] = 1
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian_matrix = degree_matrix - adj_matrix
    sampled_indices = np.random.choice(num_features, size=int(sampling_rate * num_features), replace=False)
    S_high_values = []
    for idx in sampled_indices:
        x_k = features[:, idx]
        numerator = x_k.T @ laplacian_matrix @ x_k
        denominator = x_k.T @ x_k
        if denominator != 0:
            S_high = numerator / denominator
            if not np.isnan(S_high):
                S_high_values.append(S_high)

    if S_high_values:
        mean_shigh = np.mean(S_high_values)
    else:
        mean_shigh = np.nan

    return mean_shigh


# =============================================================== flgo/algorithm/fedspa.py (aynen)
def sum_of_layer_fisher_differences(fish_dict1, fish_dict2, layer):
    total_difference = 0.0
    for key in fish_dict1:
        if layer in key:
            difference = fish_dict1[key] - fish_dict2[key]
            total_difference += torch.sum(torch.abs(difference))
    return total_difference

def normalize_list(values):
    total = sum(values)
    if total == 0:
        raise ValueError("Sum of the elements is zero, cannot normalize.")
    normalized_values = [value / total for value in values]
    return normalized_values


def calculate_fisher_average(fisher_dicts):
    if not fisher_dicts:
        raise ValueError("The list of Fisher dictionaries is empty.")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fisher_avg = {}
    for key in fisher_dicts[0].keys():
        fisher_avg[key] = torch.zeros_like(fisher_dicts[0][key], device=device)

    for fish_dict in fisher_dicts:
        for key in fish_dict:
            fisher_avg[key] += fish_dict[key].to(device)
    num_dicts = len(fisher_dicts)
    for key in fisher_avg:
        fisher_avg[key] /= num_dicts

    return fisher_avg


def normalize(data):
    min_val = min(data)
    max_val = max(data)
    range = max_val - min_val
    normalized_data = [(x - min_val) / range for x in data]
    return normalized_data


def calculate_weights(differences):
    float_diff = [x.item() if isinstance(x, torch.Tensor) else x for x in differences]
    normalized_diff = normalize(float_diff)
    pre_weights = [np.exp(-x) for x in normalized_diff]
    sum_weights = sum(pre_weights)
    weights = [w / sum_weights for w in pre_weights]
    return weights

def fish_cal(net, data, adj_low_un, adj_low, adj_high, high_list, low_list, mlp_list):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    label = data.y
    data, label = data.to(device), label.to(device)
    output = net(data, adj_low_un, adj_low, adj_high, high_list, low_list, mlp_list)
    pre = F.log_softmax(output, dim=1)
    log_likelihoods = pre.gather(1, label.view(-1, 1)).squeeze()
    net.zero_grad()
    log_likelihoods.mean().backward()
    average_grads = {name: param.grad.clone() for name, param in net.named_parameters() if param.grad is not None}
    fish_dict = {name: grad ** 2 for name, grad in average_grads.items()}
    return fish_dict


class _TrainData:
    """flgo TaskDataset'in kullanılan kısmı: .data (grafik, maskelerle)."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return int(self.data.train_mask.numel())


class FedSPAClient:
    """flgo/algorithm/fedspa.py: Client'ın eğitimle ilgili kısmı.

    [port] Ortak FGCL ayarı: optimizer Adam (flgo'daki get_optimizer'ın 'adam' dalı: yalnızca
    requires_grad parametreler, lr ve weight_decay), num_steps = yerel epoch sayısı; her adım
    tüm görev grafiği üzerinde (orijinal yükleyici de tek grafiği tek "batch" olarak veriyor)."""

    def __init__(self, cid, learning_rate, weight_decay, num_steps):
        self.id = cid
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_steps = num_steps
        self.criterion = torch.nn.NLLLoss()
        self.model_g = None
        self.fr_low_g_list = None
        self.fr_high_g_list = None
        self.mlp_list = None
        self.train_data = None
        self.adj_low_un = self.adj_low = self.adj_high = None
        self.radio_diff = None

    @property
    def datavol(self):
        return len(self.train_data)

    def set_task(self, data, adjs):
        self.train_data = _TrainData(data)
        self.adj_low_un, self.adj_low, self.adj_high = adjs

    def get_optimizer(self, model, lr, weight_decay):
        filter_fn = filter(lambda p: p.requires_grad, model.parameters())
        return torch.optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)

    def compute_loss(self, model, data, adj_low_un, adj_low, adj_high, fr_low_g_list, fr_high_g_list, mlp_list):
        # GeneralCalculator.compute_loss (aynen)
        tdata = data
        valid_mask = (tdata.y >= 0) & tdata.train_mask
        outputs = model(tdata, adj_low_un, adj_low, adj_high, fr_high_g_list, fr_low_g_list, mlp_list)
        if valid_mask.sum().item() == 0:
            return {'loss': torch.tensor(0.0)}
        loss = self.criterion(outputs[valid_mask], tdata.y[valid_mask])
        return {'loss': loss}

    def train(self, model):
        model.train()

        optimizer = self.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.model_g is not None:
            for param in self.model_g.parameters():
                param.requires_grad = False

        self.fr_low_g_list = []
        self.fr_high_g_list = []
        self.mlp_list = []

        for iter in range(self.num_steps):
            if iter == 0:
                for gcn_layer in model.model.gcns:
                    fr_low_g = gcn_layer.att_vec_low_unk
                    self.fr_low_g_list.append(fr_low_g)

                    fr_high_g = gcn_layer.att_vec_high_unk
                    self.fr_high_g_list.append(fr_high_g)

                    mlp_g = gcn_layer.att_vec_mlp
                    self.mlp_list.append(mlp_g)
            batch_data = self.train_data.data
            model.zero_grad()
            loss = self.compute_loss(model, batch_data, self.adj_low_un, self.adj_low, self.adj_high, self.fr_low_g_list, self.fr_high_g_list, self.mlp_list)['loss']
            loss.backward()
            optimizer.step()
        return loss.item()


class FedSPAServer:
    """flgo/algorithm/fedspa.py: Server.aggregate (HBDA). [port] logger -> print."""

    def __init__(self, model, clients):
        self.model = model
        self.clients = clients
        self.received_clients = list(range(len(clients)))

    def new_task(self):
        # [port] HBDA'nın homofili ağırlıkları (diffs/weights2) orijinalde eğitimin başında bir kez
        # hesaplanıyor; burada her görevin başında, o görevin alt-grafiğiyle yeniden hesaplanır.
        if hasattr(self, "diffs"):
            del self.diffs
        if hasattr(self, "weights2"):
            del self.weights2

    def aggregate(self, models: list, *args, **kwargs):
        if len(models) == 0: return self.model

        nan_exists = [m.has_nan() for m in models]
        if any(nan_exists):
            if all(nan_exists): raise ValueError("All the received local models have parameters of nan value.")

            print('Warning("There exists nan-value in local models, which will be automatically removed from the aggregation list.")')

            new_models = []
            received_clients = []

            for ni, mi, cid in zip(nan_exists, models, self.received_clients):
                if ni: continue
                new_models.append(mi)
                received_clients.append(cid)
            self.received_clients = received_clients
            models = new_models

        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)

        fishs = []
        for model, cid in zip(models, self.received_clients):
            fishs.append(fish_cal(model, self.clients[cid].train_data.data, self.clients[cid].adj_low_un,
                                  self.clients[cid].adj_low, self.clients[cid].adj_high,
                                  self.clients[cid].fr_high_g_list, self.clients[cid].fr_low_g_list,
                                  self.clients[cid].mlp_list))

        fish_g = calculate_fisher_average(fishs)

        aggregated_params = collections.defaultdict(lambda: 0)
        if not hasattr(self, 'diffs'):
            self.diffs = []
            for client in self.clients:
                self.diffs.append(abs(client.radio_diff))
            self.weights2 = calculate_weights(self.diffs)
        fisher_weights = {}
        for layer in ["fr", "be"]:
            layer_fish_diffs = []
            for fish in fishs:
                layer_fish_diffs.append(sum_of_layer_fisher_differences(fish, fish_g, layer))
            layer_fish_diffs = normalize_list(layer_fish_diffs)
            fisher_weights[layer] = calculate_weights(layer_fish_diffs)

        for i, model in enumerate(models):
            for name, param in model.named_parameters():
                if any(layer in name for layer in fisher_weights.keys()):
                    for layer in fisher_weights.keys():
                        if layer in name:
                            aggregated_params[name] += param.data * fisher_weights[layer][i]
                else:
                    aggregated_params[name] += param.data * self.weights2[i]
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_params:
                    param.data.copy_(aggregated_params[name])

        self.last_weights = dict(homophily=[float(w) for w in self.weights2],
                                 fr=[float(w) for w in fisher_weights["fr"]],
                                 be=[float(w) for w in fisher_weights["be"]])
        return self.model
