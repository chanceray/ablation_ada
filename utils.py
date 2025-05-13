import math
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
# from scipy.sparse.linalg.eigen.arpack import eigsh
from typing import Optional
import random
import sys
import os
import pdb
import torch
import torch.nn as nn
from torch_geometric.utils.num_nodes import maybe_num_nodes
import shutil
import layers
import scipy.optimize as optimize

# from dgl.data import CoraGraphDataset, KarateClubDataset, CiteseerGraphDataset, PubmedGraphDataset
# from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.model_selection import train_test_split

from normalization import fetch_normalization, row_normalize


datadir = "data"


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # preprocess feature
    features = preprocess_features(features)
    features = torch.tensor(features, dtype=torch.float32)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # preprocess adj
    adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
    # adj = torch_normalize_adj(adj)
    # adj2 = preprocess_adj(adj)
    # adj2 = sparse_mx_to_torch_sparse_tensor(adj2).to_dense()
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    _, l_num = labels.shape
    labels = torch.tensor((labels * range(l_num)).sum(axis=1), dtype=torch.int64)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y)+500))

    print(f"train: {len(idx_test)} val: {len(idx_val)} test: {len(idx_test)}")

    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    #return sparse_to_tuple(features)
    return features.todense()


def torch_normalize_adj(adj, device):
    # adj = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to(device)
    return adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    #return sparse_to_tuple(adj_normalized)
    return adj_normalized


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


# def chebyshev_polynomials(adj, k):
#     """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
#     print("Calculating Chebyshev polynomials up to order {}...".format(k))

#     adj_normalized = normalize_adj(adj)
#     laplacian = sp.eye(adj.shape[0]) - adj_normalized
#     largest_eigval, _ = eigsh(laplacian, 1, which='LM')
#     scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

#     t_k = list()
#     t_k.append(sp.eye(adj.shape[0]))
#     t_k.append(scaled_laplacian)

#     def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
#         s_lap = sp.csr_matrix(scaled_lap, copy=True)
#         return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

#     for i in range(2, k+1):
#         t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

#     return sparse_to_tuple(t_k)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_adj_raw(dataset_str):

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../dataset/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../dataset/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    adj_raw = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj_raw

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def load_citation(dataset_str="cora", normalization="AugNormAdj", porting_to_torch=True,data_path=datadir, task_type="semi"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str.lower(), names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # degree = np.asarray(G.degree)
    degree = np.sum(adj, axis=1)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    if task_type == "full":
        print("Load full supervised task.")
        #supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(ally)- 500)
        idx_val = range(len(ally) - 500, len(ally))
    elif task_type == "semi":
        print("Load semi-supervised task.")
        #semi-supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
    else:
        raise ValueError("Task type: %s is not supported. Available option: full and semi.")

    adj, features = preprocess_citation(adj, features, normalization)
    features = np.array(features.todense())
    labels = np.argmax(labels, axis=1)
    # porting to pytorch
    if porting_to_torch:
        features = torch.FloatTensor(features).float()
        labels = torch.LongTensor(labels)
        # labels = torch.max(labels, dim=1)[1]
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        degree = torch.LongTensor(degree)
    learning_type = "transductive"
    
    print(f"train: {len(idx_train)} val: {len(idx_val)} test: {len(idx_test)}")
    return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type

def sgc_precompute(features, adj, degree):
    #t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = 0 #perf_counter()-t
    return features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

    
def print_sparsity(edge_masks, edge_num):
    with torch.no_grad():
        spar_ls = []
        for ln in range(len(edge_masks)):
            spar = (1 - (edge_masks[ln].sum().item() / edge_num)) * 100
            spar_ls.append(spar)
            print(f"layer {ln}: [{spar:.4f}%]")    
        print("="*20)
        print(f"avg sparsity: [{np.mean(spar_ls):.4f}%]")
        print("="*20)

def judge_spar(spar, target):
    return spar >= (target - 2) and spar <= (target + 2)


def calcu_sparsity(edge_masks, edge_num):
    if edge_masks is None:
        return 0
    with torch.no_grad():
        spar_ls = []
        for ln in range(len(edge_masks)):
            spar = (1 - (edge_masks[ln].sum().item() / edge_num)) * 100
            spar_ls.append(spar)
        return np.mean(spar_ls)


def create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + \
            math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + \
            math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)


def denominator(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    # ks_sum refers to O_k in the paper
    ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones)
    return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)


def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m/d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)


def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / \
        torch.sqrt(torch.sqrt(torch.tensor(
            data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / \
        torch.sqrt(torch.tensor(
            projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape)-1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash,
                      dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                                                        dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    return data_dash


def degree(index: torch.Tensor, num_nodes: Optional[int] = None,
           dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`

    Example:

        >>> row = torch.tensor([0, 1, 0, 2, 0])
        >>> degree(row, dtype=torch.long)
        tensor([3, 1, 1])
    """
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(
                    path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)


def row_normalize_adjacency_matrix(adj_matrix):
    device = adj_matrix.device
    # Calculate the degree matrix D by summing along the rows of the adjacency matrix
    degree_matrix = torch.diag(1. / torch.sum(adj_matrix, dim=1))
    
    # Calculate the inverse of the degree matrix
    degree_inv_matrix = degree_matrix.masked_fill_(degree_matrix == float('inf'), 0)
    degree_inv_matrix.masked_fill_(degree_inv_matrix.isnan(), 0)

    # with torch.no_grad():
    #     print(f"[{(degree_inv_matrix.isnan()).sum().item()}]")
    #     print(f"[{(degree_inv_matrix == float('inf')).sum().item()}]")
    
    # Compute the normalized adjacency matrix A_norm = -D^{-1} A
    normalized_adj_matrix = torch.mm(degree_inv_matrix, adj_matrix)
    

    # zero_row_indices = torch.where(normalized_adj_matrix.sum(dim=1) == 0)[0]
    # normalized_adj_matrix[zero_row_indices, zero_row_indices] = 1
    # return torch.eye(adj_matrix.shape[0]).to(device) - normalized_adj_matrix
    return  normalized_adj_matrix


@torch.no_grad()
def net_weight_sparsity(model: nn.Module):
    total, keep = 0., 0.
    for layer in model.modules():
        if isinstance(layer, layers.MaskedLinear):
            abs_weight = torch.abs(layer.weight)
            threshold = layer.threshold.view(abs_weight.shape[0], -1)
            abs_weight = abs_weight-threshold
            mask = layer.step(abs_weight)
            ratio = torch.sum(mask) / mask.numel()
            total += mask.numel()
            keep += torch.sum(mask)
            # logger.info("Layer threshold {:.4f}".format(layer.threshold[0]))
            # logger.info("{}, keep ratio {:.4f}".format(layer, ratio))
    if not total:
        return 0
    else:
        return float(1 - keep / total) * 100
    
def initalize_thres(coef):
    def equation(x):
        return x**3 + 20*x + 0.2/coef
    
    result = optimize.root_scalar(equation, bracket=[-10, 10], method='bisect')
    return result.root


import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
from normalization import fetch_normalization, row_normalize
import pickle as pkl
import networkx as nx
import json
import sys
import random
import os
from networkx.readwrite import json_graph
import math

import pdb

sys.setrecursionlimit(99999)


def _preprocess_adj(normalization, adj, cuda):
    adj_normalizer = fetch_normalization(normalization)
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    if cuda:
        r_adj = r_adj.cuda()
    return r_adj


def _preprocess_fea(fea, cuda):
    if cuda:
        return fea.cuda()
    else:
        return fea


def stub_sampler(train_adj, train_features, normalization, cuda):
    """
    The stub sampler. Return the original data.
    """
    trainadj_cache = {}
    if normalization in trainadj_cache:
        r_adj = trainadj_cache[normalization]
    else:
        r_adj = _preprocess_adj(normalization, train_adj, cuda)
        trainadj_cache[normalization] = r_adj
    fea = _preprocess_fea(train_features, cuda)
    return r_adj, fea


def randomedge_sampler(train_adj, train_features, percent, normalization, cuda):
    """
    Randomly drop edge and preserve percent% edges.
    """
    "Opt here"
    if percent >= 1.0:
        return stub_sampler(train_adj, train_features, normalization, cuda)

    nnz = train_adj.nnz
    perm = np.random.permutation(nnz)
    preserve_nnz = int(nnz * percent)
    perm = perm[:preserve_nnz]
    r_adj = sp.coo_matrix((train_adj.data[perm],
                           (train_adj.row[perm],
                            train_adj.col[perm])),
                          shape=train_adj.shape)
    r_adj = _preprocess_adj(normalization, r_adj, True)
    fea = _preprocess_fea(train_features, True)
    # import ipdb; ipdb.set_trace()
    return r_adj, fea


def get_test_set(adj, features, normalization, cuda):
    """
    Return the test set.
    """

    r_adj = _preprocess_adj(normalization, adj, cuda)
    fea = _preprocess_fea(features, cuda)
    return r_adj, fea


def get_val_set(adj, features, normalization, cuda):
    """
    Return the validataion set. Only for the inductive task.
    Currently behave the same with get_test_set
    """
    return get_test_set(adj, features, normalization, cuda)


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)

    features = row_normalize(features)
    return adj, features


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum == 0) * 1 + rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# adapted from PetarV/GAT
def run_dfs(adj, msk, u, ind, nb_nodes):
    if msk[u] == -1:
        msk[u] = ind
        # for v in range(nb_nodes):
        for v in adj[u, :].nonzero()[1]:
            # if adj[u,v]== 1:
            run_dfs(adj, msk, v, ind, nb_nodes)


def dfs_split(adj):
    # Assume adj is of shape [nb_nodes, nb_nodes]
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)

    graph_id = 0

    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1

    return ret


def test(adj, mapping):
    nb_nodes = adj.shape[0]
    for i in range(nb_nodes):
        # for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] != mapping[j]:
                #  if adj[i,j] == 1:
                return False
    return True


def find_split(adj, mapping, ds_label):
    nb_nodes = adj.shape[0]
    dict_splits = {}
    for i in range(nb_nodes):
        # for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] == 0 or mapping[j] == 0:
                dict_splits[0] = None
            elif mapping[i] == mapping[j]:
                if ds_label[i]['val'] == ds_label[j]['val'] and ds_label[i]['test'] == ds_label[j]['test']:

                    if mapping[i] not in dict_splits.keys():
                        if ds_label[i]['val']:
                            dict_splits[mapping[i]] = 'val'

                        elif ds_label[i]['test']:
                            dict_splits[mapping[i]] = 'test'

                        else:
                            dict_splits[mapping[i]] = 'train'

                    else:
                        if ds_label[i]['test']:
                            ind_label = 'test'
                        elif ds_label[i]['val']:
                            ind_label = 'val'
                        else:
                            ind_label = 'train'
                        if dict_splits[mapping[i]] != ind_label:
                            print('inconsistent labels within a graph exiting!!!')
                            return None
                else:
                    print('label of both nodes different, exiting!!')
                    return None
    return dict_splits


def load_ppi():
    print('Loading G...')
    with open('ppi/ppi-G.json') as jsonfile:
        g_data = json.load(jsonfile)
    # print (len(g_data))
    G = json_graph.node_link_graph(g_data)

    # Extracting adjacency matrix
    adj = nx.adjacency_matrix(G)

    prev_key = ''
    for key, value in g_data.items():
        if prev_key != key:
            # print (key)
            prev_key = key

    # print ('Loading id_map...')
    with open('ppi/ppi-id_map.json') as jsonfile:
        id_map = json.load(jsonfile)
    # print (len(id_map))

    id_map = {int(k): int(v) for k, v in id_map.items()}
    for key, value in id_map.items():
        id_map[key] = [value]
    # print (len(id_map))

    print('Loading features...')
    features_ = np.load('ppi/ppi-feats.npy')
    # print (features_.shape)

    # standarizing features
    from sklearn.preprocessing import StandardScaler

    train_ids = np.array([id_map[n] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])
    train_feats = features_[train_ids[:, 0]]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    features_ = scaler.transform(features_)

    features = sp.csr_matrix(features_).tolil()

    print('Loading class_map...')
    class_map = {}
    with open('ppi/ppi-class_map.json') as jsonfile:
        class_map = json.load(jsonfile)
    # print (len(class_map))

    # pdb.set_trace()
    # Split graph into sub-graphs
    # print ('Splitting graph...')
    splits = dfs_split(adj)

    # Rearrange sub-graph index and append sub-graphs with 1 or 2 nodes to bigger sub-graphs
    # print ('Re-arranging sub-graph IDs...')
    list_splits = splits.tolist()
    group_inc = 1

    for i in range(np.max(list_splits) + 1):
        if list_splits.count(i) >= 3:
            splits[np.array(list_splits) == i] = group_inc
            group_inc += 1
        else:
            # splits[np.array(list_splits) == i] = 0
            ind_nodes = np.argwhere(np.array(list_splits) == i)
            ind_nodes = ind_nodes[:, 0].tolist()
            split = None

            for ind_node in ind_nodes:
                if g_data['nodes'][ind_node]['val']:
                    if split is None or split == 'val':
                        splits[np.array(list_splits) == i] = 21
                        split = 'val'
                    else:
                        raise ValueError('new node is VAL but previously was {}'.format(split))
                elif g_data['nodes'][ind_node]['test']:
                    if split is None or split == 'test':
                        splits[np.array(list_splits) == i] = 23
                        split = 'test'
                    else:
                        raise ValueError('new node is TEST but previously was {}'.format(split))
                else:
                    if split is None or split == 'train':
                        splits[np.array(list_splits) == i] = 1
                        split = 'train'
                    else:
                        pdb.set_trace()
                        raise ValueError('new node is TRAIN but previously was {}'.format(split))

    # counting number of nodes per sub-graph
    list_splits = splits.tolist()
    nodes_per_graph = []
    for i in range(1, np.max(list_splits) + 1):
        nodes_per_graph.append(list_splits.count(i))

    # Splitting adj matrix into sub-graphs
    subgraph_nodes = np.max(nodes_per_graph)
    adj_sub = np.empty((len(nodes_per_graph), subgraph_nodes, subgraph_nodes))
    feat_sub = np.empty((len(nodes_per_graph), subgraph_nodes, features.shape[1]))
    labels_sub = np.empty((len(nodes_per_graph), subgraph_nodes, 121))

    for i in range(1, np.max(list_splits) + 1):
        # Creating same size sub-graphs
        indexes = np.where(splits == i)[0]
        subgraph_ = adj[indexes, :][:, indexes]

        if subgraph_.shape[0] < subgraph_nodes or subgraph_.shape[1] < subgraph_nodes:
            subgraph = np.identity(subgraph_nodes)
            feats = np.zeros([subgraph_nodes, features.shape[1]])
            labels = np.zeros([subgraph_nodes, 121])
            # adj
            subgraph = sp.csr_matrix(subgraph).tolil()
            subgraph[0:subgraph_.shape[0], 0:subgraph_.shape[1]] = subgraph_
            adj_sub[i - 1, :, :] = subgraph.todense()
            # feats
            feats[0:len(indexes)] = features[indexes, :].todense()
            feat_sub[i - 1, :, :] = feats
            # labels
            for j, node in enumerate(indexes):
                labels[j, :] = np.array(class_map[str(node)])
            labels[indexes.shape[0]:subgraph_nodes, :] = np.zeros([121])
            labels_sub[i - 1, :, :] = labels

        else:
            adj_sub[i - 1, :, :] = subgraph_.todense()
            feat_sub[i - 1, :, :] = features[indexes, :].todense()
            for j, node in enumerate(indexes):
                labels[j, :] = np.array(class_map[str(node)])
            labels_sub[i - 1, :, :] = labels

    # Get relation between id sub-graph and tran,val or test set
    dict_splits = find_split(adj, splits, g_data['nodes'])

    # Testing if sub graphs are isolated
    # print ('Are sub-graphs isolated?')
    # print (test(adj, splits))

    # Splitting tensors into train,val and test
    train_split = []
    val_split = []
    test_split = []
    for key, value in dict_splits.items():
        if dict_splits[key] == 'train':
            train_split.append(int(key) - 1)
        elif dict_splits[key] == 'val':
            val_split.append(int(key) - 1)
        elif dict_splits[key] == 'test':
            test_split.append(int(key) - 1)

    train_adj = adj_sub[train_split, :, :]
    val_adj = adj_sub[val_split, :, :]
    test_adj = adj_sub[test_split, :, :]

    train_feat = feat_sub[train_split, :, :]
    val_feat = feat_sub[val_split, :, :]
    test_feat = feat_sub[test_split, :, :]

    train_labels = labels_sub[train_split, :, :]
    val_labels = labels_sub[val_split, :, :]
    test_labels = labels_sub[test_split, :, :]

    train_nodes = np.array(nodes_per_graph[train_split[0]:train_split[-1] + 1])
    val_nodes = np.array(nodes_per_graph[val_split[0]:val_split[-1] + 1])
    test_nodes = np.array(nodes_per_graph[test_split[0]:test_split[-1] + 1])

    # Masks with ones

    tr_msk = np.zeros((len(nodes_per_graph[train_split[0]:train_split[-1] + 1]), subgraph_nodes))
    vl_msk = np.zeros((len(nodes_per_graph[val_split[0]:val_split[-1] + 1]), subgraph_nodes))
    ts_msk = np.zeros((len(nodes_per_graph[test_split[0]:test_split[-1] + 1]), subgraph_nodes))

    for i in range(len(train_nodes)):
        for j in range(train_nodes[i]):
            tr_msk[i][j] = 1

    for i in range(len(val_nodes)):
        for j in range(val_nodes[i]):
            vl_msk[i][j] = 1

    for i in range(len(test_nodes)):
        for j in range(test_nodes[i]):
            ts_msk[i][j] = 1

    train_adj_list = []
    val_adj_list = []
    test_adj_list = []
    for i in range(train_adj.shape[0]):
        adj = sp.coo_matrix(train_adj[i])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        tmp = sys_normalized_adjacency(adj)
        train_adj_list.append(sparse_mx_to_torch_sparse_tensor(tmp))
    for i in range(val_adj.shape[0]):
        adj = sp.coo_matrix(val_adj[i])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        tmp = sys_normalized_adjacency(adj)
        val_adj_list.append(sparse_mx_to_torch_sparse_tensor(tmp))
        adj = sp.coo_matrix(test_adj[i])
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        tmp = sys_normalized_adjacency(adj)
        test_adj_list.append(sparse_mx_to_torch_sparse_tensor(tmp))

    train_feat = torch.FloatTensor(train_feat)
    val_feat = torch.FloatTensor(val_feat)
    test_feat = torch.FloatTensor(test_feat)

    train_labels = torch.FloatTensor(train_labels)
    val_labels = torch.FloatTensor(val_labels)
    test_labels = torch.FloatTensor(test_labels)

    tr_msk = torch.LongTensor(tr_msk)
    vl_msk = torch.LongTensor(vl_msk)
    ts_msk = torch.LongTensor(ts_msk)

    return train_adj_list, val_adj_list, test_adj_list, train_feat, val_feat, test_feat, train_labels, val_labels, test_labels, train_nodes, val_nodes, test_nodes


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


# def one_hot_embedding(labels, num_classes, soft):
#     """Embedding labels to one-hot form.
#     Args:
#       labels: (LongTensor) class labels, sized [N,].
#       num_classes: (int) number of classes.
#     Returns:
#       (tensor) encoded labels, sized [N, #classes].
#     """
#     soft = torch.argmax(soft.exp(), dim=1)
#     y = torch.eye(num_classes)
#     return y[soft]

def add_label_noise(idx_train, labels, noise_ratio, seed):
    if noise_ratio == None:
        return labels
    random.seed(seed)
    num_nodes = idx_train[-1]
    erasing_pool = torch.arange(num_nodes)
    print('pool', erasing_pool.shape)
    np.random.seed(seed)
    noise_num = int(num_nodes * noise_ratio)
    print('noise_num', noise_num)

    sele_idx = [j for j in random.sample(range(0, num_nodes), noise_num)]
    for i in sele_idx:
        re_lb = random.sample([j for j in range(max(labels) + 1) if j != labels[idx_train[i]]], 1)
        labels[idx_train[i]] = re_lb[0]
    # import ipdb; ipdb.set_trace()
    return labels


def calculate_smoothness(features, adj):
    """
    计算图神经网络的平滑度(smoothingness)

    平滑度定义为特征与邻居特征之间的相似度
    值越高表示图越平滑（相邻节点特征越相似）

    参数:
        features: 节点特征矩阵 [num_nodes, feature_dim]
        adj: 邻接矩阵 (稀疏格式)

    返回:
        smoothness: 平滑度值
    """
    # 确保输入是PyTorch张量
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)

    # 将稀疏邻接矩阵转换为密集矩阵
    if hasattr(adj, 'to_dense'):
        dense_adj = adj.to_dense()
    else:
        dense_adj = adj

    # 计算节点特征的L2范数
    norm_features = F.normalize(features, p=2, dim=1)

    # 计算节点与其邻居的特征相似度
    # 使用归一化特征的点积作为相似度度量
    similarity = torch.mm(norm_features, norm_features.t())

    # 只考虑邻接矩阵中存在的边
    edge_similarity = similarity * dense_adj

    # 计算平均相似度作为平滑度指标
    # 只对实际存在的边计算平均值
    num_edges = dense_adj.sum()
    if num_edges > 0:
        smoothness = edge_similarity.sum() / num_edges
    else:
        smoothness = torch.tensor(0.0)

    return smoothness.item()


def calculate_l2_smoothness(features, adj):
    """
    使用差值二范数计算图的平滑度

    平滑度定义为相邻节点特征之间的平均L2距离
    值越低表示图越平滑（相邻节点特征越相似）
    为了与原始smoothness保持一致（值越大表示越平滑），返回1减去归一化后的平均距离

    参数:
        features: 节点特征矩阵 [num_nodes, feature_dim]
        adj: 邻接矩阵 (稀疏格式)

    返回:
        l2_smoothness: 基于L2范数的平滑度值，越大表示越平滑
    """
    print("L2平滑度计算开始")

    # 确保输入是PyTorch张量
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)

    # 将稀疏邻接矩阵转换为密集矩阵
    if hasattr(adj, 'to_dense'):
        print("将稀疏邻接矩阵转为密集矩阵")
        dense_adj = adj.to_dense()
    else:
        dense_adj = adj

    print(f"邻接矩阵形状: {dense_adj.shape}")
    print(f"特征矩阵形状: {features.shape}")

    # 创建对角线掩码(对角线元素为0，其他为1)，用于过滤自环
    diag_mask = ~torch.eye(dense_adj.shape[0], dtype=torch.bool, device=dense_adj.device)
    # 应用对角线掩码过滤掉自环
    adj_no_self_loop = dense_adj * diag_mask

    # 获取存在边的索引
    edge_indices = adj_no_self_loop.nonzero(as_tuple=True)
    num_edges = edge_indices[0].size(0)
    print(f"边的数量: {num_edges}")

    if num_edges > 0:
        # 如果边的数量太多，进行采样
        MAX_EDGES = 100000  # 最大计算的边数
        if num_edges > MAX_EDGES:
            print(f"边数过多，从 {num_edges} 边中采样 {MAX_EDGES} 边")
            perm = torch.randperm(num_edges)[:MAX_EDGES]
            source_nodes = edge_indices[0][perm]
            target_nodes = edge_indices[1][perm]
        else:
            source_nodes = edge_indices[0]
            target_nodes = edge_indices[1]

        # 使用批处理避免内存溢出
        batch_size = 10000
        total_distances = 0.0
        total_count = 0

        try:
            print("开始批处理计算欧氏距离")
            for i in range(0, len(source_nodes), batch_size):
                end_idx = min(i + batch_size, len(source_nodes))
                batch_source = source_nodes[i:end_idx]
                batch_target = target_nodes[i:end_idx]

                # 获取源节点和目标节点的特征
                batch_source_features = features[batch_source]
                batch_target_features = features[batch_target]

                # 计算每对节点之间的欧氏距离
                batch_distances = torch.norm(batch_source_features - batch_target_features, p=2, dim=1)

                # 累加距离
                total_distances += batch_distances.sum().item()
                total_count += batch_distances.size(0)

                if (i // batch_size) % 10 == 0:
                    print(f"已处理 {i} / {len(source_nodes)} 边")

            # 计算平均距离
            avg_distance = total_distances / total_count
            print(f"平均距离: {avg_distance}")

            # 归一化距离，使其在0到1之间
            # 假设特征已经归一化或在相近范围内，最大距离约为sqrt(2)
            norm_factor = math.sqrt(2.0)
            normalized_distance = min(avg_distance / norm_factor, 1.0)
            print(f"归一化距离: {normalized_distance}")

            # 返回1减去归一化距离，使得值越大表示越平滑（与原始平滑度定义一致）
            l2_smoothness = 1.0 - normalized_distance
            print(f"L2平滑度: {l2_smoothness}")
        except Exception as e:
            print(f"计算距离时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 如果计算出错，返回默认值
            l2_smoothness = 0.5
    else:
        # 如果没有边，返回最大平滑度
        print("没有边，返回默认平滑度1.0")
        l2_smoothness = 1.0

    print("L2平滑度计算完成")
    return l2_smoothness