import pickle as pkl
import sys
import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from normalization import fetch_normalization, row_normalize
from make_dataset import get_dataset, get_train_val_test_split

datadir = "data"


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_citation(dataset_str="cora", normalization="AugNormAdj",
                  porting_to_torch=True, data_path=datadir, task_type="full"):
    """
    Load Citation Networks Datasets.
    """
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
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

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = np.argmax(labels, axis=1)

        G = nx.from_dict_of_lists(graph)
        adj = nx.adjacency_matrix(G)

        if task_type == "full":
            print("Load full supervised task.")
            # supervised setting
            idx_test = test_idx_range.tolist()
            idx_train = range(len(ally) - 500)
            idx_val = range(len(ally) - 500, len(ally))
        elif task_type == "semi":
            print("Load semi-supervised task.")
            # semi-supervised setting
            idx_test = test_idx_range.tolist()
            idx_train = range(len(y))
            idx_val = range(len(y), len(y) + 500)
        else:
            raise ValueError("Task type: %s is not supported. Available option: full and semi.")

    else:
        dataset_str = dataset_str_mapping(dataset_str)
        seed = 233
        np.random.seed(seed)
        random_state = np.random.RandomState(seed)
        # Load data
        slug = get_dataset(dataset_str, data_path + '/npz/' + dataset_str + '.npz', True, None)
        labels = slug[2].astype('int32')
        labels = np.argmax(labels, axis=1)
        adj = slug[0]
        features = slug[1]

        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, slug[-1], train_examples_per_class=20,
                                                                val_examples_per_class=30)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    degree = np.sum(adj, axis=1)

    adj_preprocessed, features = preprocess_citation(adj, features, normalization)
    features = np.array(features.todense())

    learning_type = "transductive"
    return adj, adj_preprocessed, features, labels, idx_train, idx_val, idx_test, degree, learning_type


def data_loader(dataset, data_path=datadir, normalization="AugNormAdj", porting_to_torch=True, task_type="full"):
    (ori_adj,
     adj,
     features,
     labels,
     idx_train,
     idx_val,
     idx_test,
     degree,
     learning_type) = load_citation(dataset, normalization, porting_to_torch, data_path, task_type)
    train_adj = adj
    train_features = features
    return ori_adj, adj, train_adj, features, train_features, labels, \
           idx_train, idx_val, idx_test, degree, learning_type


def dataset_str_mapping(dataset):
    map_dict = {'cora': 'cora',
                'citeseer': 'citeseer',
                'pubmed': 'pubmed',
                'mac': 'ms_academic_cs',
                'map': 'ms_academic_phy',
                'aec': 'amazon_electronics_computers',
                'aep': 'amazon_electronics_photo'}
    return map_dict[dataset]
