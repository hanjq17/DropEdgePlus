# coding=utf-8
import numpy as np
import torch
import scipy.sparse as sp
from utils import data_loader, sparse_mx_to_torch_sparse_tensor
from normalization import fetch_normalization
from torch.nn.parameter import Parameter
import math
import torch.nn


class Sampler:
    """Sampling the input graph data."""
    def __init__(self, dataset, data_path="data", task_type="full"):
        self.dataset = dataset
        self.data_path = data_path
        (self.ori_adj,
         self.adj,
         self.train_adj,
         self.features,
         self.train_features,
         self.labels,
         self.idx_train, 
         self.idx_val,
         self.idx_test, 
         self.degree,
         self.learning_type) = data_loader(dataset, data_path, "NoNorm", False, task_type)
        
        # convert some data to torch tensor ---- may be not the best practice here.
        self.ori_features = self.train_features
        self.features = torch.FloatTensor(self.features).float()
        self.train_features = torch.FloatTensor(self.train_features).float()

        # self.train_adj = self.train_adj.tocsr()
        self.mask = None

        self.labels_torch = torch.LongTensor(self.labels)
        self.idx_train_torch = torch.LongTensor(self.idx_train)
        self.idx_val_torch = torch.LongTensor(self.idx_val)
        self.idx_test_torch = torch.LongTensor(self.idx_test)

        self.pos_train_idx = np.where(self.labels[self.idx_train] == 1)[0]
        self.neg_train_idx = np.where(self.labels[self.idx_train] == 0)[0]

        self.nfeat = self.features.shape[1]
        self.nclass = int(self.labels.max().item() + 1)
        self.trainadj_cache = {}
        self.adj_cache = {}
        self.degree_p = None

        self.loss = None

    def _preprocess_adj(self, normalization, adj, cuda):
        # print(len(adj.data))
        adj_normalizer = fetch_normalization(normalization)
        r_adj = adj_normalizer(adj)
        r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
        if cuda:
            r_adj = r_adj.cuda()
        return r_adj

    def _preprocess_fea(self, fea, cuda):
        if cuda:
            return fea.cuda()
        else:
            return fea

    def get_sample_func(self):
        return self.sampler

    def stub_sampler(self, normalization, cuda, **kwargs):
        """
        The stub sampler. Return the original data. 
        """
        r_adj = self._preprocess_adj(normalization, self.train_adj, cuda)
        fea = self._preprocess_fea(self.train_features, cuda)
        return r_adj, fea

    def sampler(self, normalization, cuda, **kwargs):
        layer_num = kwargs['layer_num']
        if layer_num <= 1:
            return self.stub_sampler(normalization, cuda)
        b = kwargs['b']
        a = kwargs['a']
        # y = kx + b, k = ((1 - b) * a + b - b ) / (layer_num - 1)
        nnz = self.train_adj.nnz
        k = (1 - b) * a / (layer_num - 1)
        keep_num = [int(nnz * (b + _ * k)) for _ in range(layer_num)]
        mask = self.mask
        prob = mask.data
        prob /= sum(prob)
        edge_idx = np.array([_ for _ in range(mask.nnz)])
        adj = self.train_adj.tocoo()
        r_adj_list = []
        if k >= 0:
            keep_num.reverse()
        for _ in range(layer_num):
            edge_idx_ = np.random.choice([_ for _ in range(len(edge_idx))], size=keep_num[_], replace=False, p=prob)
            edge_idx = edge_idx[edge_idx_]
            prob = prob[edge_idx_]
            prob /= sum(prob)
            r_adj = sp.coo_matrix((adj.data[edge_idx],
                                   (adj.row[edge_idx], adj.col[edge_idx])), shape=adj.shape)
            r_adj = self._preprocess_adj(normalization, r_adj, cuda)
            r_adj_list.append(r_adj)
        if k >= 0:
            r_adj_list.reverse()

        fea = self._preprocess_fea(self.train_features, cuda)
        return r_adj_list, fea

    def get_test_set(self, normalization, cuda):
        """
        Return the test set. 
        """
        if self.learning_type == "transductive":
            return self.stub_sampler(normalization, cuda)
        else:
            if normalization in self.adj_cache:
                r_adj = self.adj_cache[normalization]
            else:
                r_adj = self._preprocess_adj(normalization, self.adj, cuda)
                self.adj_cache[normalization] = r_adj
            fea = self._preprocess_fea(self.features, cuda)
            return r_adj, fea

    def get_val_set(self, normalization, cuda):
        """
        Return the validataion set. Only for the inductive task.
        Currently behave the same with get_test_set
        """
        return self.get_test_set(normalization, cuda)

    def get_label_and_idxes(self, cuda):
        """
        Return all labels and indexes.
        """
        if cuda:
            return self.labels_torch.cuda(), self.idx_train_torch.cuda(), \
                   self.idx_val_torch.cuda(), self.idx_test_torch.cuda()
        return self.labels_torch, self.idx_train_torch, self.idx_val_torch, self.idx_test_torch
