import scipy.io
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.io
from einops import rearrange, reduce
from pmath import project, logmap0
import manifolds


class FKernel(torch.nn.Module):
    def __init__(self, c):
        super(FKernel, self).__init__()
        # self.device = device
        self.c = c

    def forward(self, x):
        output = project(x, c=self.c)
        output = logmap0(output, c=self.c)
        return output


import torch
import torch.nn.functional as F
from torch import nn
import scipy
import scipy.sparse as sp
from einops import rearrange

class Module_1(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, c, a_couping, dropout, use_att, local_agg, manifold):
        """
        Args:
        ----
            input_dim: input feature dimension
            hidden_dim: hidden layer output dimension
            c: number of ROIs or graph size context
            a_couping: adjacency coupling strategy or parameter
            dropout: dropout rate
            use_att: whether to use attention in GCN
            local_agg: use of local aggregation
            manifold: manifold-related configuration
        """
        super(Module_1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.c = c
        self.a_para = a_couping

        self.gcn1 = GraphConvolution(input_dim, hidden_dim, c, dropout, use_att, local_agg, manifold, use_bias=True)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim, c, dropout, use_att, local_agg, manifold, use_bias=True)

    def forward(self, fMRI):
        # Preprocess fMRI to obtain adjacency matrix and features
        a_fMRI, f_fMRI = preprocess(fMRI)  # a: (B, N, N), f: (B, N, F)
        a_fMRI = a_fMRI.cuda()
        f_fMRI = f_fMRI.cuda()


        epsilon = 1e-8
        f_fMRI_norm = f_fMRI / (f_fMRI.norm(p=2, dim=-1, keepdim=True) + epsilon)
        a_fMRI_norm = a_fMRI / (a_fMRI.norm(p=2, dim=-1, keepdim=True) + epsilon)


        a = a_fMRI_norm.detach().cpu().numpy()
        adj = scipy.linalg.block_diag(*abs(a))
        adj_csr = sp.csr_matrix(adj)
        adj_nor = normalization(adj_csr).cuda().to(torch.float32)

        fea = rearrange(f_fMRI_norm, 'b n f -> (b n) f').cuda().to(torch.float32)


        x1 = self.gcn1(adj_nor, fea)
        gcn1 = F.relu(x1)
        x2 = self.gcn2(adj_nor, gcn1)
        gcn2 = F.relu(x2)

        x = rearrange(gcn2, '(b n) c -> b n c', b=f_fMRI.shape[0], n=f_fMRI.shape[1])
        return x


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, c, dropout, use_att, local_agg, manifold, use_bias=True):
        """
        Args:
        ----------
            input_dim: the dimension of the input feature

            output_dim: the dimension of the output feature

            use_bias : bool, optional

        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.dropout = dropout
        self.c = c
        self.use_att = use_att
        self.local_agg = local_agg
        self.manifold = getattr(manifolds, manifold)()
        self.relu = nn.ReLU()

        self.fkernel = FKernel(self.c)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        drop_weight = F.dropout(self.weight, self.dropout, training=True)
        # print('drop_weight.shape, input_feature.shape:',drop_weight.shape, input_feature.shape)
        drop_weight = drop_weight.transpose(-1, -2)  # YMM新加的

        support = self.manifold.mobius_matvec(drop_weight, input_feature,
                                              self.c)  # 先用logmap映射到切空间，再在切空间上计算内积，最后用expmap转换回双曲空间

        # print('support.shape:',support.shape)
        if self.use_bias:
            #    output += self.bias
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            support = self.manifold.mobius_add(support, hyp_bias, c=self.c)
            support = self.manifold.proj(support, self.c)

        support_tangent = self.manifold.logmap0(support, c=self.c)  # 先映射到到欧式空间

        if self.use_att:
            # 这个local是在x附近的切空间上进行运算的
            if self.local_agg:  # 使用局部聚合
                x_local_tangent = []
                for i in range(support.size(0)):
                    x_local_tangent.append(self.manifold.logmap(support[i], support, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(support_tangent, adjacency)  # 计算注意力权重
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)  # 根据注意力进行聚合并投影到双曲空间上
                output = self.manifold.proj(self.manifold.expmap(support, support_t, c=self.c), c=self.c)
                return output
            else:  # 不使用局部聚合
                adj_att = self.att(support_tangent, adjacency)
                support_t = torch.matmul(adj_att, support_tangent)
        else:  # 不使用注意力机制，直接对邻接矩阵adj进行矩阵乘法
            support_t = torch.spmm(adjacency, support_tangent)
            # 映射回双曲空间并投影
        output = self.relu(support_t)

        return output  # (N,output_dim=hidden_dim)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'


def normalization(adjacency):
    """calculate L=D^-0.5 * (A+I) * D^-0.5,
    Args:
        adjacency: sp.csr_matrix.
    Returns:
        normalized matrix, type torch.sparse.FloatTensor
    """
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)

    return tensor_adjacency


def preprocess(data):
    Adj = []
    for i in range(len(data)):
        pc = np.corrcoef(data.cpu()[i].T)  # (116,116)
        pc = np.nan_to_num(pc)
        pc = abs(pc)
        Adj.append(pc)
    adj = torch.from_numpy(np.array(Adj))
    fea = adj
    return adj, fea
