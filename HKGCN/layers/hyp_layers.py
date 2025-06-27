"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))   # input_dim=feat_dim, output_dim=dim
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        #curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
        curvatures = [nn.Parameter(torch.Tensor([args.c])) for _ in range(n_curvatures)]#如果把trainable的curvature换成args.c,会对结果有影响吗
    else:
        # fixed curvature 
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias) #双曲线性层
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)   # aggregation
        self.hyp_act = HypAct(manifold, c_in, c_out, act)   # active function

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj #HGCN处理完之后adj没变，只对feature进行了改变
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

 #这里的输入特征是双曲的。这个函数相当于先对weight函数进行dropout，然后在将其与输入x进行双曲内积运算，然后将其维持在双曲空间上，所以最后的输出结果也是双曲的
    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c) #先用logmap映射到切空间，再在切空间上计算内积，最后用expmap转换回双曲空间
        res = self.manifold.proj(mv, self.c) #再加个project函数维持在双曲流形上
        if self.use_bias: 
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


#双曲聚合层
class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att: #是否使用注意力机制，如果使用，初始化注意力层‘DenseAtt’
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c) #先映射到到欧式空间
        if self.use_att:
            #这个local是在x附近的切空间上进行运算的
            if self.local_agg: #使用局部聚合
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c)) 
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj) #计算注意力权重
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1) #根据注意力进行聚合并投影到双曲空间上
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else: #不使用局部聚合
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent) 
        else: #不使用注意力机制，直接对邻接矩阵adj进行矩阵乘法
            support_t = torch.spmm(adj, x_tangent) 
        #映射回双曲空间并投影
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output
#GCN 实现过程在，在将邻接矩阵与特征进行相乘之后还需要与Weight矩阵相乘，即进行线性运算，一般利用nn.Linear函数进行处理
#此外，在GCN的实现过程中，还需要非线性激活函数，一般用Relu，这里我们可以换成别的



    def extra_repr(self): #返回额外信息，包括曲率c
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in)) #激活函数
        xt = self.manifold.proj_tan0(xt, c=self.c_out)  #维持原有的切空间上的流形
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out) #利用expmap0映射回双曲空间，并利用proj函数确保数据在该流形上

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
