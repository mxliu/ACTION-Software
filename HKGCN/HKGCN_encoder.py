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
import scipy.linalg  


def preprocess_dense(data):
    Adj = []
    for i in range(len(data)):
        pc = np.corrcoef(data.cpu()[i].T)  # (116,116)
        pc = np.nan_to_num(pc)
        pc = abs(pc)
        Adj.append(pc)
    adj = torch.from_numpy(np.array(Adj))
    fea = adj
    return adj,fea



def preprocess(data):
    adj_list = []
    fea_list = []

    for i in range(data.shape[0]):
        pc = np.corrcoef(data[i].cpu().T)  # ✅ GPU tensor 转 CPU 再转 NumPy
        pc = np.nan_to_num(pc)
        pc = np.abs(pc)

        # 每张图保留前 50% 最大相关边
        flat = pc.flatten()
        k = int(flat.shape[0] * 0.500)
        threshold = np.partition(flat, -k)[-k]
        sparse_pc = np.where(pc >= threshold, pc, 0)

        adj_list.append(sparse_pc)
        fea_list.append(pc)  

    adj = np.array(adj_list)
    fea = np.array(fea_list)
    return torch.from_numpy(adj).float(), torch.from_numpy(fea).float()


class Cosine(torch.nn.Module): 
    def __init__(self, data_dim = -1, phi = 3.1415926 * 0.3, bias = False):
        super(Cosine, self).__init__()

        if bias is False:
            self.phi = phi 
        else:
            self.phi = 0.0

        if data_dim > 0:
            self.A = np.sqrt( 1.0 / (2.0 * data_dim) )
        else:
            self.A = 1

    def forward(self, x):
        # x = self.A * cosine_activator_.apply(x + self.phi)
        x = self.A * torch.cos(x + self.phi)
        return x



class FKernel(torch.nn.Module):
    def __init__(self, c):
        super(FKernel, self).__init__()
        #self.device = device
        self.c = c
    def forward(self, x):
        output = project(x, c=self.c)
        output = logmap0(output, c=self.c)
        return output
    


class WKernel(torch.nn.Module):
    def __init__(self, n_input, n_output,a_prompt):
        super(WKernel, self).__init__()
        #self.device = device
        self.fc0 = torch.nn.Linear(n_input, n_output, bias=True)
        self.fc1 = torch.nn.Linear(n_input, n_output, bias=True)
        self.bn0 = torch.nn.BatchNorm1d(116) #batchnormlization 应该是对于（137,116,128）中的116
        self.bn1 = torch.nn.BatchNorm1d(116)
        self.cos = Cosine(data_dim=n_output)
        self.n_input = n_input
        self.n_output = n_output
        self.a_prompt = a_prompt
        self.init_params()
    def init_params(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(module.weight, 1.0)
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x):

        linear0=self.fc0(x)
        bn0=self.bn0(linear0)
        x1 = self.cos(bn0)
        x2 = torch.relu(self.bn1(self.fc1(x)))
        return self.a_prompt*x1 + x2
    



class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
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
        support = torch.mm(input_feature, self.weight)  # XW (N,output_dim=hidden_dim)
        output = torch.sparse.mm(adjacency, support)  # L(XW)  (N,output_dim=hidden_dim)
        if self.use_bias:
            output += self.bias
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



class Module_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, c, a, dropout, pretrained_path = None):
        """

        Args:
        ----
            input_dim: input dimension
            hidden_dim: output dimension
            num_classes: category number (default: 2)
        """
        super(Module_1, self).__init__()
        self.pretrained_path = pretrained_path
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.c = c
        self.a = a
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.fkernel = FKernel(self.c)  #这里也加了双曲核
        self.cos = Cosine()

        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        if self.pretrained_path: 
            self.load_pretrained_fMRI()
        #print("counts")

    def load_pretrained_fMRI(self):
        """
        加载预训练的 fMRI 编码器，仅保留 first_encoder.xxx 前缀的键，
        从而跳过 predictor.xxx 等无关层。加载后会打印相关信息，用于检查加载效果。
        """
        import os
        
        if not os.path.isfile(self.pretrained_path):
            print(f"⚠️ 预训练 fMRI 模型未找到: {self.pretrained_path}")
            return
        
        print(f"🔹 准备加载预训练 fMRI 模型: {self.pretrained_path}")
        checkpoint = torch.load(self.pretrained_path, map_location="cpu")

        # 如果是 {"state_dict": ...} 就取 "state_dict"
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # 打印检查点里所有的键名
        print("该 checkpoint 中包含的全部键如下：")
        for k in state_dict.keys():
            print(k)
        print("Test finished")

        # 只保留 first_encoder.xxx，去掉 "first_encoder." 前缀后写入 filtered_state_dict
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("first_encoder."):
                new_k = k.replace("first_encoder.", "")  # 去掉前缀以匹配当前模型
                filtered_state_dict[new_k] = v

        # 打印过滤后剩余的键
        print("⚙ 过滤后剩余的键：", list(filtered_state_dict.keys()))

        # 执行加载
        msg = self.load_state_dict(filtered_state_dict, strict=False)

        # 查看实际加载情况
        print("missing_keys:", msg.missing_keys)
        print("unexpected_keys:", msg.unexpected_keys)

        print("✅ 预训练 fMRI 编码器加载完成！")
        


    def forward(self, data):
        adj_initial,fea = preprocess(data)
        f =  fea

        # a(nbatch,116,116) f (nbatch,nroi,ninputdim)
        adj = adj_initial.cpu().numpy()#(nbatch,nroi,nroi)
        adj = scipy.linalg.block_diag(*abs(adj))  # (nbatch*nroi,nbatch*nroi)
        adj_csr = sp.csr_matrix(adj)
        adj_nor = normalization(adj_csr).cuda()
        adj_nor = adj_nor.to(torch.float32)
        fea = rearrange(f, 'a b c-> (a b) c').cuda()#(nbatch*nroi,nroi)
        fea = fea.to(torch.float32)  
        
        x1 = self.gcn1(adj_nor, fea) 
        x1 = self.fkernel(x1)
        gcn1 = F.relu(x1) + self.a*self.cos(x1)  #(nbatch*nroi,hiddendim)# (N,hidden_dim)
        gcn1 = F.dropout(gcn1, p=self.dropout, training=self.training)  # ✅ Dropout 添加
       
        x2 = self.gcn2(adj_nor, gcn1)

        x2 = self.fkernel(x2)
        gcn2 = F.relu(x2) + self.a*self.cos(x2) #(nbatch*nroi,hiddendim)
        gcn2 = F.dropout(gcn2, p=self.dropout, training=self.training)  # ✅ Dropout 添加
        x = rearrange(gcn2, '(b n) c -> b n c', b=int(len(adj_nor) / adj_initial.shape[1]), n= adj_initial.shape[1])
        return x #  (nbatch*nroi,hiddendim) #将形状为(nbatch*nroi,hiddendim)的张量重新排为(nbatch，nroi,hiddendim)
