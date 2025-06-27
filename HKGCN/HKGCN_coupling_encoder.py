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


class Sine(torch.nn.Module): #pi初始值3.14*1.2
    def __init__(self, data_dim = -1, phi = 3.1415926 * 0.3, bias = False):
        super(Sine, self).__init__()

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


class Cosine(torch.nn.Module): #pi初始值3.14*1.2
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

class Cosine_fMRI(torch.nn.Module): #pi初始值3.14*1.2
    def __init__(self, data_dim = -1, phi = 3.1415926 * 0.3 , bias = False):
        super(Cosine_fMRI, self).__init__()

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
        self.sine = Sine(data_dim=n_output)
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
        #print('self.n_input,self.n_output',self.n_input,self.n_output)
        #print("WKernel x.shape",x.shape)
        linear0=self.fc0(x)
        #print("WKernel linear0.shape",linear0.shape)
        bn0=self.bn0(linear0)
        #print("WKernel bn0.shape",bn0.shape)
        x1 = self.sine(bn0)
        #print("WKernel x1.shape",x1.shape)

        # x1 = self.cosine( self.fc0(x) )
        x2 = torch.relu(self.bn1(self.fc1(x)))
        # x2 = self.sine( self.fc1(x) )
        #return x* x1 + x2
        return self.a_prompt*x1 + x2
    



class HGNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim,  c, a_couping, a_fMRI,a_DTI, weight_fMRI, weight_DTI, dropout, pretrained_path=None):
        """

        Args:
        ----
            input_dim: input dimension
            hidden_dim: output dimension
            num_classes: category number (default: 2)
        """
        super(HGNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.c = c 
        self.a_para = a_couping
        self.weight_fMRI = weight_fMRI
        self.weight_DTI = weight_DTI
        self.dropout = dropout
        
        self.fkernel = FKernel(self.c)  #这里也加了双曲核
        self.cos = Sine()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        #self.gcn1 = GraphConvolution(input_dim*4, hidden_dim)
        self.gcn1 = GraphConvolution(128, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.coupling = coupling(input_dim, hidden_dim, c, a_fMRI, a_DTI,weight_fMRI, weight_DTI, dropout, pretrained_path)
        #self.transformer = TransformerModel(feature_size, num_layers, nhead)

    def forward(self, DTI, adj_DTI, fMRI):
        adj_coupled, DTI_HKGCN,fMRI_HKGCN,adj_DTI,adj_fMRI= self.coupling(DTI, adj_DTI, fMRI) 

        #f = torch.cat((self.weight_DTI*DTI_norm, self.weight_fMRI*f_fMRI_norm), dim=-1) #DTI本身就norm之后相加的
        f = torch.cat((self.weight_DTI*DTI_HKGCN, self.weight_fMRI*fMRI_HKGCN), dim=-1) #DTI本身就norm之后相加的
        #fea = fea.float()
        #f = nn.Linear(fea.size(-1), 116).to(f_fMRI.device)(fea)   #加了一个linear 层
        adj = adj_coupled
        #a = a.cpu().numpy()#(nbatch,nroi,nroi)
        a = adj.detach().cpu().numpy()
        adj = scipy.linalg.block_diag(*abs(a))  # (nbatch*nroi,nbatch*nroi) 构建对角矩阵        
        adj_csr = sp.csr_matrix(adj) #将adj转换为稀疏矩阵存储
        adj_nor = normalization(adj_csr).cuda() #归一化邻接矩阵，这一步有必要吗？
        adj_nor = adj_nor.to(torch.float32)
        fea = rearrange(f, 'a b c-> (a b) c').cuda()#(nbatch*nroi,nroi) 维度重排，将f从（a，b,c）维度展成（a*b，c）的维度
        fea = fea.to(torch.float32)
        x1 =  self.gcn1(adj_nor, fea)
        #x1 = self.bn1(x1)  # ✅ BatchNorm 加在 GCN 后、激活前
        x1 = self.fkernel(x1)
        gcn1 = F.relu(x1) + self.a_para*self.cos(x1)  #(nbatch*nroi,hiddendim)# (N,hidden_dim)
        gcn1 = F.dropout(gcn1, p=self.dropout, training=self.training)  # ✅ Dropout 添加
        
        x2 = self.gcn2(adj_nor, gcn1)
        x2 = self.fkernel(x2)
        #x2 = self.bn2(x2)  # ✅ 第二层也同理
        gcn2 = F.relu(x2) + self.a_para*self.cos(x2) #(nbatch*nroi,hiddendim)
        gcn2 = F.dropout(gcn2, p=self.dropout, training=self.training)  # ✅ Dropout 添加
       
        x = rearrange(gcn2, '(b n) c -> b n c', b=int(len(adj_nor) / a.shape[1]), n= a.shape[1])
        fea_coupled = x
        return fea_coupled, adj_coupled, DTI_HKGCN,fMRI_HKGCN, adj_DTI, adj_fMRI#  (nbatch*nroi,hiddendim) #将形状为(nbatch*nroi,hiddendim)的张量重新排为(nbatch，nroi,hiddendim)


class coupling(torch.nn.Module):
    def __init__(self, in_channels, feature_size, c, a_fMRI, a_DTI, weight_fMRI, weight_DTI, dropout, pretrained_path = None):
        super(coupling, self).__init__()
        self.module1 = Module_DTI(in_channels*3, feature_size, c, a_DTI, dropout)
        self.module2 = Module_fMRI(in_channels, feature_size, c, a_fMRI, dropout, pretrained_path)
        #self.a_fMRI = a_fMRI
        #self.a_DTI = a_DTI
        self.weight_fMRI = weight_fMRI
        self.weight_DTI = weight_DTI

    
    def forward(self, DTI,adj_DTI, fMRI):
        #utilize the two graph embedding by 2 GCNs
        data_DTI = self.module1(DTI,adj_DTI)
        data_fMRI, adj_fMRI = self.module2(fMRI) #或者看一下module_1原来的
        
        epsilon = 1e-7 
        data_fMRI_norm = data_fMRI / (data_fMRI.norm(p=2, dim=-1, keepdim=True) + epsilon)
        data_DTI_norm = data_DTI / (data_DTI.norm(p=2, dim=-1, keepdim=True) + epsilon)
        #print('shape of DTI',data_fMRI.shape,'shape of fMRI',data_DTI.shape)
        data_DTI_transposed = data_DTI_norm.transpose(1, 2)  # Shape: (64, 64, 116)
        #print('data_fMRI.shape',data_fMRI.shape,'data_DTI.shape',data_DTI.shape)
        coupled_data = torch.matmul(data_fMRI_norm, data_DTI_transposed) # Shape will be (64, 116, 128)
        #print('coupled_data.shape',coupled_data.shape)

        #稀疏化
        adj = coupled_data.clone()

        # 1. 将整个矩阵展平成 1D 向量
        flat = adj.view(-1)

        # 2. 找出前 50% 最大值的阈值
        k = int(flat.numel() * 0.5)
        threshold = torch.topk(flat, k, largest=True).values[-1]  # 取第k大的值作为阈值

        # 3. 将小于该阈值的值置 0，但保留为 dense tensor（仍然是 116×116）
        sparse_like_adj = torch.where(adj >= threshold, adj, torch.zeros_like(adj))
        '''max_val = torch.max(sparse_like_adj)
        min_val = torch.min(sparse_like_adj)
        mean_val = torch.mean(sparse_like_adj)

        print("coupling: Max:", max_val.item())
        print("coupling: Min:", min_val.item())
        print("coupling: Mean:", mean_val.item())'''
        return sparse_like_adj, data_DTI_norm, data_fMRI_norm,adj_DTI, adj_fMRI

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
        #print("self.input_dim, self.output_dim, input_feature.shape", self.input_dim, self.output_dim, input_feature.shape)
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

def preprocess(data):
    Adj = []
    for i in range(len(data)):
        pc = np.corrcoef(data.cpu()[i].T)  # (116,116)
        pc = np.nan_to_num(pc)
        pc = abs(pc)
        Adj.append(pc)
    adj = torch.from_numpy(np.array(Adj))
    fea = adj
    return adj,fea

class Module_DTI(nn.Module):
    def __init__(self, input_dim, hidden_dim, c, a_DTI, dropout):
        """

        Args:
        ----
            input_dim: input dimension
            hidden_dim: output dimension
            num_classes: category number (default: 2)
        """
        super(Module_DTI, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.c = c
        self.a_DTI = a_DTI
        self.dropout = dropout
        self.fkernel = FKernel(self.c)  #这里也加了双曲核
        self.cos = Cosine()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        #self.transformer = TransformerModel(feature_size, num_layers, nhead)

    def forward(self, data,adj):
        f = data
        a = adj
        a = a.cpu().numpy()#(nbatch,nroi,nroi)
        adj = scipy.linalg.block_diag(*abs(a))  # (nbatch*nroi,nbatch*nroi) 构建对角矩阵
        
        adj_csr = sp.csr_matrix(adj) #将adj转换为稀疏矩阵存储
        adj_nor = normalization(adj_csr).cuda() #归一化邻接矩阵
        adj_nor = adj_nor.to(torch.float32)
        fea = rearrange(f, 'a b c-> (a b) c').cuda()#(nbatch*nroi,nroi) 维度重排，将f从（a，b,c）维度展成（a*b，c）的维度
        fea = fea.to(torch.float32)
        #gcn1 = F.relu(self.gcn1(adj_nor, fea))  #(nbatch*nroi,hiddendim)# (N,hidden_dim)
        #gcn2 = F.relu(self.gcn2(adj_nor, gcn1)) #(nbatch*nroi,hiddendim)   
        x1 =  self.gcn1(adj_nor, fea) 
        #x1 = self.bn1(x1)  # ✅ BatchNorm 加在 GCN 后、激活前 
        x1 = self.fkernel(x1)
        gcn1 = F.relu(x1) + self.a_DTI*self.cos(x1)  #(nbatch*nroi,hiddendim)# (N,hidden_dim)
        gcn1 = F.dropout(gcn1, p=self.dropout, training=self.training)  # ✅ Dropout 添加
        
        x2 = self.gcn2(adj_nor, gcn1)
        #x2 = self.bn2(x2)  # ✅ 第二层也同理
        x2 = self.fkernel(x2)
        gcn2 = F.relu(x2) + self.a_DTI*self.cos(x2) #(nbatch*nroi,hiddendim)
        gcn2 = F.dropout(gcn2, p=self.dropout, training=self.training)  # ✅ Dropout 添加
        x = rearrange(gcn2, '(b n) c -> b n c', b=int(len(adj_nor) / a.shape[1]), n= a.shape[1])
        return x#  (nbatch*nroi,hiddendim) #将形状为(nbatch*nroi,hiddendim)的张量重新排为(nbatch，nroi,hiddendim)


class Module_fMRI(nn.Module):
    def __init__(self, input_dim, hidden_dim, c, a_fMRI, dropout, pretrained_path = None):
        """

        Args:
        ----
            input_dim: input dimension
            hidden_dim: output dimension
            num_classes: category number (default: 2)
        """
        super(Module_fMRI, self).__init__()
        self.pretrained_path = pretrained_path
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.c = c
        self.a_fMRI = a_fMRI
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.fkernel = FKernel(self.c)  #这里也加了双曲核
        self.cos = Cosine_fMRI()

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
        a_initial,f_initial = preprocess(data)
        adj = a_initial 
        f =  f_initial
        
        #fMRI进行稀疏化
        # 1. 将整个矩阵展平成 1D 向量
        flat = adj.view(-1)

        # 2. 找出前 50% 最大值的阈值
        k = int(flat.numel() * 0.5)
        threshold = torch.topk(flat, k, largest=True).values[-1]  # 取第k大的值作为阈值

        # 3. 将小于该阈值的值置 0，但保留为 dense tensor（仍然是 116×116）
        sparse_like_adj = torch.where(adj >= threshold, adj, torch.zeros_like(adj))
        '''max_val = torch.max(sparse_like_adj)
        min_val = torch.min(sparse_like_adj)
        mean_val = torch.mean(sparse_like_adj)

        print("fMRI: Max:", max_val.item())
        print("fMRI: Min:", min_val.item())
        print("fMRI: Mean:", mean_val.item())'''

        # a(nbatch,116,116) f (nbatch,nroi,ninputdim)
        adj = sparse_like_adj.cpu().numpy()#(nbatch,nroi,nroi)
        adj = scipy.linalg.block_diag(*abs(adj))  # (nbatch*nroi,nbatch*nroi)
        adj_csr = sp.csr_matrix(adj)
        adj_nor = normalization(adj_csr).cuda()
        adj_nor = adj_nor.to(torch.float32)
        fea = rearrange(f, 'a b c-> (a b) c').cuda()#(nbatch*nroi,nroi)
        fea = fea.to(torch.float32)  
        
        x1 = self.gcn1(adj_nor, fea) 
        #x1 = self.bn1(x1)  # ✅ BatchNorm 加在 GCN 后、激活前 
        '''print("🔁 正式 forward 时 gcn1.weight 的前5个元素：")
        print('self.gcn1.weight.view', self.gcn1.weight.view(-1)[:5].detach().cpu())
        print('self.gcn1.weight.grad', self.gcn1.weight.grad)
        print('self.gcn1.weight.is_leaf', self.gcn1.weight.is_leaf)
        print('self.gcn1.weight.requires_grad', self.gcn1.weight.requires_grad)
        print('self.gcn1.weight.grad_fn', self.gcn1.weight.grad_fn)'''

        x1 = self.fkernel(x1)
        gcn1 = F.relu(x1) + self.a_fMRI*self.cos(x1)  #(nbatch*nroi,hiddendim)# (N,hidden_dim)
        gcn1 = F.dropout(gcn1, p=self.dropout, training=self.training)  # ✅ Dropout 添加

        x2 = self.gcn2(adj_nor, gcn1)
        #x2 = self.bn2(x2)  # ✅ 第二层也同理
        x2 = self.fkernel(x2)
        gcn2 = F.relu(x2) + self.a_fMRI*self.cos(x2) #(nbatch*nroi,hiddendim)
        gcn2 = F.dropout(gcn2, p=self.dropout, training=self.training)  # ✅ Dropout 添加
        x = rearrange(gcn2, '(b n) c -> b n c', b=int(len(adj_nor) / a_initial.shape[1]), n= a_initial.shape[1])
        return x,a_initial#  (nbatch*nroi,hiddendim) #将形状为(nbatch*nroi,hiddendim)的张量重新排为(nbatch，nroi,hiddendim)
