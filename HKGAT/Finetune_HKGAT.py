#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:58:05 2024

@author: mmy
"""
#Note that please input your to-be-analyzed data in the class named Data, including bold with shape of (nsub,nlength,nroi) and label with the shape of (nsub,)
import warnings
warnings.filterwarnings('ignore')
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
#import model_finetune
import torch
import numpy as np
import random
from sklearn.model_selection import KFold
import argparse
import scipy
from scipy.io import loadmat
import time
import pickle
from pmath import project, logmap0
from sklearn.metrics import roc_curve, auc
from datetime import datetime
import torch.nn.functional as F 

from einops import repeat, rearrange, reduce


##############################Parameter Setting ##############################
parser = argparse.ArgumentParser(description='PyTorch Finetne')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batchsize', default=64, type=int, metavar='N',
                    help='the batch size to use for training and testing')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')  # ？
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--dim', default=64, type=int,
                    help='feature dimension (default: 64)')
parser.add_argument('--pred-dim', default=32, type=int,
                    help='hidden dimension of the predictor (default: 32)')
parser.add_argument('--num_classes', default=2, type=int,
                    help='number of classes (default: 2)')
parser.add_argument('--dropout', default=0.3, type=float, 
                    help='dropout value of weight')
parser.add_argument('--threshold_scale', default=1.0, type=float, 
                    help='the scaling parementer for adjusting thredshold of the acc prediction')
parser.add_argument('--c', default=0.0001, type=float, 
                    help='curvature value of the hyperbolic model')
parser.add_argument('--a', default=0.01, type=float, 
                    help='rescale factor of cosine activate function')
parser.add_argument('--a_coupling', default=0.01, type=float, 
                    help='rescale factor of cosine activate function')
parser.add_argument('--a_DTI', default=0.01, type=float, 
                    help='rescale factor of cosine activate function')
parser.add_argument('--a_fMRI', default=0.01, type=float, 
                    help='rescale factor of cosine activate function')
parser.add_argument('--weight_fMRI', default=1.0, type=float, 
                    help='weight of fMRI')
parser.add_argument('--weight_DTI', default=1.0, type=float, 
                    help='weight of DTI')
parser.add_argument('--nhead', default=1, type=int, 
                    help='number of attention heads')      
parser.add_argument('--num_layer', default=2, type=int, 
                    help='number of transformer layers')             
parser.add_argument('--use_att', default=False, type=bool, help='whether to use hyperbolic attention or not'),
parser.add_argument('--use_bias', default=True, type=bool, help='whether to use bias'),
parser.add_argument('--local_agg', default=True, type=bool, help='whether to use local agg'),
parser.add_argument('--manifold',  
                    choices= ['PoincareBall','Hyperboloid', 'Euclidean'], 
                    default='PoincareBall',
                    help='which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),



parser.add_argument('--encoder_type',
                    choices=['HKGAT','HKGCN','GAT', 'GIN', 'GCN','BrainNetCNN','Transformer'],
                    default='HKGAT',  # Change the default to 'gat'
                    help="Choose the type of GNN encoder: 'gcn', 'gin', or 'gat' (default: 'gat')")


args = parser.parse_args()



#定义数据加载类
class Data(object):
    def read_data(self):
        np.random.seed(42)

        ######### 模拟 fMRI 数据 ###########
        n_samples = 137
        time_points = 230
        n_regions = 116
        data_fMRI = np.random.rand(n_samples, time_points, n_regions).astype(np.float32)

        # 模拟标签：一半 0 一半 1
        y_fMRI = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
        np.random.shuffle(y_fMRI)

        ######### 模拟 DTI 数据 ###########
        # 三个结构连接矩阵：(137, 116, 116)
        fn_array = np.random.rand(n_samples, n_regions, n_regions).astype(np.float32)
        fa_array = np.random.rand(n_samples, n_regions, n_regions).astype(np.float32)
        le_array = np.random.rand(n_samples, n_regions, n_regions).astype(np.float32)

        y_DTI = y_fMRI.copy()  # 保证标签一致
        fn_tensor = torch.tensor(fn_array, dtype=torch.float32)
        fa_tensor = torch.tensor(fa_array, dtype=torch.float32)
        le_tensor = torch.tensor(le_array, dtype=torch.float32)

        data_DTI = torch.cat((fn_tensor, fa_tensor, le_tensor), dim=-1)
        adj_DTI = fn_tensor + fa_tensor + le_tensor

        print('fn_shape', fn_array.shape, 'fa_shape', fa_array.shape, 'le_shape', le_array.shape)
        print('fa_array[0].shape', fa_array[0].shape)
        print('fa_array.shape', fa_array.shape, 'fa_type', fa_array.dtype)
        print('data_DTI.shape', data_DTI.shape, 'data_DTI_type', data_DTI.dtype)
        print('y_DTI.shape', y_DTI.shape, 'y_DTI_type', y_DTI.dtype)
        print("Determine label consistency", (y_fMRI == y_DTI).sum())

        return data_fMRI, data_DTI.numpy(), adj_DTI, y_DTI

    def __init__(self):
        super(Data, self).__init__()
        data_fMRI, data_DTI, adj_DTI, y_DTI = self.read_data()

        self.data_fMRI = torch.from_numpy(data_fMRI)
        self.data_DTI = torch.from_numpy(data_DTI)
        self.adj_DTI = adj_DTI
        self.y_DTI = torch.from_numpy(y_DTI)
        self.n_samples = data_fMRI.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.data_fMRI[index], self.data_DTI[index], self.adj_DTI[index], self.y_DTI[index], index



# 定义动态阈值选择函数
def find_optimal_threshold(y_true, y_scores, threshold_scale = args.threshold_scale ):
    """
    通过 Youden's J statistic 找到最优阈值
    """
    fpr, tpr, thresholds = roc_curve(y_true.cpu().numpy(), y_scores.cpu().numpy())
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    print(f"best threshold: {optimal_threshold}")
    adjust_threshold = threshold_scale * optimal_threshold
    print(f"adjust threshold: {adjust_threshold}")
    return adjust_threshold


def calculate_metric(gt, pred, threshold=0.5):
    """
    Calculate classification metrics.
    """
    # 使用指定的阈值进行分类
    pred = (pred > threshold).int()
    
    # 检查标签是否符合要求
    assert set(gt.cpu().numpy()).issubset({0, 1}), "gt must only contain 0 and 1"
    assert set(pred.cpu().numpy()).issubset({0, 1}), "pred must only contain 0 and 1"
    
    # 计算混淆矩阵
    confusion = confusion_matrix(gt.cpu().numpy(), pred.cpu().numpy())
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    
    # 计算指标
    acc = (TP + TN) / float(TP + TN + FP + FN)
    sen = TP / float(TP + FN) if (TP + FN) > 0 else 0
    spe = TN / float(TN + FP) if (TN + FP) > 0 else 0
    bac = (sen + spe) / 2
    pre = TP / float(TP + FP) if (TP + FP) > 0 else 0
    rec = TP / float(TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0

    # 打印调试信息
    print(f'Threshold: {threshold}')
    print(f'TP={TP}, TN={TN}, FP={FP}, FN={FN}')
    
    return acc, sen, spe, bac, pre, f1_score


# 将数据分组并保存的函数
def save_grouped_data(data, identifier, directory, fold):
    # 计算四种情况的索引
    true_positives = (y_true == 1) & (y_pred == 1)
    true_negatives = (y_true == 0) & (y_pred == 0)
    false_positives = (y_true == 0) & (y_pred == 1)
    false_negatives = (y_true == 1) & (y_pred == 0)

    # 分组保存
    for group, idx in zip(["TP", "TN", "FP", "FN"], [true_positives, true_negatives, false_positives, false_negatives]):
        group_data = data[idx]
        torch.save(group_data, os.path.join(directory, f"{identifier}_{group}_{fold}.pt"))

def preprocess(data):
    Adj = []
    for i in range(len(data)):
        #Pearson correlation coefficient matrix.
        #The Pearson correlation between two features measures the linear relationship between them, and its value ranges from -1 to 1
        pc = np.corrcoef(data.cpu()[i].T)  # (116,116)
        pc = np.nan_to_num(pc)
        # focus only on the magnitude of the relationship, ignoring whether it's positive or negative
        pc = abs(pc)
        Adj.append(pc)
    adj = torch.from_numpy(np.array(Adj))
    fea = adj
    return adj,fea


################################ functions for model finetuning functions################################

HGNNEncoder = getattr(__import__('{}_coupling_encoder'.format(args.encoder_type)), 'HGNNEncoder') #获取里面Module1的变量值并赋给Module1
print(HGNNEncoder)


class FKernel(torch.nn.Module):
    def __init__(self, c):
        super(FKernel, self).__init__()
        #self.device = device
        self.c = c
    def forward(self, x):
        output = project(x, c=self.c)
        output = logmap0(output, c=self.c)
        return output



class Model(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, dim, pred_dim, c, a,a_couping, a_fMRI, a_DTI, weight_fMRI, weight_DTI, pretrained_path=None):
        """
        dim: feature dimension
        pred_dim: hidden dimension of the predictor 
        """
        super(Model, self).__init__()
        self.c = c
        self.a = a

        self.encoder = HGNNEncoder(116, dim, c, a_couping, a_fMRI, a_DTI, weight_fMRI,weight_DTI, pretrained_path)

        self.fkernel = FKernel(self.c)


        self.predictor = nn.Sequential()
        self.predictor.add_module('FKernel', FKernel(self.c))
        self.predictor.add_module('L1', nn.Linear(dim, pred_dim, bias=False))
        self.predictor.add_module('BN', nn.BatchNorm1d(pred_dim))
        self.predictor.add_module('RL', nn.ReLU(inplace=True))  #Predictor这边
        self.predictor.add_module('FKernel', FKernel(self.c))
        self.predictor.add_module('L2', nn.Linear(pred_dim, 2))  # output layer

    def forward(self, DTI,adj_DTI, fMRI):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        fea_coupled, adj_coupled, DTI_HKGAT,fMRI_HKGAT,adj_DTI,adj_fMRI   = self.encoder(DTI,adj_DTI, fMRI) 
        z = reduce(fea_coupled, 'b n c ->b c', 'mean') 

        p = self.predictor(z)  
        output = F.softmax(p, dim=1)  
        return z, p, output, fea_coupled, adj_coupled, DTI_HKGAT,fMRI_HKGAT,adj_DTI,adj_fMRI 

ACC = []
SEN = []
SPE = []
BAC = []
PRE = []
F1_SCORE = []
AUC = []
 



for seed in [1,2,3,4,5]: 


    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Arguments and their values:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    #数据记录
    if torch.cuda.is_available():
            print("CUDA IS AVAILABLE")

    save_path1 = "_".join(
        [
            str(args.encoder_type),
            str(args.batchsize),
            str(args.lr),
            str(seed),
            str(args.dim),
            str(args.c),
            str(args.nhead),
            str(args.num_layer),
            str(args.weight_fMRI),
            str(args.weight_DTI),
            str(args.epochs),
        ]
    )

    save_path = "./result/coupling_HKGAT/"


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S") 

    result_dir_csv = save_path +save_path1 + str(time_str) + ".csv"

    print('result_dir_csv:',result_dir_csv)

    #保存中间过程数据和模型
    final_dir = save_path + "/final_results" + time_str
    os.makedirs(final_dir, exist_ok=True)  # 如果文件夹不存在则创建      
        
        
    with open(result_dir_csv, 'a') as f:
        f.write("#####################################################################\n")

    with open(result_dir_csv, 'a') as f:
        f.write("Arguments and their values:")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")


    #进行5折交叉验证
    full_dataset = Data()



    k = 5
    kfold = KFold(n_splits=k, random_state=seed, shuffle=True)

    Acc2 = []
    Sen2 = []
    Spe2 = []
    Bac2 = []
    Pre2 = []
    F1_score2 = []
    Auc2 = []


    #训练和评估模型
    for fold, (train_idx, test_idx) in enumerate(kfold.split(full_dataset)):
        print('------------fold no---------{}----------------------'.format(fold))
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
        final_fold_dir = os.path.join(final_dir, f"fold_{fold}")  # 按照fold创建子文件夹
        os.makedirs(final_fold_dir, exist_ok=True)  # 如果文件夹不存在则创建

        training_data_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=args.batchsize, sampler=train_subsampler)  # 16,64
        testing_data_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=args.batchsize, sampler=test_subsampler)
        from sklearn.metrics import confusion_matrix
        
        ##############################To load the pretrained model ##############################
        pretrained='Pretrained_HKGAT_019.pth.tar'.format(args.encoder_type) #这几个提前训练好的压缩文件是什么？


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'DEVICE: {device}')
        model =Model(args.dim, args.pred_dim,args.c, args.a, args.a_coupling, args.a_DTI, args.a_fMRI, args.weight_fMRI, args.weight_DTI,  pretrained_path = pretrained)  # args.pred_dim

        model.to(device)
        print(model)
        #  Initialize weights with normal distribution (mean=0.0, std=0.01)
        model.predictor.L1.weight.data.normal_(mean=0.0, std=0.01)
        model.predictor.BN.weight.data.normal_(mean=0.0, std=0.01)
        model.predictor.L2.weight.data.normal_(mean=0.0, std=0.01)
        ## Initialize biases to zero
        model.predictor.BN.bias.data.zero_()
        model.predictor.L2.bias.data.zero_()

  
        criterion = nn.CrossEntropyLoss()
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        torch.manual_seed(seed)
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

        cudnn.benchmark = True
        ###This code is used for freeze the encoder
        # for param in  model.encoder.parameters():
        #     param.requires_grad = False
        grad = any(param.requires_grad for param in model.encoder.parameters())
        print(grad)
        ##############################Model Finetuning ##############################
        for epoch in range(args.start_epoch, args.epochs):
            # print(epoch)
            train_acc = 0.0
            train_loss = 0.0
            test_acc = 0.0
            test_loss = 0.0
            # training
            model.train()
            for i, data in enumerate(training_data_loader):
                bold, data_DTI, adj_DTI, label, index = data
                bold = bold.to(device)
                data_DTI = data_DTI.to(device)
                adj_DTI = adj_DTI.to(device)

                label = label.to(device)
                optimizer.zero_grad()

                z, p, outputs, fea_coupled, adj_coupled, DTI_HKGAT,fMRI_HKGAT,adj_DTI,adj_fMRI = model(data_DTI, adj_DTI,bold)

                #用的是output
                batch_loss = criterion(outputs, label.long())
                _, train_pred = torch.max(outputs, 1)
                batch_loss.backward()
                optimizer.step()
                train_acc += (train_pred.cpu() == label.cpu()).sum().item()
                train_loss += batch_loss.item()


            All_index =[]
            Labels = []
            Test_pred = []
            Pre_score = []
            All_DTI_initial = []
            All_fMRI_initial = []
            All_z = []
            All_p = []
            All_output = []
            All_fea_coupled =[]
            All_adj_coupled = []
            All_DTI_HKGAT = []
            All_fMRI_HKGAT = []

            model.eval() #在测试过程中是直关闭dropout layer 和batchnorm layer的吗
            with torch.no_grad():
                for i, data in enumerate(testing_data_loader):
                    bold,data_DTI,adj_DTI, label, index = data  # torch.Size([32, 232, 116])
                    bold = bold.to(device)
                    adj_DTI = adj_DTI.to(device)
                    data_DTI = data_DTI.to(device)

                    label = label.to(device)
                    Labels.append(label)


                    z, p, output, fea_coupled, adj_coupled, DTI_HKGAT,fMRI_HKGAT,adj_DTI,adj_fMRI = model(data_DTI, adj_DTI,bold)
                    All_index.append(index.cpu())
                    All_DTI_initial.append(data_DTI.cpu())
                    All_fMRI_initial.append(bold.cpu())
                    All_z.append(z.cpu())
                    All_p.append(p.cpu())
                    All_output.append(output.cpu())
                    All_fea_coupled.append(fea_coupled.cpu())
                    All_adj_coupled.append(adj_coupled.cpu())
                    All_DTI_HKGAT.append(DTI_HKGAT.cpu())
                    All_fMRI_HKGAT.append(fMRI_HKGAT.cpu())

                    batch_loss = criterion(output, label.long())
                    pre_score = output[:, 1]
                    Pre_score.append(pre_score)
                    PPre_score = torch.cat(Pre_score, -1).cpu()

                    _, test_pred = torch.max(output, 1)
                    Test_pred.append(test_pred)
                    test_acc += (
                            test_pred.cpu() == label.cpu()).sum().item()  # get the index of the class with the highest probability
                    test_loss += batch_loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Test Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, args.epochs, train_acc / len(train_idx), train_loss / len(training_data_loader),
                    test_acc / len(test_idx), test_loss / len(testing_data_loader)))
                with open(result_dir_csv, 'a') as f:
                    f.write('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Test Acc: {:3.6f} loss: {:3.6f}\n'.format(
                    epoch + 1, args.epochs, train_acc / len(train_idx), train_loss / len(training_data_loader),
                    test_acc / len(test_idx), test_loss / len(testing_data_loader)) )

                All_index = torch.cat(All_index, dim=0)
                All_DTI_initial = torch.cat(All_DTI_initial, dim=0)
                All_fMRI_initial = torch.cat(All_fMRI_initial, dim=0)
                y_true = torch.cat(Labels, -1).cpu()
                PPre_score = torch.cat(Pre_score, -1).cpu()
                All_p = torch.cat(All_p, dim=0)
                All_z = torch.cat(All_z, dim=0)
                All_output = torch.cat(All_output, dim=0)
                All_fea_coupled = torch.cat(All_fea_coupled, dim=0)
                All_adj_coupled = torch.cat(All_adj_coupled, dim=0) 
                All_DTI_HKGAT = torch.cat(All_DTI_HKGAT, dim=0)
                All_fMRI_HKGAT = torch.cat(All_fMRI_HKGAT, dim=0)
                 
                 
         # 保存最终 epoch 的结果
            if epoch == args.epochs - 1:
                model_final_state = model.state_dict()  
                final_model_name = f"final_model_epoch_{epoch + 1}.pth"  
                model_final_path = os.path.join(final_fold_dir, final_model_name)
                final_threshold = find_optimal_threshold(y_true, PPre_score)
                y_pred = (PPre_score > final_threshold).int()


                torch.save(model_final_state, model_final_path)  
                # 保存最终模型文件名信息
                if isinstance(adj_coupled, np.ndarray):  
                    adj_coupled = torch.from_numpy(adj_coupled)
                final_results = {
                    "index": All_index.cpu(),
                    "DTI_initial": All_DTI_initial.cpu(),
                    "fMRI_initial": All_fMRI_initial.cpu(),
                    "y_true": y_true.cpu(),
                    "y_pred" : y_pred.cpu(),
                    "PPre_score":PPre_score.cpu(),
                    "DTI_HKGAT": All_DTI_HKGAT.cpu(),
                    "fMRI_HKGAT": All_fMRI_HKGAT.cpu(),
                    "p":All_p.cpu(),
                    "z":All_z.cpu(),
                    "fea_coupled":All_fea_coupled.cpu(),
                    "adj_coupled": All_adj_coupled.cpu(),
                    "output":All_output.cpu(),  
                }
                 
                for key in ["index", "DTI_initial", "fMRI_initial", "y_true", "y_pred", "DTI_HKGAT", "fMRI_HKGAT","p","z","fea_coupled", "adj_coupled","output"]:
                        data_classified = final_results[key]
                        save_grouped_data(data_classified, key, final_fold_dir,fold)
                                    
                
                # 计算最优阈值
                optimal_threshold = find_optimal_threshold(y_true, PPre_score, threshold_scale = args.threshold_scale)
                y_pred = (PPre_score > optimal_threshold).int()

        for key, value in final_results.items():
            torch.save(value, os.path.join(final_fold_dir, f"{key}_final_{fold}.pt"))

        print('PPre_score=',PPre_score)
        print('y_pred=',y_pred)

        acc, sen, spe, bac, pre, f1_score = calculate_metric(y_true, y_pred)
        from sklearn import metrics

        fpr, tpr, threshold = metrics.roc_curve(y_true, PPre_score)
        auc = metrics.auc(fpr, tpr)
        Acc2.append(acc)
        Sen2.append(sen)
        Spe2.append(spe)
        Bac2.append(bac)
        Pre2.append(pre)
        F1_score2.append(f1_score)
        Auc2.append(auc)

    k=5
    avg_Acc = sum(Acc2) / k
    std_Acc = np.std(Acc2, ddof=1)
    print('Acc mean',avg_Acc)
    with open(result_dir_csv, 'a') as f:
        f.write("Acc mean={:.4f}\n".format(avg_Acc))
    print('Acc std', std_Acc)
    with open(result_dir_csv, 'a') as f:
        f.write("Acc std={:.4f}\n".format(std_Acc))

    avg_Sen = sum(Sen2) / k
    std_Sen = np.std(Sen2, ddof=1)
    print('Sen mean',avg_Sen)
    print('Sen std', std_Sen)
    with open(result_dir_csv, 'a') as f:
        f.write("Sen mean={:.4f}\n".format(avg_Sen))

    with open(result_dir_csv, 'a') as f:
        f.write("Sen std={:.4f}\n".format(std_Sen))

    avg_Spe = sum(Spe2) / k
    std_Spe = np.std(Spe2, ddof=1)
    print('Spe mean', avg_Spe)
    print('Spe std', std_Spe)
    with open(result_dir_csv, 'a') as f:
        f.write("Spe mean={:.4f}\n".format(avg_Spe))
    with open(result_dir_csv, 'a') as f:
        f.write("Spe std={:.4f}\n".format(std_Spe))

    avg_Bac = sum(Bac2) / k
    std_Bac = np.std(Bac2, ddof=1)
    print('Bac mean',avg_Bac)
    print('Bac std', std_Bac)
    with open(result_dir_csv, 'a') as f:
        f.write("Bac mean={:.4f}\n".format(avg_Bac))
    with open(result_dir_csv, 'a') as f:
        f.write("Bac std={:.4f}\n".format(std_Bac))

    avg_Pre = sum(Pre2) / k
    std_Pre = np.std(Pre2, ddof=1)
    print('Pre mean', avg_Pre)
    print('Pre std', std_Pre)
    with open(result_dir_csv, 'a') as f:
        f.write("Pre mean={:.4f}\n".format(avg_Pre))
    with open(result_dir_csv, 'a') as f:
        f.write("Pre std={:.4f}\n".format(std_Pre))

    avg_F1_score = sum(F1_score2) / k
    std_F1_score = np.std(F1_score2, ddof=1)
    print('F1 mean',avg_F1_score)
    print('F1_score std', std_F1_score)
    with open(result_dir_csv, 'a') as f:
        f.write("F1 mean={:.4f}\n".format(avg_F1_score))
    with open(result_dir_csv, 'a') as f:
        f.write("F1_score std={:.4f}\n".format(std_F1_score))

    avg_Auc = sum(Auc2) / k
    std_Auc = np.std(Auc2, ddof=1)
    print('Auc mean', avg_Auc)
    print('Auc std', std_Auc)
    with open(result_dir_csv, 'a') as f:
        f.write("Auc mean={:.4f}\n".format(avg_Auc))
    with open(result_dir_csv, 'a') as f:
        f.write("Auc std={:.4f}\n".format(std_Auc))



    
    ACC.extend([avg_Acc])
    SEN.extend([avg_Sen])
    SPE.extend([avg_Spe])
    BAC.extend([avg_Bac])
    #PPV.extend([avg_Ppv])
    #NPV.extend([avg_Npv])
    PRE.extend([avg_Pre])
    #REC.extend([avg_Rec])
    F1_SCORE.extend([avg_F1_score])
    AUC.extend([avg_Auc])

with open(result_dir_csv, 'a') as f:
    f.write("#####################################################################\n")

print("#####################################################################\n")

print('Mean ACC:', np.mean(ACC))
with open(result_dir_csv, 'a') as f:
    f.write("Mean ACC = {:.4f}\n".format(np.mean(ACC)))
print('ACC std:', np.std(ACC, ddof=1))
with open(result_dir_csv, 'a') as f:
    f.write("ACC std = {:.4f}\n".format(np.std(ACC, ddof=1)))

print('Mean SEN:', np.mean(SEN))
with open(result_dir_csv, 'a') as f:
    f.write("Mean SEN = {:.4f}\n".format(np.mean(SEN)))
print('SEN std:', np.std(SEN, ddof=1))
with open(result_dir_csv, 'a') as f:
    f.write("SEN std = {:.4f}\n".format(np.std(SEN, ddof=1)))

print('Mean SPE:', np.mean(SPE))
with open(result_dir_csv, 'a') as f:
    f.write("Mean SPE = {:.4f}\n".format(np.mean(SPE)))
print('SPE std:', np.std(SPE, ddof=1))
with open(result_dir_csv, 'a') as f:
    f.write("SPE std = {:.4f}\n".format(np.std(SPE, ddof=1)))
print('Mean BAC:', np.mean(BAC))
with open(result_dir_csv, 'a') as f:
    f.write("Mean BAC = {:.4f}\n".format(np.mean(BAC)))
print('BAC std:', np.std(BAC, ddof=1))
with open(result_dir_csv, 'a') as f:
    f.write("BAC std = {:.4f}\n".format(np.std(BAC, ddof=1)))
#print('Mean PPV:', np.mean(PPV))
#print('PPV std:', np.std(PPV, ddof=1))

#print('Mean NPV:', np.mean(NPV))
#print('NPV std:', np.std(NPV, ddof=1))

print('Mean PRE:', np.mean(PRE))
with open(result_dir_csv, 'a') as f:
    f.write("Mean PRE = {:.4f}\n".format(np.mean(PRE)))
print('STD PRE:', np.std(PRE, ddof=1))
with open(result_dir_csv, 'a') as f:
    f.write("STD PRE = {:.4f}\n".format(np.std(PRE, ddof=1)))
#print('Mean REC:', np.mean(REC))
#print('REC std:', np.std(REC, ddof=1))

print('Mean F1_SCORE:', np.mean(F1_SCORE))
with open(result_dir_csv, 'a') as f:
    f.write("Mean F1_SCORE = {:.4f}\n".format(np.mean(F1_SCORE)))
print('F1_SCORE std:', np.std(F1_SCORE, ddof=1))
with open(result_dir_csv, 'a') as f:
    f.write("F1_SCORE std = {:.4f}\n".format(np.std(F1_SCORE, ddof=1)))

print('Mean AUC:', np.mean(AUC))
with open(result_dir_csv, 'a') as f:
    f.write("Mean AUC = {:.4f}\n".format(np.mean(AUC)))
print('AUC std:', np.std(AUC, ddof=1))
with open(result_dir_csv, 'a') as f:
    f.write("AUC std = {:.4f}\n".format(np.std(AUC, ddof=1)))


