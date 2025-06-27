import torch
import torch.nn as nn
import manifolds
from utils.math_utils import tanh

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        self.dim = args.dim
        self.c = args.c
        self.manifold_name = args.manifold
        self.manifold = getattr(manifolds, self.manifold_name)()
        #self.f_k = nn.Bilinear(args.dim, args.dim, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def f_k(self, p1, p2, gamma=1.0, theta=-1.0): ### hyperbolic kernel
        inner_product_function = nn.Bilinear(self.dim, self.dim, 1).to(self.args.device)

        if self.manifold_name == 'PoincareBall':
            inner_product = inner_product_function(p1, p2).to(self.args.device)
            #inner_product = torch.matmul(p1, p2).to(self.args.device) #YMM
            poincare_sig_kernel = torch.tanh(gamma / (1 - self.c*inner_product) + theta).to(self.args.device)
            result = poincare_sig_kernel

        elif self.manifold_name == 'Hyperboloid':
            e_p1 = self.manifold.logmap0(p1.squeeze(), c=self.c).to(self.args.device) #将数据映射到切空间上
            e_p2 = self.manifold.logmap0(p2.squeeze(), c=self.c).to(self.args.device) #将数据映射到切空间上

            if not self.args.cuda == -1:
                e_p1 = e_p1.to(self.args.device)
                e_p2 = e_p2.to(self.args.device)
                inner_product_function = inner_product_function.to(self.args.device) #bilinear是什么意思？

            inner_product = inner_product_function(e_p1, e_p2)
          
            #inner_product = torch.matmul(e_p1, e_p2) #YMM
            tau = 25.0
            lorentz_sig_kernel = torch.tanh(gamma * inner_product + theta)

            d = 2.0
            lorentz_poly_kernel = (gamma * inner_product + 1)**d

            result = torch.unsqueeze(lorentz_sig_kernel, 0)
            #result = lorentz_sig_kernel

        return result

    #def forward(self, c_1, c_2, c_da_1, c_da_2, h_1_pl, h_2_pl, h_1_mi, h_2_mi):   ### c:图级表���，h_pl:�������������点表征，h_mi:特征损坏后的节点表征
    def forward(self, c_1, c_2, h_1_pl, h_2_pl, h_1_mi, h_2_mi):
        if c_1 == None and c_2 == None:
            sc_1 = torch.squeeze(self.f_k(h_2_pl, h_1_pl), 2).to(self.args.device)
            sc_2 = torch.squeeze(self.f_k(h_1_pl, h_2_pl), 2).to(self.args.device)
            sc_3 = torch.squeeze(self.f_k(h_2_mi, h_1_pl), 2).to(self.args.device)
            sc_4 = torch.squeeze(self.f_k(h_1_mi, h_2_pl), 2).to(self.args.device)
            logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)


        elif c_1 != None and c_2 == None:
            c_x = torch.unsqueeze(c_1, 1).to(self.args.device)
            c_x = c_x.expand_as(h_1_pl).to(self.args.device)
            #c_x_da = torch.unsqueeze(c_da_1, 1)
            #c_x_da = c_x_da.expand_as(h_1_pl)

            sc_1 = torch.squeeze(self.f_k(h_1_pl, c_x), 2).to(self.args.device)  ### 正样本：图表征与原节点表征
            sc_2 = torch.squeeze(self.f_k(h_1_mi, c_x), 2).to(self.args.device)  ### 负样本1：图表征与特征损�������后的节点表征
            #sc_3 = torch.squeeze(self.f_k(h_1_pl, c_x_da), 2)  ### ������������本2：特征损坏后的图表���与节点表征

            #logits = torch.cat((sc_1, sc_2, sc_3), 1)
            logits = torch.cat((sc_1, sc_2), 1)

        else:
            c_x_1 = torch.unsqueeze(c_1, 1).to(self.args.device)
            c_x_1 = c_x_1.expand_as(h_1_pl).to(self.args.device)
            #c_x_da_1 = torch.unsqueeze(c_da_1, 1)
            #c_x_da_1 = c_x_da_1.expand_as(h_1_pl)

            c_x_2 = torch.unsqueeze(c_2, 1).to(self.args.device)
            c_x_2 = c_x_2.expand_as(h_2_pl).to(self.args.device)
            #c_x_da_2 = torch.unsqueeze(c_da_2, 1)
            #c_x_da_2 = c_x_da_2.expand_as(h_2_pl)

            sc_1 = torch.squeeze(self.f_k(h_2_pl, c_x_1), 2).to(self.args.device)  ### 正样本：原图表征与对比节点表征；对比图表征与原节点表征
            sc_2 = torch.squeeze(self.f_k(h_1_pl, c_x_2), 2).to(self.args.device)
            sc_3 = torch.squeeze(self.f_k(h_2_mi, c_x_1), 2).to(self.args.device)  ### 负样本1：原���������表征与特征损坏后的对比节点表征；对比图表征与损坏原��点��征
            sc_4 = torch.squeeze(self.f_k(h_1_mi, c_x_2), 2).to(self.args.device)
            #sc_5 = torch.squeeze(self.f_k(h_2_pl, c_x_da_1), 2)  ### 负样本2：损坏原图表征与对比节点表征；损坏对比图表征与原节点���征
            #sc_6 = torch.squeeze(self.f_k(h_1_pl, c_x_da_2), 2)

            #logits = torch.cat((sc_1, sc_2, sc_3, sc_4, sc_5, sc_6), 1)
            logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)

        return logits

