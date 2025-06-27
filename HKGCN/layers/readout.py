import torch
import torch.nn as nn

#def to_hyperboloid(x, c):
#    K = 1./ c
#    sqrtK = K ** 0.5
#    sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
#    return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)

def to_hyperboloid(Y, args, eps=1e-6,c=1.0):
    sqrt_c=c**0.5
    mink_pts = torch.zeros((Y.shape[0], Y.shape[1] + 1))
    if not args.cuda == -1:
        mink_pts = mink_pts.to(args.device)
    r = torch.norm(Y, dim=1)
    mink_pts[:,0] = (1/sqrt_c)*(1 + c*(r ** 2)) / (1 - c*(r ** 2) + eps)
    temp=2 / (1 - c*(r ** 2) + eps)
    for i in range(Y.shape[1]):
        mink_pts[:,i+1] = temp * Y[:,i]
    return mink_pts

#def to_poincare(x, c):
#    K = 1. / c
#    sqrtK = K ** 0.5
#    d = x.size(-1) - 1
#    return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)

def to_poincare(x, eps=1e-6,c=1.0):
    sqrt_c=c**0.5
    poincare_pt = torch.zeros((len(x)-1,))
    poincare_pt = x[1:]/(1+(sqrt_c*x[0]) + eps)
    if torch.norm(poincare_pt) >= 1:
        poincare_pt = poincare_pt/torch.norm(poincare_pt) - 0.1
    return poincare_pt

def h2k(x,c):
    sqrt_c=c**0.5
    tmp = x.narrow(-1, 1, x.size(-1) - 1) / x.narrow(-1, 0, 1)
    return (1/sqrt_c)*tmp

def k2h(x, args,c):
    sqrt_c=c**0.5
    x_norm_square = x.pow(2).sum(-1, keepdim=True)
    x_norm_square = torch.clamp(x_norm_square, max=0.9)
    tmp0 = torch.ones(1)
    tmp = torch.full_like(tmp0,1/sqrt_c)
    if not args.cuda == -1:
        tmp = tmp.to(args.device)
    tmp1 = torch.cat((tmp, x),dim=0)
    tmp2 = 1.0 / torch.sqrt(1.0 - c*x_norm_square)
    tmp3 = (tmp1 * tmp2)
    return tmp3

def lorenz_factor(x,c=1.0,keepdim=True):
    x_norm = x.pow(2).sum(-1, keepdim=keepdim)
    x_norm = torch.clamp(x_norm, 0, 0.9)
    tmp = 1 / torch.sqrt(1 - c * x_norm)
    return tmp

def compute_hyperbolic_mean(X, args,c):
    #print("x:", X.shape)
    X_klein=h2k(X,c)   #(node_num,dim)
    #print("k_x:", X_klein.shape)
    lamb = lorenz_factor(X_klein, c=1.0, keepdim=True)   #(node_num,1)
    k_mean = (torch.sum(lamb * X_klein, dim=0, keepdim=True) / (torch.sum(lamb, dim=0, keepdim=True))).squeeze()
    #print("k_mean:",k_mean.shape)
    h_mean = k2h(k_mean, args,c)
    #print("h_mean:",h_mean.shape)
    return h_mean

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

class HypAvgReadout(nn.Module):
    def __init__(self):
        super(HypAvgReadout, self).__init__()

    def forward(self, seq, msk, args, c):
        seq = seq.squeeze()
        if args.manifold == 'PoincareBall':
            seq = to_hyperboloid(seq, args, c)
        
        if msk is None:
            result = compute_hyperbolic_mean(seq, args, c)
        else:   ### not change
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

        if args.manifold == 'PoincareBall':
            result = to_poincare(result, c)
        return torch.unsqueeze(result, 0)