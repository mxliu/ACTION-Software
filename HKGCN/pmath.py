"""
Implementation of various mathematical operations in the Poincare ball model of hyperbolic space. Some
functions are based on the implementation in https://github.com/geoopt/geoopt (copyright by Maxim Kochurov).
"""

from turtle import width
import numpy as np
import torch
from scipy.special import gamma


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


# +
class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class RiemannianGradient(torch.autograd.Function):

    c = 1

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # x: B x d

        scale = (1 - RiemannianGradient.c * x.pow(2).sum(-1, keepdim=True)).pow(2) / 4
        return grad_output * scale


# -


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x + torch.sqrt_(1 + x.pow(2))).clamp_min_(1e-5).log_()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


def artanh(x):
    return Artanh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def arcosh(x, eps=1e-5):  # pragma: no cover
    x = x.clamp(-1 + eps, 1 - eps)
    return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))



def project(x, *, c=1.0): #PPF
    r"""
    Safe projection on the manifold for numerical stability. This was mentioned in [1]_
    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        projected vector on the manifold
    References
    ----------
    .. [1] Hyperbolic Neural Networks, NIPS2018
        https://arxiv.org/abs/1805.09112
    """
    c = torch.as_tensor(c).type_as(x)
    return _project(x, c)

def _project(x, c): 
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5) #求指定位置(最后一维)的2范数，并且设置最小值为1e-5
    maxnorm = (1 - 1e-3) / (c ** 0.5) #最大范数
    cond = norm > maxnorm #如果norm>maxnorm,则为ture，否则为false
    projected = x / norm * maxnorm     #project函数的实现
    return torch.where(cond, projected, x) #torch.where(condition，a，b)，输入参数condition为限制条件，如果为true，则a，否则输出b

#def _project(x, c):
#    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5) #求指定位置的范数，并且设置最小值为1e-5
#    maxnorm = (1 - 1e-3) / (c ** 0.5) #最大范数
#    cond = norm > maxnorm
#    projected = x / (norm + 1)* maxnorm #这里改动过
#    return torch.where(cond, projected, x)

def lambda_x(x, *, c=1.0, keepdim=False):
    r"""
    Compute the conformal factor :math:`\lambda^c_x` for a point on the ball
    .. math::
        \lambda^c_x = \frac{1}{1 - c \|x\|_2^2}
    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    Returns
    -------
    tensor
        conformal factor
    """
    c = torch.as_tensor(c).type_as(x)
    return _lambda_x(x, c, keepdim=keepdim)


def _lambda_x(x, c, keepdim: bool = False):
    return 2 / (1 - c * x.pow(2).sum(-1, keepdim=keepdim))


def mobius_add(x, y, *, c=1.0):
    r"""
    Mobius addition is a special operation in a hyperbolic space.
    .. math::
        x \oplus_c y = \frac{
            (1 + 2 c \langle x, y\rangle + c \|y\|^2_2) x + (1 - c \|x\|_2^2) y
            }{
            1 + 2 c \langle x, y\rangle + c^2 \|x\|^2_2 \|y\|^2_2
        }
    In general this operation is not commutative:
    .. math::
        x \oplus_c y \ne y \oplus_c x
    But in some cases this property holds:
    * zero vector case
    .. math::
        \mathbf{0} \oplus_c x = x \oplus_c \mathbf{0}
    * zero negative curvature case that is same as Euclidean addition
    .. math::
        x \oplus_0 y = y \oplus_0 x
    Another usefull property is so called left-cancellation law:
    .. math::
        (-x) \oplus_c (x \oplus_c y) = y
    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    y : tensor
        point on the Poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        the result of mobius addition
    """
    c = torch.as_tensor(c).type_as(x)
    return _mobius_add(x, y, c)


def _mobius_add(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / (denom + 1e-5)


def dist(x, y, *, c=1.0, keepdim=False):
    r"""
    Distance on the Poincare ball
    .. math::
        d_c(x, y) = \frac{2}{\sqrt{c}}\tanh^{-1}(\sqrt{c}\|(-x)\oplus_c y\|_2)
    .. plot:: plots/extended/poincare/distance.py
    Parameters
    ----------
    x : tensor
        point on poincare ball
    y : tensor
        point on poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`y`
    """
    c = torch.as_tensor(c).type_as(x)
    return _dist(x, y, c, keepdim=keepdim)


def _dist(x, y, c, keepdim: bool = False):
    sqrt_c = c ** 0.5
    dist_c = artanh(sqrt_c * _mobius_add(-x, y, c).norm(dim=-1, p=2, keepdim=keepdim))
    return dist_c * 2 / sqrt_c


def dist0(x, *, c=1.0, keepdim=False):
    r"""
    Distance on the Poincare ball to zero
    Parameters
    ----------
    x : tensor
        point on poincare ball
    c : float|tensor
        ball negative curvature
    keepdim : bool
        retain the last dim? (default: false)
    Returns
    -------
    tensor
        geodesic distance between :math:`x` and :math:`0`
    """
    c = torch.as_tensor(c).type_as(x)
    return _dist0(x, c, keepdim=keepdim)


def _dist0(x, c, keepdim: bool = False):
    sqrt_c = c ** 0.5
    dist_c = artanh(sqrt_c * x.norm(dim=-1, p=2, keepdim=keepdim))
    return dist_c * 2 / sqrt_c


def expmap(x, u, *, c=1.0):
    r"""
    Exponential map for Poincare ball model. This is tightly related with :func:`geodesic`.
    Intuitively Exponential map is a smooth constant travelling from starting point :math:`x` with speed :math:`u`.
    A bit more formally this is travelling along curve :math:`\gamma_{x, u}(t)` such that
    .. math::
        \gamma_{x, u}(0) = x\\
        \dot\gamma_{x, u}(0) = u\\
        \|\dot\gamma_{x, u}(t)\|_{\gamma_{x, u}(t)} = \|u\|_x
    The existence of this curve relies on uniqueness of differential equation solution, that is local.
    For the Poincare ball model the solution is well defined globally and we have.
    .. math::
        \operatorname{Exp}^c_x(u) = \gamma_{x, u}(1) = \\
        x\oplus_c \tanh(\sqrt{c}/2 \|u\|_x) \frac{u}{\sqrt{c}\|u\|_2}
    Parameters
    ----------
    x : tensor
        starting point on poincare ball
    u : tensor
        speed vector on poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        :math:`\gamma_{x, u}(1)` end point
    """
    c = torch.as_tensor(c).type_as(x)
    return _expmap(x, u, c)


def _expmap(x, u, c):  # pragma: no cover
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    second_term = (
        tanh(sqrt_c / 2 * _lambda_x(x, c, keepdim=True) * u_norm)
        * u
        / (sqrt_c * u_norm)
    )
    gamma_1 = _mobius_add(x, second_term, c)
    return gamma_1


def expmap0(u, *, c=1.0):
    r"""
    Exponential map for Poincare ball model from :math:`0`.
    .. math::
        \operatorname{Exp}^c_0(u) = \tanh(\sqrt{c}/2 \|u\|_2) \frac{u}{\sqrt{c}\|u\|_2}
    Parameters
    ----------
    u : tensor
        speed vector on poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    c = torch.as_tensor(c).type_as(u)
    return _expmap0(u, c)


def _expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1


def logmap(x, y, *, c=1.0):
    r"""
    Logarithmic map for two points :math:`x` and :math:`y` on the manifold.
    .. math::
        \operatorname{Log}^c_x(y) = \frac{2}{\sqrt{c}\lambda_x^c} \tanh^{-1}(
            \sqrt{c} \|(-x)\oplus_c y\|_2
        ) * \frac{(-x)\oplus_c y}{\|(-x)\oplus_c y\|_2}
    The result of Logarithmic map is a vector such that
    .. math::
        y = \operatorname{Exp}^c_x(\operatorname{Log}^c_x(y))
    Parameters
    ----------
    x : tensor
        starting point on poincare ball
    y : tensor
        target point on poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        tangent vector that transports :math:`x` to :math:`y`
    """
    c = torch.as_tensor(c).type_as(x)
    return _logmap(x, y, c)


def _logmap(x, y, c):  # pragma: no cover
    sub = _mobius_add(-x, y, c)
    sub_norm = sub.norm(dim=-1, p=2, keepdim=True)
    lam = _lambda_x(x, c, keepdim=True)
    sqrt_c = c ** 0.5
    return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm


def logmap0(y, *, c=1.0):
    r"""
    Logarithmic map for :math:`y` from :math:`0` on the manifold.
    .. math::
        \operatorname{Log}^c_0(y) = \tanh^{-1}(\sqrt{c}\|y\|_2) \frac{y}{\|y\|_2}
    The result is such that
    .. math::
        y = \operatorname{Exp}^c_0(\operatorname{Log}^c_0(y))
    Parameters
    ----------
    y : tensor
        target point on poincare ball
    c : float|tensor
        ball negative curvature
    Returns
    -------
    tensor
        tangent vector that transports :math:`0` to :math:`y`
    """
    c = torch.as_tensor(c).type_as(y)
    return _logmap0(y, c)


def _logmap0(y, c):
    sqrt_c = c ** 0.5
    y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), 1e-5)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def mobius_matvec(m, x, *, c=1.0):
    r"""
    Generalization for matrix-vector multiplication to hyperbolic space defined as
    .. math::
        M \otimes_c x = (1/\sqrt{c}) \tanh\left(
            \frac{\|Mx\|_2}{\|x\|_2}\tanh^{-1}(\sqrt{c}\|x\|_2)
        \right)\frac{Mx}{\|Mx\|_2}
    Parameters
    ----------
    m : tensor
        matrix for multiplication
    x : tensor
        point on poincare ball
    c : float|tensor
        negative ball curvature
    Returns
    -------
    tensor
        Mobius matvec result
    """
    c = torch.as_tensor(c).type_as(x)
    return _mobius_matvec(m, x, c)


def _mobius_matvec(m, x, c):
    x_norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    sqrt_c = c ** 0.5
    mx = x @ m.transpose(-1, -2)
    mx_norm = mx.norm(dim=-1, keepdim=True, p=2)
    res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
    cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return _project(res, c)


def _tensor_dot(x, y):
    res = torch.einsum("ij,kj->ik", (x, y))
    return res


def _mobius_addition_batch(x, y, c):
    xy = _tensor_dot(x, y)  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = 1 + 2 * c * xy + c * y2.permute(1, 0)  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c ** 2 * x2 * y2.permute(1, 0)
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res


def _hyperbolic_softmax(X, A, P, c):
    lambda_pkc = 2 / (1 - c * P.pow(2).sum(dim=1))
    k = lambda_pkc * torch.norm(A, dim=1) / torch.sqrt(c)
    mob_add = _mobius_addition_batch(-P, X, c)
    num = 2 * torch.sqrt(c) * torch.sum(mob_add * A.unsqueeze(1), dim=-1)
    denom = torch.norm(A, dim=1, keepdim=True) * (1 - c * mob_add.pow(2).sum(dim=2))
    logit = k.unsqueeze(1) * arsinh(num / denom)
    return logit.permute(1, 0)


def p2k(x, c):
    denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
    return 2 * x / denom


def k2p(x, c):
    denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
    return x / denom


def lorenz_factor(x, *, c=1.0, dim=-1, keepdim=False):
    """

    Parameters
    ----------
    x : tensor
        point on Klein disk
    c : float
        negative curvature
    dim : int
        dimension to calculate Lorenz factor
    keepdim : bool
        retain the last dim? (default: false)

    Returns
    -------
    tensor
        Lorenz factor
    """
    return 1 / torch.sqrt(1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))


def poincare_mean(x, dim=0, c=1.0):
    x = p2k(x, c)
    lamb = lorenz_factor(x, c=c, keepdim=True)
    mean = torch.sum(lamb * x, dim=dim, keepdim=True) / torch.sum(
        lamb, dim=dim, keepdim=True
    )
    mean = k2p(mean, c)
    return mean.squeeze(dim)


def _dist_matrix(x, y, c):
    sqrt_c = c ** 0.5
    return (
        2
        / sqrt_c
        * artanh(sqrt_c * torch.norm(_mobius_addition_batch(-x, y, c=c), dim=-1))
    )


def dist_matrix(x, y, c=1.0):
   # print("hyperbolic distance")
    c = torch.as_tensor(c).type_as(x)
    return _dist_matrix(x, y, c)

def _kernel_dist_matrix_square(u, v, c=1.0):
    uv = _tensor_dot(u,v)
  
    v_per = v.T
    u_norm_square = torch.norm(u, p=2, dim=1, keepdim=True)**2
    v_norm_square = torch.norm(v_per, p=2, dim=0, keepdim=True)**2
    u_norm_square_matrix = u_norm_square.repeat(1,v.shape[0])
    v_norm_square_matrix = v_norm_square.repeat(u.shape[0],1)
    K_distance_square = 1/(1-c*u_norm_square_matrix) + 1/(1-c*v_norm_square_matrix) -(2-c*uv)/(1-2*uv+(c**2)*u_norm_square_matrix*v_norm_square_matrix)
    return(
        K_distance_square
    )



def kernel_dist_matrix_square(x, y, c=1.0):
   # print("hyperbolic distance")
    c = torch.as_tensor(c).type_as(x)
    return _kernel_dist_matrix_square(x, y, c)





def DAkernelmatrix(x, y, c=1.0):
    #print('DAkernel')
    xy = _tensor_dot(x, y)
    #print(xy.equal(xy2))
    c = torch.as_tensor(c).type_as(xy)
    #square_c = c**2
    return   1/(1-c*xy)

def EUClinearmatrix(x, y):
   # print('EUCkernel')
    xy = _tensor_dot(x, y)
    #print(xy.equal(xy2))
    #c = torch.as_tensor(c).type_as(xy)
    #square_c = c**2
    return  xy
    



def EUCPolykernelmatrix(x, y, width=1, degree=2.0):
   # print('EUCPoly')
   # c = torch.as_tensor(c).type_as(x)
    xy = _tensor_dot(x, y)
    #c = torch.as_tensor(c).type_as(xy)
    #square_c = c**2
   # dakernel=1/(1-c*xy)
    return   (xy +width)**degree

def EUCSigmoidkernelmatrix(x, y, width=1.0):
  #  print('EUCSigmoid')
    xy = _tensor_dot(x, y)
    return   (
        torch.tanh(width*xy-1)
    )

def EUCGaussmatrix(x, y,  width=1.0):
  #  print('EUCGauss')
    #square_c = c**2
    square_width = width**2
    y_per=y.T
    x_norm_square = torch.norm(x, p=2, dim=1, keepdim=True)**2
    y_norm_square = torch.norm(y_per, p=2, dim=0, keepdim=True)**2
    x_norm_square_matrix = x_norm_square.repeat(1,y.shape[0])
    y_norm_square_matrix = y_norm_square.repeat(x.shape[0],1)
    xy_distance = x_norm_square_matrix+y_norm_square_matrix-2*_tensor_dot(x, y)

    return (
        #(-1/square_width)*(1/(1-_tensor_dot(x, x)/square_c)+1/(1-_tensor_dot(y, y)/square_c)-2/(1-_tensor_dot(x, y)/square_c))
        -0.5/square_width*xy_distance
        )




def EUCLaplacematrix(x, y,  width=1.0):
 #   print('EUCLaplace')
    #square_c = c**2
    #square_width = width**2
    y_per=y.T
    x_norm_square = torch.norm(x, p=2, dim=1, keepdim=True)**2
    y_norm_square = torch.norm(y_per, p=2, dim=0, keepdim=True)**2
    x_norm_square_matrix = x_norm_square.repeat(1,y.shape[0])
    y_norm_square_matrix = y_norm_square.repeat(x.shape[0],1)
    xy_distance = x_norm_square_matrix + y_norm_square_matrix-2*_tensor_dot(x, y)

    return (
        #(-1/square_width)*(1/(1-_tensor_dot(x, x)/square_c)+1/(1-_tensor_dot(y, y)/square_c)-2/(1-_tensor_dot(x, y)/square_c))
        -0.5/width*torch.sqrt(xy_distance)
        )

def NormRadialkernelRealNumSquare(u, v, c = 1.0):  #其实是带c的DA核的norm的square形式
    uv = _tensor_dot(u,v)
  
    #input()
    v_per = v.T
    u_norm_square = torch.norm(u, p=2, dim=1, keepdim=True)**2
    v_norm_square = torch.norm(v_per, p=2, dim=0, keepdim=True)**2
    u_norm_square_matrix = u_norm_square.repeat(1,v.shape[0])
    v_norm_square_matrix = v_norm_square.repeat(u.shape[0],1)
    A = 1 - 2*c*uv+c*v_norm_square_matrix
    B = 1 - c*u_norm_square_matrix
    D = 1-2*c*uv+(c**2)*u_norm_square_matrix*v_norm_square_matrix
    K_norm_square = 1-c*(A*A*u_norm_square_matrix-2*A*B*uv+B*B*v_norm_square_matrix)/(D*D)
    return(
        K_norm_square
    )

def NormRadialkernelReal(u, v, c = 1.0):
    uv = _tensor_dot(u,v)
  
    #input()
    v_per = v.T
    u_norm_square = torch.norm(u, p=2, dim=1, keepdim=True)**2
    v_norm_square = torch.norm(v_per, p=2, dim=0, keepdim=True)**2
    u_norm_square_matrix = u_norm_square.repeat(1,v.shape[0])
    v_norm_square_matrix = v_norm_square.repeat(u.shape[0],1)
    A = 1 - 2*c*uv+c*v_norm_square_matrix
    B = 1 - c*u_norm_square_matrix
    D = 1-2*c*uv+(c**2)*u_norm_square_matrix*v_norm_square_matrix
    K_norm_square = 1-c*(A*A*u_norm_square_matrix-2*A*B*uv+B*B*v_norm_square_matrix)/(D*D)
    return(
        torch.sqrt(K_norm_square)
    )






def RadialkernelRealNum(x, y, a, c=1.0):
    #print('RadialkernelRealNum')
    K_square = NormRadialkernelRealNumSquare(x, y, c) # [40, 64]
    #print('K_norm',K_norm)
    
    #print('K_square',K_square)
   # print('a:',a)
    a_size=a.size()
    # print('\n')
    # print('a_size:',a_size)
    K_tmp_expand = K_square.unsqueeze(-1)
    K_tmp_expand = K_tmp_expand.repeat(1,1,a_size[0])

    K_tmp_expand = K_tmp_expand.cumprod(dim=2)
    K_tmp = torch.zeros_like(K_square, dtype=torch.float)
    for i in range (0,a_size[0],1):
        K_tmp = K_tmp + a[i]* K_tmp_expand[:,:,i] # 这个地方其实也能改矩阵计算 但是感觉10个的话其实很快没必要
    return K_tmp
        



def normDAkernelmatrix(x, y, c=1.0):
    c = torch.as_tensor(c).type_as(x)
   # print('normDAkernel')
   # print('c:',c)
   # input()
    xy = _tensor_dot(x, y)
    #print('xy_size:',xy.size())
    y_per=y.T
    #print(xy.equal(xy2))
    #c = torch.as_tensor(c).type_as(xy)
    #square_c = c**2
    xydakernel = 1/(1-c*xy)
   # print('xydakernel',xydakernel)
    #input()

    x_norm_square = torch.norm(x, p=2, dim=1, keepdim=True)**2
    y_norm_square = torch.norm(y_per, p=2, dim=0, keepdim=True)**2
    x_norm_square_matrix = x_norm_square.repeat(1,y.shape[0])
    y_norm_square_matrix = y_norm_square.repeat(x.shape[0],1)
    kx_norm_reciprocal = torch.sqrt(1-c*x_norm_square_matrix)
    ky_norm_reciprocal = torch.sqrt(1-c*y_norm_square_matrix)
    k_norm=xydakernel*kx_norm_reciprocal*ky_norm_reciprocal
    #print('k_norm:', k_norm)
    #input()

    return   (
        #xydakernel*kx_norm_reciprocal*ky_norm_reciprocal
        k_norm
    )
   


def DAPolykernelmatrix(x, y, c=1.0, width=1, degree=2.0):
   # print('DAPoly')
    c = torch.as_tensor(c).type_as(x)
    xy = _tensor_dot(x, y)
    c = torch.as_tensor(c).type_as(xy)
    #square_c = c**2
    dakernel=1/(1-c*xy)
    return   (dakernel+width)**degree


def DASigmoidkernelmatrix(x, y, c=1.0, width=1.0):
    #print('DASigmoid')
    c = torch.as_tensor(c).type_as(x)
    xy = _tensor_dot(x, y)
    #square_c = c**2
    dakernel=1/(1-c*xy)
    return   (
        torch.tanh(width*dakernel-1)
    )


def Radialkernelmatrix(x, y, a, c=1.0):
    K_norm = normDAkernelmatrix(x, y, c) # [40, 64]
    #print('K_norm',K_norm)
    K_square = K_norm ** 2
    #print('K_square',K_square)
   # print('a:',a)
    a_size=a.size()
    # print('\n')
    # print('a_size:',a_size)
    K_tmp_expand = K_square.unsqueeze(-1)
    K_tmp_expand = K_tmp_expand.repeat(1,1,a_size[0])

    K_tmp_expand = K_tmp_expand.cumprod(dim=2)
    K_tmp = torch.zeros_like(K_square, dtype=torch.float)
    for i in range (0,a_size[0],1):
        K_tmp = K_tmp + a[i]* K_tmp_expand[:,:,i] # 这个地方其实也能改矩阵计算 但是感觉10个的话其实很快没必要
    return K_tmp
        


def DAGaussmatrix(x, y, c=1.0, width=1.0):
    #print('CDAGauss')
    #square_c = c**2
    c = torch.as_tensor(c).type_as(x)
    square_width = width**2
    y_per=y.T
    x_norm_square = torch.norm(x, p=2, dim=1, keepdim=True)**2
    y_norm_square = torch.norm(y_per, p=2, dim=0, keepdim=True)**2
    x_norm_square_matrix = x_norm_square.repeat(1,y.shape[0])
    y_norm_square_matrix = y_norm_square.repeat(x.shape[0],1)
    distance_CDA = 1/(1-c*x_norm_square_matrix)+1/(1-c*y_norm_square_matrix)-2/(1-_tensor_dot(x, y)*c)

    return (
        #(-1/square_width)*(1/(1-_tensor_dot(x, x)/square_c)+1/(1-_tensor_dot(y, y)/square_c)-2/(1-_tensor_dot(x, y)/square_c))
        -0.5/square_width*distance_CDA
        )
#torch.nn.paramter


def DALaplacematrix(x, y, c=1.0, width=1.0):
    #print('CDALaplace')
    #square_c = c**2
    #square_width = width**2
    y_per=y.T
    x_norm_square = torch.norm(x, p=2, dim=1, keepdim=True)**2
    y_norm_square = torch.norm(y_per, p=2, dim=0, keepdim=True)**2
    x_norm_square_matrix = x_norm_square.repeat(1,y.shape[0])
    y_norm_square_matrix = y_norm_square.repeat(x.shape[0],1)
    distance_CDA = 1/(1-c*x_norm_square_matrix)+1/(1-c*y_norm_square_matrix)-2/(1-_tensor_dot(x, y)*c)

    return (
        #(-1/square_width)*(1/(1-_tensor_dot(x, x)/square_c)+1/(1-_tensor_dot(y, y)/square_c)-2/(1-_tensor_dot(x, y)/square_c))
        -0.5/width*torch.sqrt(distance_CDA)
        )


def PseudoGaussmatrix(x, y, c=1.0, width=1.0):
    #print('PseudoGauss')
    #square_c = c**2
    square_width = width**2
    y_per=y.T
    x_norm_square = torch.norm(x, p=2, dim=1, keepdim=True)**2
    y_norm_square = torch.norm(y_per, p=2, dim=0, keepdim=True)**2
    x_norm_square_matrix = x_norm_square.repeat(1,y.shape[0])
    y_norm_square_matrix = y_norm_square.repeat(x.shape[0],1)
    xy = torch.mm(x, y_per)
    square_norm_matrix=x_norm_square_matrix + y_norm_square_matrix -2*xy
    
    dakernel=1/(1-c*xy)
    #pgk = (-1/square_width)*square_norm_matrix*(dakernel**2)
    #print(pgk)
    #print(torch.norm(pgk, p=2, dim=1, keepdim=True))
    return (
        -(1/square_width)*square_norm_matrix*(torch.mul(dakernel, dakernel))
    )


#def PseudoLaplacematrixReal(u, v, c=1.0, width=1.0):
def PseudoLaplacematrixReal(u, v, c=1.0):
    #print('PseudoLaplaceReal')
    uv = _tensor_dot(u,v)
  
    #input()
    v_per = v.T
    u_norm_square = torch.norm(u, p=2, dim=1, keepdim=True)**2
    v_norm_square = torch.norm(v_per, p=2, dim=0, keepdim=True)**2
    u_norm_square_matrix = u_norm_square.repeat(1,v.shape[0])
    v_norm_square_matrix = v_norm_square.repeat(u.shape[0],1)
    A = 1 - 2*c*uv+c*v_norm_square_matrix
    B = 1 - c*u_norm_square_matrix
    D = 1-2*c*uv+(c**2)*u_norm_square_matrix*v_norm_square_matrix
    pseudodistance = torch.sqrt(A*A*u_norm_square_matrix-2*A*B*uv+B*B*v_norm_square_matrix)/D
    return(
        pseudodistance 
    )



#def PseudoGaussmatrixReal(u, v, c=1.0, width=1.0):
def Pseudodistance_square(u, v, c=1.0):
    #print('PseudGaussReal')
    uv = _tensor_dot(u,v)
  
    #input()
    v_per = v.T
    u_norm_square = torch.norm(u, p=2, dim=1, keepdim=True)**2
    v_norm_square = torch.norm(v_per, p=2, dim=0, keepdim=True)**2
    u_norm_square_matrix = u_norm_square.repeat(1,v.shape[0])
    v_norm_square_matrix = v_norm_square.repeat(u.shape[0],1)
    A = 1 - 2*c*uv+c*v_norm_square_matrix
    B = 1 - c*u_norm_square_matrix
    D = 1-2*c*uv+(c**2)*u_norm_square_matrix*v_norm_square_matrix
    pseudodistance_square = (A*A*u_norm_square_matrix-2*A*B*uv+B*B*v_norm_square_matrix)/(D*D)

    return(
        pseudodistance_square 
    )



def PseudoLaplacematrix(x, y, c=1.0, width=1.0):
    #print('PseudoLaplace')
    square_c = c**2
    y_per=y.T
    x_norm_square = torch.norm(x, p=2, dim=1, keepdim=True)**2
    y_norm_square = torch.norm(y_per, p=2, dim=0, keepdim=True)**2
    x_norm_square_matrix = x_norm_square.repeat(1,y.shape[0])
    y_norm_square_matrix = y_norm_square.repeat(x.shape[0],1)
    xy = torch.mm(x, y_per)

    square_norm_matrix=x_norm_square_matrix + y_norm_square_matrix -2*xy
    dakernel=1/(1-c*xy)

    return (
        -(1/width)*torch.sqrt(square_norm_matrix)*dakernel
    )






def hyperbolicTangentmatrix(x, y, c):
   # print('hyperbolicTangent')
    #xy = _tensor_dot(x, y)
    xy = torch.mm(x,y)
    c = torch.as_tensor(c).type_as(xy)
    #square_c = c**2
    y_per=y.T
    x_norm_square = torch.norm(x, p=2, dim=1, keepdim=True)**2
    y_norm_square = torch.norm(y_per, p=2, dim=0, keepdim=True)**2
    x_norm_square_matrix = x_norm_square.repeat(1,y.shape[0])
    y_norm_square_matrix = y_norm_square.repeat(x.shape[0],1)
    tangent_x=torch.atan(torch.sqrt(c)*x_norm_square_matrix) / x_norm_square_matrix 
    tangent_y=torch.atan(torch.sqrt(c)*y_norm_square_matrix) / y_norm_square_matrix 

    return (
        tangent_x * tangent_y * xy
    )

def hyperbolicTangentmatrix(x, y, c):
   # print('hyperbolicTangent')
    xy = _tensor_dot(x, y)
    c = torch.as_tensor(c).type_as(xy)
    #square_c = c**2
    y_per=y.T
    x_norm_square = torch.norm(x, p=2, dim=1, keepdim=True)**2
    y_norm_square = torch.norm(y_per, p=2, dim=0, keepdim=True)**2
    x_norm_square_matrix = x_norm_square.repeat(1,y.shape[0])
    y_norm_square_matrix = y_norm_square.repeat(x.shape[0],1)
    tangent_x=torch.atan(torch.sqrt(c)*x_norm_square_matrix) / x_norm_square_matrix 
    tangent_y=torch.atan(torch.sqrt(c)*y_norm_square_matrix) / y_norm_square_matrix 

    return (
        tangent_x * tangent_y * xy
    )

def _hyperbolicRBFmatrix(x, y, c):
   # print('hyperbolicTangent')
    xy = _tensor_dot(x, y)
    c = torch.as_tensor(c).type_as(xy)
    #square_c = c**2
    y_per=y.T
    x_norm_square = torch.norm(x, p=2, dim=1, keepdim=True)**2
    y_norm_square = torch.norm(y_per, p=2, dim=0, keepdim=True)**2
    x_norm_square_matrix = x_norm_square.repeat(1,y.shape[0])
    y_norm_square_matrix = y_norm_square.repeat(x.shape[0],1)
    tangent_x=torch.atan(torch.sqrt(c)*x_norm_square_matrix) / x_norm_square_matrix 
    tangent_y=torch.atan(torch.sqrt(c)*y_norm_square_matrix) / y_norm_square_matrix 

    return (
        tangent_x * tangent_y * xy
    )



def _kernel_matrix(x, y, c=1.0, d=2.0, width=1.0, degree=2.0, kerneltype='DA'):
    if kerneltype=='DA':
        return DAkernelmatrix(x, y, c)
    if kerneltype=='normDA':
        return normDAkernelmatrix(x, y, c)
    elif kerneltype=='DAGauss': 
        return DAGaussmatrix(x, y, c, width)
    elif kerneltype=='DALaplace': 
        return DALaplacematrix(x, y, c, width)
    elif kerneltype=='DAPoly': 
        return DAPolykernelmatrix(x, y, c, width, degree)
    elif kerneltype=='DASigmoid': 
        return DASigmoidkernelmatrix(x, y, c, d)
    elif kerneltype=='pseudoGauss': 
        return PseudoGaussmatrix(x, y, c, width)
    elif kerneltype=='pseudoLaplace': 
        return PseudoLaplacematrix(x, y, c, width)
    elif kerneltype=='hyperbolicTangent': 
        return hyperbolicTangentmatrix(x, y, c)

def kernel_matrix(x, y, c=1.0, d=2.0, width=1.0, degree=2.0, kerneltype='DA'):
    c = torch.as_tensor(c).type_as(x)
    #print(kerneltype)
    #d = torch.as_tensor(d).type_as(x)
    #delta = torch.as_tensor(delta).type_as(x)
    #print(x.shape, y.shape)
    return _kernel_matrix(x, y, c, d, width, degree, kerneltype)



def auto_select_c(d):
    """
    calculates the radius of the Poincare ball,
    such that the d-dimensional ball has constant volume equal to pi
    """
    dim2 = d / 2.0
    R = gamma(dim2 + 1) / (np.pi ** (dim2 - 1))
    R = R ** (1 / float(d))
    c = 1 / (R ** 2)
    return c
