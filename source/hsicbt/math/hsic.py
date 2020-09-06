import torch
import numpy as np
from torch.autograd import Variable, grad
# pylint: disable=no-member
# pylint: disable=not-callable
def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X,Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med=np.mean(Tri)
    if med<1E-2:
        med=1E-2
    return med

def distmat(X, requires_grad=False):
    """  distance matrix  |X .X - 2(X x Xt) + (X.X)t|
    more memory efficient and 25% faster dist matrix
    in place ops save space, if no grad required, kill it
    allocates tensor shape (len(X[0]), len(X[0])) only once
    Args
        requires_grad    (bool[False]) if True passes gradient to out
    """
    _cloned = False
    if X.requires_grad and not requires_grad:
        X = X.clone().detach()
        _cloned = True
    out = torch.mm(X, X.T).mul_(-2.0)
    out.add_((X*X).sum(1, keepdim=True))
    out.add_((X*X).sum(1, keepdim=True).T)
    if _cloned:
        del X
    return out.abs_()

def kernelmat(X, sigma=None, requires_grad=False):
    """ kernel matrix baker
        Args
            X             (tensor_ shape (batchsize, datadimension)
            sigma         (float [None]) from config
            requires_grad (bool [False]) removes gradient from output
        minimized memory allocation, fixed device, removes grad if requested
    """
    m, dim = X.size()
    H = torch.eye(m, device=X.device).sub_(1/m)
    Kx = distmat(X, requires_grad=requires_grad)

    if sigma:
        variance = 2.*sigma*sigma*dim
        torch.exp_(Kx.mul_(-1.0/variance))
    else:
        try:
            sx = sigma_estimation(X, X)
            variance = 2.*sx*sx
            torch.exp_(Kx.mul_(-1.0/variance))
        except RuntimeError as e:
            raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                sx, torch.max(X), torch.min(X)))

    Kxc = torch.mm(Kx, H)
    del H
    del Kx
    return Kxc

def distcorr(X, sigma=1.0):
    X = distmat(X)
    X = torch.exp(-X / (2.*sigma*sigma))
    return torch.mean(X)

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def mmd(x, y, sigma=None, use_cuda=True, to_numpy=False):
    m = int(x.size()[0])
    H = torch.eye(m) - (1./m) * torch.ones([m,m])
    # H = Variable(H)
    Dxx = distmat(x)
    Dyy = distmat(y)

    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
        sxy = sigma
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    # Kxc = torch.mm(Kx,H)            # centered kernel matrices
    # Kyc = torch.mm(Ky,H)
    Dxy = distmat(torch.cat([x,y]))
    Dxy = Dxy[:x.size()[0], x.size()[0]:]
    Kxy = torch.exp( -Dxy / (1.*sxy*sxy))

    mmdval = torch.mean(Kx) + torch.mean(Ky) - 2*torch.mean(Kxy)

    return mmdval

def mmd_pxpy_pxy(x,y,sigma=None,use_cuda=True, to_numpy=False):
    """
    """
    if use_cuda:
        x = x.cuda()
        y = y.cuda()
    m = int(x.size()[0])

    Dxx = distmat(x)
    Dyy = distmat(y)
    if sigma:
        Kx  = torch.exp( -Dxx / (2.*sigma*sigma))   # kernel matrices
        Ky  = torch.exp( -Dyy / (2.*sigma*sigma))
    else:
        sx = sigma_estimation(x,x)
        sy = sigma_estimation(y,y)
        sxy = sigma_estimation(x,y)
        Kx = torch.exp( -Dxx / (2.*sx*sx))
        Ky = torch.exp( -Dyy / (2.*sy*sy))
    A = torch.mean(Kx*Ky)
    B = torch.mean(torch.mean(Kx,dim=0)*torch.mean(Ky, dim=0))
    C = torch.mean(Kx)*torch.mean(Ky)
    mmd_pxpy_pxy_val = A - 2*B + C 
    return mmd_pxpy_pxy_val

def hsic_regular(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """
    """
    Kxc = kernelmat(x, sigma)
    Kyc = kernelmat(y, sigma)
    KtK = torch.mul(Kxc, Kyc.t())
    Pxy = torch.mean(KtK)
    return Pxy

def hsic_normalized(x, y, sigma=None, use_cuda=True, to_numpy=True):
    """
    """
    m = int(x.size()[0])
    Pxy = hsic_regular(x, y, sigma, use_cuda)
    Px = torch.sqrt(hsic_regular(x, x, sigma, use_cuda))
    Py = torch.sqrt(hsic_regular(y, y, sigma, use_cuda))
    thehsic = Pxy/(Px*Py)
    return thehsic

def hsic_normalized_cca(x, y, sigma=None, requires_grad=False):
    """ reuse tensors, cleanup, maintains device, cleans grad
        x, y of shape (num_batches, -1)
    """
    epsilon = 1E-5
    m = x.size()[0]
    K_I = torch.eye(m, device=x.device).mul_(epsilon*m)

    Kc = kernelmat(x, sigma=sigma, requires_grad=requires_grad)
    Rx = Kc.mm(Kc.add(K_I).inverse())

    Kc = kernelmat(y, sigma=sigma, requires_grad=requires_grad)
    Ry = Kc.mm(Kc.add(K_I).inverse())

    out = Rx.mul_(Ry.t()).sum()

    del Rx
    del Ry
    del Kc
    del K_I
    return out
