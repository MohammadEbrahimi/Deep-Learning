pass
from DLCore.layers import *
from DLCore.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def max_pool_2d_in_3d_forward(x, pool_param):
#x(N,Fr,C,H,W)
#out(N,Fr,C,NH,NW)
    N,Fr,C,H,W=x.shape
    xr=x.reshape(N*Fr,C,H,W)
    a, mp_cache = max_pool_forward_fast(xr, pool_param)
    NFr,C,H,W=a.shape
    out=a.reshape(N,Fr,C,H,W)
    return out,mp_cache

def max_pool_2d_in_3d_backward(dout, cache):

    N,Fr,C,H,W=dout.shape
    dr=dout.reshape(N*Fr,C,H,W)
    dx= max_pool_backward_fast(dr, cache)
    NFr,C,H,W=dx.shape
    dx=dx.reshape(N,Fr,C,H,W)
    return dx

def conv2d_in_3d_forward(x,w,b, conv_param):
#x (N,Fr,C,H,W)
#w (Fi,C,HH,WW)
#b (Fi)
    N,Fr,C,H,W=x.shape
    xr=x.reshape(N*Fr,C,H,W)
    a, conv_cache = conv_forward_fast(xr, w, b, conv_param)
    #(x, w, b, conv_param, x_cols)=conv_cache
    NFr,C,H,W=a.shape
    out=a.reshape(N,Fr,C,H,W)
    return out,conv_cache

def conv2d_in_3d_backward(dout, cache):

    N,Fr,C,H,W=dout.shape
    dr=dout.reshape(N*Fr,C,H,W)
    dx,dw,db = conv_backward_fast(dr, cache)
    NFr,C,H,W=dx.shape
    dx=dx.reshape(N,Fr,C,H,W)
    return dx,dw,db

def conv2d_in_rot3d_forward(x,w,b, conv_param):
#x (N,Fr,C,H,W)
#w (Fi,C,HH,WW)
#b (Fi)
    N,Fr,C,H,W=x.shape
    xr=x.transpose(0,2,1,3,4)
    xr=xr.reshape(N*C,Fr,H,W)
    a, conv_cache = conv_forward_fast(xr, w, b, conv_param)
    NC,Fr,H,W=a.shape
    out=a.reshape(N,C,Fr,H,W)
    return out,conv_cache

def conv2d_in_rot3d_backward(dout, cache):

    N,C,Fr,H,W=dout.shape
    dr=dout.reshape(N*C,Fr,H,W)
    dx,dw,db = conv_backward_fast(dr, cache)
    NC,Fr,H,W=dx.shape
    dx=dx.reshape(N,C,Fr,H,W)
    dx=dx.transpose(0,2,1,3,4)
    return dx,dw,db

def conv1d_in_3d_forward(x,w,b):
#x (N,Fr,C,H,W)
#w (Ft,Fr)
#b (Ft)
#out (N,C,Ft,H,W)
    xr=x.transpose(0,2,1,3,4)
    out=np.sum(xr[:,:,None,:,:,:] * w[None,None,:,:,None,None],3)+b[None,None,:,None,None]
    cache=(xr,w,b)
    return out,cache

def conv1d_in_3d_backward(dout, cache):
    (xr,w,b)=cache
    dxr=np.sum(w[None,None,:,:,None,None] * dout[:,:,:,None,:,:],2)
    dx=dxr.transpose(0,2,1,3,4) 
    dw=np.sum(xr[:,:,None,:,:,:]*dout[:,:,:,None,:,:],axis=(0,1,4,5))
    db=np.sum(dout,axis=(0,1,3,4))
    return dx,dw,db

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
