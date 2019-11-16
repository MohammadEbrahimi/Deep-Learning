from builtins import object
import numpy as np

from DLCore.layers import *
from DLCore.fast_layers import *
from DLCore.layer_utils import *


class ThreeLayerTemporalConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(16,3, 120, 160), num_filters=32,num_Tfilters=8, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,dropout=0.1,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.dropout=1-dropout
        poolsize=2

        ############################################################################
                          
        ############################################################################
        self.params['W1']= weight_scale * np.random.randn(num_filters,input_dim[1],filter_size,filter_size)
        self.params['b1']= np.zeros(num_filters)
        self.params['W2']= weight_scale * np.random.randn(num_Tfilters,input_dim[0])
        self.params['b2']= np.zeros(num_Tfilters)
        self.params['W3']= weight_scale * np.random.randn(num_filters,num_Tfilters,filter_size,filter_size)
        self.params['b3']= np.zeros(num_filters)
        self.params['W4']= weight_scale * np.random.randn(num_filters,num_filters,filter_size,filter_size)
        self.params['b4']= np.zeros(num_filters)
        self.params['W5']= weight_scale * np.random.randn(int(num_filters*num_filters*input_dim[2]*input_dim[3]/(poolsize**8)),hidden_dim)
        self.params['b5']= np.zeros(hidden_dim)
        self.params['W6']= weight_scale * np.random.randn(hidden_dim,num_classes)
        self.params['b6']= np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        W6, b6 = self.params['W6'], self.params['b6']


        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        dropout_param={}
        dropout_param['p']= self.dropout
        if y is None: 
                dropout_param['mode'] = 'test'
        else:
	        dropout_param['mode'] = 'train'


        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        (H,C1)   = conv2d_in_3d_forward(X,W1,b1,conv_param)
        (H,Cr1) = relu_forward(H)
        (H,Cm1) = max_pool_2d_in_3d_forward(H,pool_param)
        (H,Cd1) = dropout_forward(H,dropout_param)
       
        (H,C2)   = conv1d_in_3d_forward(H,W2,b2)
        (H,Cr2) = relu_forward(H)
        (H,Cm2) = max_pool_2d_in_3d_forward(H,pool_param)
        (H,Cd2) = dropout_forward(H,dropout_param)
       
        (H,C3)   = conv2d_in_3d_forward(H,W3,b3,conv_param)
        (H,Cr3) = relu_forward(H)
        (H,Cm3) = max_pool_2d_in_3d_forward(H,pool_param)
       
        (H,C4)   = conv2d_in_rot3d_forward(H,W4,b4,conv_param)
        (H,Cr4) = relu_forward(H)
        (H,Cm4) = max_pool_2d_in_3d_forward(H,pool_param)
       
        (H,C5)   = affine_relu_forward(H,W5,b5)
       
        (scores,C6)   = affine_forward(H,W6,b6)
        #print('Forward Finished!')

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        (loss,dout) = softmax_loss(scores,y.astype(int))
        
        (dout,grads['W6'],grads['b6'])=affine_backward(dout, C6)
        
        (dout,grads['W5'],grads['b5'])=affine_relu_backward(dout, C5)
        
        (dout)=max_pool_2d_in_3d_backward(dout,Cm4)
        (dout)=relu_backward(dout,Cr4)
        (dout,grads['W4'],grads['b4'])=conv2d_in_rot3d_backward(dout,C4)
        
        (dout)=max_pool_2d_in_3d_backward(dout,Cm3)
        (dout)=relu_backward(dout,Cr3)
        (dout,grads['W3'],grads['b3'])=conv2d_in_3d_backward(dout,C3)
        
        (dout)=dropout_backward(dout,Cd2)
        (dout)=max_pool_2d_in_3d_backward(dout,Cm2)
        (dout)=relu_backward(dout,Cr2)
        (dout,grads['W2'],grads['b2'])=conv1d_in_3d_backward(dout,C2)
        
        (dout)=dropout_backward(dout,Cd1)
        (dout)=max_pool_2d_in_3d_backward(dout,Cm1)
        (dout)=relu_backward(dout,Cr1)
        (dout,grads['W1'],grads['b1'])=conv2d_in_3d_backward(dout,C1)
 
        loss += 0.5 * self.reg * np.sum(W1**2)        
        #loss += 0.5 * self.reg * np.sum(W2**2)        
        loss += 0.5 * self.reg * np.sum(W3**2)      
        loss += 0.5 * self.reg * np.sum(W4**2)        
        loss += 0.5 * self.reg * np.sum(W5**2)
        loss += 0.5 * self.reg * np.sum(W6**2)  
####Temporal Smoothness 
        loss += 0.5 * self.reg * np.sum((W2[:,1:]-W2[:,:-1])**2)   
        grads['W2'][:,0]  += (W2[:,0]-W2[:,1])  *self.reg
        grads['W2'][:,-1] += (W2[:,-1]-W2[:,-2]) *self.reg
        grads['W2'][:,1:-1] += (2*W2[:,1:-1]-W2[:,2:]-W2[:,:-2]) *self.reg

        grads['W1'] += W1 *self.reg
        #grads['W2'] += W2 *self.reg
        grads['W3'] += W3 *self.reg
        grads['W4'] += W4 *self.reg
        grads['W5'] += W5 *self.reg
        grads['W6'] += W6 *self.reg

        #print('backward Finished!')


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
