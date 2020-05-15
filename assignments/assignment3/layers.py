import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    
    if predictions.ndim == 1:
        pred = predictions.copy() - np.max(predictions)
        probs = np.exp(pred) / np.sum(np.exp(pred))
    else:
        pred = predictions.copy() - np.max(predictions, axis=1)[:, np.newaxis]
        probs = np.exp(pred) / np.sum(np.exp(pred), axis=1)[:, np.newaxis]
    
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    
    prob = probs.copy()
    if probs.ndim == 1:
        loss = - np.log(prob[target_index])
    else:
        target_index = np.reshape(target_index, (target_index.shape[0], -1))
        loss = - np.mean(np.take_along_axis(np.log(prob), target_index, axis=1))
    return loss


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    
    loss = reg_strength * np.sum(W * W)
    grad = 2 * reg_strength * W
    
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    
    if predictions.ndim == 1:
        dprediction = probs - np.eye(predictions.shape[0])[target_index]
    else:
        target_index = np.reshape(target_index, (target_index.shape[0], -1))
        d_preds = (probs - np.eye(predictions.shape[1])[target_index[:, 0]]) / target_index.shape[0]

    return loss, d_preds


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.mask = (X > 0).astype(int)
        result = np.maximum(X, 0)
        
        return result

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = d_out * self.mask
        
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        result = np.dot(X, self.W.value) + self.B.value
        
        return result

    def backward(self, d_out):
        db = np.sum(d_out, axis=0)
        self.B.grad += db
        
        dw = np.dot(self.X.T, d_out)
        self.W.grad += dw
        
        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 1 + 2 * self.padding
        out_width = width - self.filter_size + 1 + 2 * self.padding
        
        self.X = X.copy()
        self.X = np.pad(self.X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),mode='constant')
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        result = np.zeros([batch_size, out_height, out_width, self.out_channels])
        converted_W = self.W.value.reshape((self.filter_size * self.filter_size * self.in_channels,
                                                    self.out_channels))
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                pixel_region = self.X[:, y:y + self.filter_size, x:x + self.filter_size, :]
                converted_pixel_region = pixel_region.reshape((batch_size,
                                                               self.filter_size * self.filter_size * self.in_channels))
                
                result[:, y, x, :] = converted_pixel_region.dot(converted_W) + self.B.value
            
        return result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        
        d_in = np.zeros(self.X.shape)
        converted_W = self.W.value.reshape((self.filter_size * self.filter_size * self.in_channels,
                                                    self.out_channels))
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                pixel_region = self.X[:, y:y + self.filter_size, x:x + self.filter_size, :]
                converted_pixel_region = pixel_region.reshape((batch_size,
                                                               self.filter_size * self.filter_size * self.in_channels))
                db = np.sum(d_out[:, y, x, :], axis=0)
                self.B.grad += db
                
                dw = (converted_pixel_region.T.dot(d_out[:, y, x, :])).reshape((self.filter_size,
                                                                            self.filter_size, self.in_channels, out_channels))
                self.W.grad += dw
                
                res = (d_out[:, y, x, :].dot(converted_W.T)).reshape((batch_size, 
                                                                    self.filter_size, self.filter_size, self.in_channels))
                d_in[:, y:y + self.filter_size, x:x + self.filter_size, :] += res
                
        return d_in[:, self.padding:height - self.padding, self.padding:width - self.padding,:]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        self.X = X.copy()
        result = np.zeros([batch_size, out_height, out_width, channels])
        
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        
        for y in range(out_height):
            for x in range(out_width):
                result[:, y, x, :] = np.max(self.X[:, y * self.stride: y * self.stride + self.pool_size,
                                                   x * self.stride: x * self.stride + self.pool_size, :], axis=(1, 2))
                
        return result
    
    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        d_in = np.zeros(self.X.shape)
        batch_inds, channels_inds = np.repeat(range(batch_size), channels), np.tile(range(channels), batch_size)
        
        for y in range(out_height):
            for x in range(out_width):
                pixel_region = self.X[:, y * self.stride: y * self.stride + self.pool_size, 
                                      x * self.stride: x * self.stride + self.pool_size, :]
                converted_pixel_region = pixel_region.reshape((batch_size, self.pool_size * self.pool_size, channels))
                
                pixel_region_in = np.zeros(converted_pixel_region.shape)
                maxpool_inds = np.argmax(converted_pixel_region, axis=1).flatten()
                
                pixel_region_in[batch_inds, maxpool_inds, channels_inds] = d_out[batch_inds, y, x, channels_inds]
                converted_pixel_region_in = pixel_region_in.reshape((batch_size, self.pool_size, self.pool_size, channels))
                
                d_in[:, y * self.stride: y * self.stride + self.pool_size, 
                                      x * self.stride: x * self.stride + self.pool_size, :] = converted_pixel_region_in
                
        return d_in
    
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        
        self.X_shape = X.shape
        return X.reshape((batch_size, height * width *channels))

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape((self.X_shape))

    def params(self):
        # No params!
        return {}