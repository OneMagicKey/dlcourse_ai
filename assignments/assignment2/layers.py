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
        probs = (np.exp(pred) + 1e-16) / (np.sum(np.exp(pred)) + 1e-16)
    else:
        pred = predictions.copy() - np.max(predictions, axis=1)[:, np.newaxis]
        probs = (np.exp(pred) + 1e-16) / (np.sum(np.exp(pred), axis=1)[:, np.newaxis] + 1e-16)
    
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
    
    loss = reg_strength * np.sum(W ** 2)
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
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        
        self.mask = (X > 0).astype(int)
        result = np.maximum(X, 0)
        
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        
        d_result = d_out * self.mask
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        
        self.X = X
        result = np.dot(X, self.W.value) + self.B.value
        
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute
        
        # It should be pretty similar to linear classifier from
        # the previous assignment
        
        db = np.sum(d_out, axis=0)
        self.B.grad += db
        
        dw = np.dot(self.X.T, d_out)
        self.W.grad += dw
        
        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
