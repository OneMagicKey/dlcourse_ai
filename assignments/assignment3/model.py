import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        width, height, n_channels = input_shape
        
        self.conv1 = ConvolutionalLayer(n_channels, conv1_channels, 3, 1)
        self.relu1 = ReLULayer()
        self.maxpool1 = MaxPoolingLayer(4, 4)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.relu2 = ReLULayer()
        self.maxpool2 = MaxPoolingLayer(4, 4)
        self.flatten = Flattener()
        
        self.fc = FullyConnectedLayer((height // 4 // 4) * (width // 4 // 4) * conv2_channels, n_output_classes)
        
        self.conv1_params = self.conv1.params()
        self.conv2_params = self.conv2.params()
        self.fc_params = self.fc.params()

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for key, value in self.params().items():
            value.grad.fill(0)
        
        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        
        conv1 = self.conv1.forward(X)
        relu1 = self.relu1.forward(conv1)
        maxpool1 = self.maxpool1.forward(relu1)
        conv2 = self.conv2.forward(maxpool1)
        relu2 = self.relu2.forward(conv2)
        maxpool2 = self.maxpool2.forward(relu2)
        flatten = self.flatten.forward(maxpool2)
        fc = self.fc.forward(flatten)
        
        loss, d_preds = softmax_with_cross_entropy(fc, y)
        
        fc = self.fc.backward(d_preds)
        flatten = self.flatten.backward(fc)
        maxpool2 = self.maxpool2.backward(flatten)
        relu2 = self.relu2.backward(maxpool2)
        conv2 = self.conv2.backward(relu2)
        maxpool1 = self.maxpool1.backward(conv2)
        relu1 = self.relu1.backward(maxpool1)
        conv1 = self.conv1.backward(relu1)
        
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        
        conv1 = self.conv1.forward(X)
        relu1 = self.relu1.forward(conv1)
        maxpool1 = self.maxpool1.forward(relu1)
        conv2 = self.conv2.forward(maxpool1)
        relu2 = self.relu2.forward(conv2)
        maxpool2 = self.maxpool2.forward(relu2)
        flatten = self.flatten.forward(maxpool2)
        fc = self.fc.forward(flatten)
        
        return np.argmax(fc, axis = 1)

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        d1 = {k+'1': v for k, v in self.conv1_params.items()}
        d2 = {k+'2': v for k, v in self.conv2_params.items()}
        d3 = {k+'3': v for k, v in self.fc_params.items()}
        result = {**d1, **d2, **d3}
            
        return result
