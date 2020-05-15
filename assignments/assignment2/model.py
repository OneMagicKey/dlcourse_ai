import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        
        # Create layers
        self.first_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.activation = ReLULayer()
        self.second_layer = FullyConnectedLayer(hidden_layer_size, n_output)
        
        # Add params to the net
        self.first_layer_params = self.first_layer.params()
        self.second_layer_params = self.second_layer.params()

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        
        for key, value in self.params().items():
            value.grad.fill(0)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        #Forward pass
        first_layer_res = self.first_layer.forward(X)
        activation_layer_res = self.activation.forward(first_layer_res)
        second_layer_res = self.second_layer.forward(activation_layer_res)
        loss, d_preds = softmax_with_cross_entropy(second_layer_res, y)
        
        #Backward_pass
        second_layer_grad = self.second_layer.backward(d_preds)
        activation_layer_grad = self.activation.backward(second_layer_grad)
        first_layer_grad = self.first_layer.backward(activation_layer_grad)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        
        for key, param in self.params().items():
            l2_loss, l2_grad = l2_regularization(param.value, self.reg)
            loss += l2_loss
            param.grad += l2_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        first_layer_res = self.first_layer.forward(X)
        activation_layer_res = self.activation.forward(first_layer_res)
        second_layer_res = self.second_layer.forward(activation_layer_res)
        
        pred = np.argmax(second_layer_res, axis=-1)
        return pred

    def params(self):
        result = {}
    
        # TODO Implement aggregating all of the params

        d1 = {k+'1': v for k, v in self.first_layer_params.items()}
        d2 = {k+'2': v for k, v in self.second_layer_params.items()}
        result = {**d1, **d2}
        
        return result
