import numpy as np
from ..utils import append_bias_term

class LinearRegression(object):
    """
    Linear regression.
    """

    def __init__(self):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.
        """
        self.w = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: You can use the closed-form solution for linear regression
        (with or without regularization). Remember to handle the bias term.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        # Add bias term (column of ones)
        X = append_bias_term(training_data)  
       

        y = training_labels 

        # Compute weights using Normal Equation:
        # w = (X^T X)^(-1) X^T y
        # Using pseudo-inverse for numerical stability
        XTX = X.T @ X
        XTy = X.T @ y

        self.w = np.linalg.inv(XTX) @ XTy

        # Predict on training data
        pred_labels = X @ self.w

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        # Add bias term
        X = append_bias_term(test_data)  

        # Compute predictions
        pred_labels = X @ self.w

        return pred_labels
