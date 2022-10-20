"""Logistic_Regression

This is a simple Logistic Regression model for a target of two classes.
It uses a basic Gradient descent algorithm to calculate the weights.
The class has metric methods to evaluate the fit of the model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Logistic_regression:
    """The class that creates the logistic regression model.
    max_iter : the number of maximum iterations of the gradient decent algorithm

    Attributes
    ******************************
    max_iter : int default = 1000
        The number of maximum iterations the gradient decent algorithm does.
    a : float
        the learning rate of gradient decent.
    error_diff:  float default = 1e-7
        The threshold for the cross entropy difference. When the difference of cross entropy between iterations of the
        gradient descend is smaller than the threshold, the algorithm stops.
    show: boolean
        printing some info about the steps of the processes if True
    regularization : int, values : (1 or 2)
        If 1 Lasso regularization  will be used
        If 2 Ridge regularization  will be used
    l : float
        The regularization factor
    trained: bool
        True if the instance has been trained(fit) on data
    w : default (None) else mp array
        When the instance is initiolized the value is None
        if the instance is trained, w will be  D X 1  array of the weights of the model.
    """

    def __init__(self, max_iter=1000, a=0.01, error_diff=1e-7, show=False, regularization=2, l=0.1):
        self.D = None
        self.trained = False
        self.max_iter = max_iter
        self.a = a
        self.N = None
        self.error_diff = error_diff
        self.show = show
        self.w = None
        self.regularization = regularization
        self.l = l

    @staticmethod
    def regularization_cost(regularization, l, w):
        """Calculated the Regularization error and returns it"""
        if regularization == 2:
            return l*w
        elif regularization == 1:
            return l * np.sign(w)
        else:
            return 0

    @staticmethod
    def sigmoid(z):
        """the transformation of the linear value  z on the sigmoid line

        Parameters
        ******************************
        z : N x 1 dimension array
            A vector, that is calculated from the linear combination of the weights and feature values
            z = W.T.dot(X)

        Returns:
            N x  1 array of the predicted value probabilities
        """
        return 1 / (1 + np.exp(-z))


    def cross_entropy(self, T, y_pred):
        """The cross entropy function in matrix form.

        Parameters
        ******************************
        Τ : one dimensional np.array
            An array with the target values .
        y_pred : one dimensional np.array
            An array with the predicted values our model gives.


        Returns:
            The cross entropy score for our model
        """
        return -T.dot(np.log(y_pred)) - (1 - T).dot(np.log(1 - y_pred))


    def fit(self, X, Y):
        """The function fit takes the dependent variables matrix and the target matrix as inputs.

        Parameters
        ******************************
         X : N x D dimensions np.array
            The training feature values.
        Y : N x 1 dimensions np.array:
            The training test/target values of our dataset

        Returns: None
            Saves in the regression instance the weights of the model.
        """

        self.N, self.D = X.shape[0], X.shape[1]

        # creating the x0 = 1 column, for the wo weight (intercept)
        ones = np.array([[1] * self.N]).T
        X = np.concatenate((ones, X), axis=1)

        # initialize weights
        self.w = 0.01 * np.random.randn(self.D + 1)

        # starting the gradient descent
        iterations = 1
        y_prob = self.sigmoid(X.dot(self.w))
        e = 1
        e_prev = self.cross_entropy(Y, y_prob)

        # print some starting info before the gradient decent
        if self.show:
            print("Starting fitting process!")
            print(f"Cross entropy is {e_prev}.\n")

        while iterations < self.max_iter and e > self.error_diff:

            reg_error = self.regularization_cost(self.regularization, self.l, self.w)

            # calculating the changes in w
            w_change = self.a * (X.T.dot(Y - y_prob) + reg_error)
            self.w += w_change

            # new predictions
            y_prob = self.sigmoid(X.dot(self.w))
            e_new = self.cross_entropy(Y, y_prob)

            # if show is True printing some info every 20 iterations
            if iterations % 20 == 0 and self.show:
                print(f"Cross entropy is {e_new}.")
                print(f"Iteration : {iterations}.\n")

            e = e_prev - e_new
            e_prev = e_new

            iterations += 1

        if self.show and e < self.error_diff:
            print("Cross entropy difference reached threshold.")
        elif iterations < self.max_iter:
            print("Gradient decent reached max iterations.")

        self.trained = True


    def get_coef(self):
        """Returns: D X 1 np array
            the weights of the model"""
        if self.trained:
            return self.w
        else:
            print("Model not trained.")

    def y_probab(self, X):
        """Calculates adn returns the predicted probability Y for each feature vector in X

        Parameters
        *******************************
        X : N X D dimensions array
            The test feature data

        Returns : N x 1 arraey
            The predicted probability values for our input features X.
        """
        return self.sigmoid(X.dot(self.w))

    def predict(self, X, p=0.5):
        if self.trained:
            ones = np.array([[1] * X.shape[0]]).T
            X = np.concatenate((ones, X), axis=1)
            return [0 if i < p else 1 for i in self.y_probab(X)]
        else:
            print("Model not trained.")

    def confusion_matrix(self, T, Y_prob, values =False):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(T)):
            if T[i] == 0:
                if Y_prob[i] == 0:
                    TP += 1
                else:
                    FN += 1
            else:
                if Y_prob[i] == 1:
                    TN += 1
                else:
                    FP += 1
        if values:
            return TP, FP, FN, TN
        else:
            return np.array([[TP, FN],[FP, TN]])

    def metrics(self, T, Y):
        """Calculates recall precision and F1 metric for each class and the accuracy of the model."""
        TP, FP, FN, TN = self.confusion_matrix(T, Y, values = True)


        d = {"class_1": {"recall": TP / (TP + FN), "precision": TP/(TP + FP), "F1" : TP/(TP + 0.5*(FP + FN))},
             "class_2": {"recall": TN / (TN + FP), "precision": TN/(TN + FN), "F1" : TN/(TN + 0.5*(FN + FP))},
             "accuracy": (TN + TP)/(TN + TP + FN + FP)}
        return d

    # noinspection PyTypeChecker
    def ROC(self, X, T):
        """Creates the ROC diagram for our regression"""

        recall = []
        specif_minus = []
        divisions = 100

        for p in np.linspace(0, 1, num=divisions ):
            y_predict = self.predict(X, p)

            TP, FP, FN, TN = self.confusion_matrix(T, y_predict, values=True)

            recall.append(TP / (TP + FN))
            specif_minus.append(1 - TN/(TN + FP))


        # calc auc score
        auc = 0
        for i in range(len(recall) - 1):
            auc += (recall[i] + recall[i+1])*(specif_minus[i+1] - specif_minus[i])*0.5

        plt.plot(specif_minus, recall, label=f"ROC curve - AUC: {auc}")
        plt.plot([0,1], [0,1],"--")
        plt.legend(loc='lower right')
        plt.ylabel("sensitivity")
        plt.xlabel("1 - specificity")
        plt.show()
