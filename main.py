"""This is a simple Logistic Regression model for a target of two classes.
It uses a Gradient descent algorithm to calculate the ideal weights."""

import numpy as np
import pandas as pd


class Logistic_regression:
    """The class that creates the logistic regression model.
    max_iter : the number of maximum iterations of the gradient decent algorithm
    a : the learning rate of gradient decent
    error_diff: the threshold for the cross entropy difference
    show: printing some info about the steps of the processes
    """

    def __init__(self, max_iter=1000, a=0.01, error_diff=1e-7, show=False):
        self.D = None
        self.trained = False
        self.max_iter = max_iter
        self.a = a
        self.w = None
        self.N = None
        self.error_diff = error_diff
        self.show = show
        self.w = None

    @staticmethod
    def sigmoid(z):
        """the transformation of the linear value z = w.T.dot(X)  on the sigmoid line"""
        return 1 / (1 + np.exp(-z))

    def cross_entropy(self, T, y_pred):
        """The cross entropy function in matrix form"""
        return -T.dot(np.log(y_pred)) - (1 - T).dot(np.log(1 - y_pred))

    def fit(self, X, Y):
        """The function fit takes the dependent variables matrix and the target matrix as inputs.
        The input matrices must be np arrays of the correct shape."""

        self.N, self.D = X.shape[0], X.shape[1]

        # creating the x0 = 1 column, for the wo weight (intercept)
        ones = np.array([[1] * self.N]).T
        X = np.concatenate((ones, X), axis=1)

        # initialize weights
        w = 0.01 * np.random.randn(self.D + 1)

        # starting the gradient descent
        iterations = 1
        y_prob = self.sigmoid(X.dot(w))
        e = 1
        e_prev = self.cross_entropy(Y, y_prob)

        # print some starting info before the gradient decent
        if self.show:
            print("Starting fitting process!")
            print(f"Cross entropy is {e_prev}.\n")

        while iterations < self.max_iter and e > self.error_diff:

            # calculating the changes in w
            w_change = self.a * X.T.dot(Y - y_prob)
            w += w_change

            # new predictions
            y_prob = self.sigmoid(X.dot(w))
            e_new = self.cross_entropy(Y, y_prob)

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
        self.w = w

    def get_coef(self):
        if self.trained:
            return self.w
        else:
            print("Model not trained.")

    def y_probab(self, X):
        return self.sigmoid(X.dot(self.w))

    def predict(self, X, p=0.5):
        if self.trained:
            ones = np.array([[1] * X.shape[0]]).T
            X = np.concatenate((ones, X), axis=1)
            return [0 if i < p else 1 for i in self.y_probab(X)]
        else:
            print("Model not trained.")

    def confusion_matrix(self, T, Y_prob):
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

        return np.array([[TP, FN],[FP, TN]])


if __name__ == "__main__":
    df = pd.read_csv("test.csv")
    Y = df.iloc[:, 1].values
    X = df.iloc[:, 2:].values

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0, stratify=Y)
    t_reg2 = Logistic_regression(max_iter=1000, a=0.001)
    t_reg2.fit(X_train, Y_train)

    pred = t_reg2.predict(X_test)
    print(t_reg2.confusion_matrix(Y_test, pred))
