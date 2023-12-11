import util
import numpy as np
import matplotlib.pyplot as plt

"""
This is the function through which we train our linear model and 
"""

np.seterr(all='raise')


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Weights Initialization
        """
        self.theta = theta

    def fit(self, X, y):
        """
        Function that Fits model
        """
        self.theta = np.linalg.solve(X.T @ X, X.T @ y)

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.
        """
        return X @ self.theta


def run(train_path, test_path, theta):
    """
    Executes the model and plots against one of the features
    """
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plt.figure(figsize=(10, 6))
    #plt.scatter(train_x[:, 1], train_y, color='green', label='Training Data')
    test_x, test_y = util.load_dataset(test_path, add_intercept=True)
    model = LinearModel()
    if theta == []:
        model.fit(train_x, train_y)
        # Load test data
        test_pred = model.predict(test_x)
        plt.scatter(test_x[:, 1], test_y, label='Test Data')
        plt.scatter(test_x[:, 1], test_pred, color='red', label='Prediction')
        # Mean Squared Error
        ms_error = np.mean((test_pred - test_y) ** 2)
        plt.xlabel('Living Area (sqft)')
        plt.ylabel('Price')
        plt.title(f'Prediction Plot, MSE = {ms_error}')
        plt.legend()
        plt.grid(True)
        plt.savefig("plot_model.jpeg")
        plt.clf()
        print(model.theta)
    else:
        model.theta = theta
        test_pred = model.predict(test_x)
        plt.scatter(test_x[:, 1], test_y, label='Test Data')
        plt.scatter(test_x[:, 1], test_pred, color='red', label='Prediction')
        ms_error = np.mean((test_pred - test_y) ** 2)
        plt.xlabel('Living Area (sqft)')
        plt.ylabel('Price')
        plt.title(f'Prediction Plot, MSE = {ms_error}')
        plt.legend()
        plt.grid(True)
        plt.savefig("q_plot_model.jpeg")
        plt.clf()
        print(model.theta)


def main(train_path, test_path, theta):
    run(train_path, test_path, [])
    run(train_path, test_path, theta)


if __name__ == '__main__':
    main(train_path='train.csv', test_path='test.csv', theta=[0.29535732, 1.02664605, 0.50149502, 0.17650161]
         )
