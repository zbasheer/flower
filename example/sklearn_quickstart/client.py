import numpy as np
from sklearn import datasets
# from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import flwr as fl
from flwr.common import Weights

# mnist = fetch_openml('mnist_784', version=1, cache=True)
X, y = datasets.load_digits(return_X_y=True)
X = X / 255
X_train = X[:-100]
y_train = y[:-100]
X_test = X[-100:-5]
y_test = y[-100:-5]

# Class to overwrite MLPClassifier that sets weights and bias
class MLPClassifierOverride(MLPClassifier):
# Overriding _init_coef method
    def _init_coef(self, fan_in, fan_out):
        # fan_in: number of incoming features
        # fan_out: number of outgoing features 
        # weights is a matrices of mxn --> fan_in x fan_out
        factor = 6.
        if self.activation == 'logistic':
            factor = 2.
        #init_bound = np.sqrt(factor / (fan_in + fan_out))
        coef_init = np.random.uniform(0,0.1, size=(fan_in,fan_out))
        intercept_init = np.random.uniform(0,1, size=(fan_out))
        return coef_init, intercept_init

model = MLPClassifierOverride(solver='adam', alpha=1e-4,learning_rate_init=0.1, 
                    hidden_layer_sizes=(50,), random_state=0, max_iter=100)

class MnistClient(fl.client.KerasClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        #super().__init__(cid)
        self.model = model
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

    def get_weights(self) -> Weights: 
        return model._init_coef

    def fit(self, weights, config):
        self.model.fit(self.X_train, self.y_train)
        weights = model.coefs_
        return weights, len(self.x_train), len(self.x_train)

    def evaluate(self, weights, config):
        accuracy = self.model.score(self.X_test, self.y_test)
        loss = self.model.loss_
        return len(self.X_test), loss, accuracy


if __name__ == "__main__":
    print('Hello Flower Scikit')
    print('Loading dataset digits')
    print('Setting up classifier: MLP')
    print('Start training')
    client = MnistClient(model, X_train, y_train, X_test, y_test)
    #fl.client.start_keras_client(server_address="[::]:8080", client=client)
    fl.client.start_keras_client(server_address="[::]:8080", client=client)
    #model.fit(X_train, y_train)
    #print('Fit results')
    #fit_results = model.score(X_train, y_train)
    #prediction = mlp.predict(X[-5:])
    # First layer is 64 x 50 
    #weights_firstlayer = model.coefs_[0]
    #print('First Layer Coefs', weights_firstlayer)
    # Second layer is 50 x 10
    #weights_secondlayer = model.coefs_[1]
    print('Print Digit Dataset:', X.data.shape)
    #print('Score results:', fit_results)
    #print(prediction)
