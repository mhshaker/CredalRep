from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np


class ensnnClassifier:

    def __init__(self, n_layers=2, nodes=10, learning_rate='constant', learning_rate_init=0.001, n_estimators=10, random_state=None) -> None:
        self.n_layers = n_layers
        self.nodes = nodes
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init

        hidden_layer_sizes = np.full(self.n_layers, self.nodes)
        # for i in range(self.n_layers):
        #     hidden_layer_sizes.append(self.nodes)

        self.model = BaggingClassifier( bootstrap=True,
                                        base_estimator=MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                                                     learning_rate=self.learning_rate,
                                                                     learning_rate_init=self.learning_rate_init,
                                                                     random_state=self.random_state), # max_iter=500, 
                                        n_estimators=self.n_estimators,
                                        random_state=self.random_state,
                                        verbose=0,
                                        warm_start=False)
    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, x_train, y_train):
        return self.model.fit(x_train, y_train)
    def predict(self, x_test):
        return self.model.predict(x_test)
    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    def score(self, x_test, y_test):
        return self.model.score(x_test, y_test)
    def get_params(self, deep=True):
        return {"n_layers": self.n_layers, "nodes": self.nodes, "n_estimators": self.n_estimators, "random_state": self.random_state, "learning_rate": self.learning_rate, "learning_rate_init": self.learning_rate_init}
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
