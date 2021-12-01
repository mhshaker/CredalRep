from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier


class ens_nn:

    def __init__(self, n_layers, nodes, n_estimators, random_state) -> None:

        hidden_layer_sizes = []
        for i in range(n_layers):
            hidden_layer_sizes.append(nodes)

        self.model = BaggingClassifier( bootstrap=True,
                                        base_estimator=MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                                                                     random_state=random_state), # max_iter=500, 
                                        n_estimators=n_estimators,
                                        random_state=random_state,
                                        verbose=0,
                                        warm_start=False)

    def predict(self, x_test):
        return self.model.predict(x_test)
    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)