import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import Data.data_provider as dp
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = dp.load_data("Jdata/spambase") #load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

pram_grid = {
    "max_depth" :        np.arange(1,10),
    "min_samples_split": np.arange(2,10),
    "criterion" :        ["gini", "entropy"],
    "max_features" :     ["auto", "sqrt", "log2"],
    "n_estimators":      [10]
}

opt = BayesSearchCV(estimator=RandomForestClassifier(random_state=0), search_spaces=pram_grid, n_iter=10, random_state=0)
opt_result = opt.fit(X_train, y_train)

print("------------------------------------")
params_searched = opt_result.cv_results_["params"]
first_param = params_searched[1]
print(params_searched)
print("------------------------------------")
print(first_param)
print(opt_result.cv_results_["mean_test_score"])
print(opt_result.cv_results_["rank_test_score"])
print("------------------------------------")
# model can be saved, used for predictions or scoring
print(opt.score(X_test, y_test))