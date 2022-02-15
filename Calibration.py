import Data.data_provider as dp
from sklearn.ensemble import RandomForestClassifier
import Algo.a_DF as df

data_name = "Jdata/parkinsons"
seed = 1

prams = {
# 'criterion'          : "entropy",
# 'max_features'       : "auto",
'max_depth'          : 10,
'n_estimators'       : 10,
'n_estimator_predict': 10,
'opt_iterations'     : 2,
'epsilon'            : 1.001,
'credal_size'        : 999,
'laplace_smoothing'  : 0,
'split'              : 0.3,
'run_start'          : 0,
'cv'                 : 0,
'opt_decision_model' : True,
'equal_model_prediction'   : False
}

features, target = dp.load_data(data_name)
x_train, x_test, y_train, y_test = dp.split_data(features, target, split=0.3, seed=seed)

predictions , t_unc, e_unc, a_unc, model = df.DF_run(x_train, x_test, y_train, y_test, prams, "bays", seed, opt_decision_model=False, equal_model_prediction=prams["equal_model_prediction"])
probs = model.predict_proba(x_test)