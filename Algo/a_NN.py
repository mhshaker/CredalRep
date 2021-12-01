import os
import numpy as np
import UncertaintyM as unc
import random
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans


def NN_run(x_train, x_test, y_train, y_test, pram, unc_method, seed, predict=True, opt_decision_model=True, log=False):
    np.random.seed(seed)
    us = unc_method.split('_')
    unc_method = us[0]
    if len(us) > 1:
        unc_mode = us[1] # spliting the active selection mode (_a _e _t) from the unc method because DF dose not work with that

    if opt_decision_model or unc_method =="set24" or unc_method =="set25" or "set30" in unc_method or "set31" in unc_method or unc_method =="set32": # or 27 or 28 or 24mix but they are not good and I dont plan on using them
        # find max depth range
        depth_model = RandomForestClassifier(pram["n_estimator_predict"], random_state=seed)
        depth_model.fit(x_train, y_train)
        max_depth = 0
        for estimator in depth_model.estimators_:
            d = estimator.tree_.max_depth
            if d > max_depth:
                max_depth = d
        if log:
            print("------------------------------------max_depth test ")
            print(max_depth)

        pram_grid = {
            "max_depth" :        np.arange(1,max_depth),
            "criterion" :        ["gini", "entropy"],
            "max_features" :     ["sqrt", "log2"],
            "n_estimators":      [pram["n_estimators"]]
        }

        opt = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=pram_grid, n_iter=pram["opt_iterations"], cv=10, random_state=seed)
        opt_result = opt.fit(x_train, y_train)      

        params_searched = np.array(opt_result.cv_results_["params"])
        params_rank = np.array(opt_result.cv_results_["rank_test_score"])
        params_score_mean = np.array(opt_result.cv_results_["mean_test_score"])
        params_score_std  = np.array(opt_result.cv_results_["std_test_score"])

        # sort based on rankings
        sorted_index = np.argsort(params_rank, kind='stable') # sort based on rank
        params_searched = params_searched[sorted_index]
        params_rank = params_rank[sorted_index]
        params_score_mean = params_score_mean[sorted_index]
        params_score_std = params_score_std[sorted_index]
        # print(f"Acc:{params_score_mean[0]:.4f} +-{params_score_std[0]:.4f} {params_searched[0]}")

        if log:
            print("------------------------------------params_searched")
            for i, param in enumerate(params_searched):
                print(f"Acc:{params_score_mean[i]:.4f} +-{params_score_std[i]:.4f} {param}")  # Eyke log


    main_model = None
    if opt_decision_model == False:
        main_model = RandomForestClassifier(bootstrap=True,
            # criterion=pram['criterion'],
            max_depth=pram["max_depth"],
            # max_features= pram["max_features"],
            n_estimators=pram["n_estimator_predict"],
            random_state=seed,
            verbose=0,
            warm_start=False)
    else:
        main_model = RandomForestClassifier(**params_searched[0],random_state=seed)

    main_model.fit(x_train, y_train)

    if predict:
        prediction = main_model.predict(x_test)
    else:
        prediction = 0

    if "bays" == unc_method:
        likelyhoods = get_likelyhood(main_model, x_train, y_train, pram["laplace_smoothing"])
        porb_matrix = get_prob(main_model, x_test, pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_bays(porb_matrix, likelyhoods)


    elif "random" == unc_method:
        total_uncertainty = np.random.rand(len(x_test))
        epistemic_uncertainty = np.random.rand(len(x_test))
        aleatoric_uncertainty = np.random.rand(len(x_test))
    else:
        print(f"[Error] No implementation of unc_method {unc_method} for DF")

    return prediction, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, model


def get_likelyhood(model_ens, x_train, y_train, laplace_smoothing, a=0, b=0, log=False):
    pass

def get_prob(model_ens, x_data, laplace_smoothing, a=0, b=0, log=False):
    pass