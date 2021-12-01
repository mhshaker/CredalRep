import os
import numpy as np
import UncertaintyM as unc
import random
from  ens_nn import ensnnClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import log_loss
import math

def NN_run(x_train, x_test, y_train, y_test, pram, unc_method, seed, predict=True, opt_decision_model=True, log=True):
    np.random.seed(seed)
    us = unc_method.split('_')
    unc_method = us[0]
    if len(us) > 1:
        unc_mode = us[1] # spliting the active selection mode (_a _e _t) from the unc method because DF dose not work with that

    if opt_decision_model or "set30" in unc_method or "set31" in unc_method:

        pram_grid = {
            "nodes"             : np.arange(int(math.sqrt(x_train.shape[1])),x_train.shape[1]),  
            "n_layers"          : np.arange(2,10),
            "learning_rate"     : ['constant', 'invscaling', 'adaptive'],
            "learning_rate_init" : [0.001, 0.005, 0.01],
            "n_estimators"      : [pram["n_estimators"]]
        }

        opt = RandomizedSearchCV(estimator=ensnnClassifier(), param_distributions=pram_grid, n_iter=pram["opt_iterations"], cv=3, random_state=seed)
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
        main_model = ensnnClassifier(3, 20, pram["n_estimator_predict"], seed)
        
    else:
        main_model = ensnnClassifier(**params_searched[0],random_state=seed)

    main_model.fit(x_train, y_train)

    if predict:
        prediction = main_model.predict(x_test)
    else:
        prediction = 0

    if "bays" == unc_method:
        likelyhoods = get_likelyhood(main_model.model, x_train, y_train, pram["laplace_smoothing"])
        porb_matrix = get_prob(main_model.model, x_test, pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_bays(porb_matrix, likelyhoods)
    elif "set30" == unc_method: # GH credal set from hyper forests
        credal_prob_matrix = []
        # Confidance interval
        conf_int = params_score_mean[0] -  1 * params_score_std[0] # include SD which is 99%
        index = len(params_score_mean) - 1
        while params_score_mean[index] < conf_int:
            index -= 1
        if index == 0:
            index = 1
        
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]
        if log:
            print("------------------------------------")
            print(f"conf_int cut index {index}")

        for i, param in enumerate(params_searched): # opt_pram_list:
            if log: 
                print(f"Acc:{params_score_mean[i]:.4f} +-{params_score_std[i]:.4f} {param}")  # Eyke log
            model = None
            model = ensnnClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)
            test_prob = model.predict_proba(x_test)
            credal_prob_matrix.append(test_prob)

        porb_matrix = np.array(credal_prob_matrix)
        porb_matrix = porb_matrix.transpose([1,0,2]) # convert to the format that uncertainty_set14 uses ## laplace smoothing has no effect on set20
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14(porb_matrix)

    elif "set31" == unc_method: # Ent version of set30
        credal_prob_matrix = []
        likelyhoods = []
        # Confidance interval
        conf_int = params_score_mean[0] -  2 * params_score_std[0] # include SD which is 99%
        index = len(params_score_mean) - 1
        while params_score_mean[index] < conf_int:
            index -= 1
        if index == 0:
            index = 1
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]

        for param in params_searched: # opt_pram_list: 
            model = None
            model = ensnnClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)
            test_prob = model.predict_proba(x_test)
            credal_prob_matrix.append(test_prob)

        porb_matrix = np.array(credal_prob_matrix)
        porb_matrix = porb_matrix.transpose([1,0,2]) # convert to the format that uncertainty_set14 uses ## laplace smoothing has no effect on set20
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set15(porb_matrix)

    elif "random" == unc_method:
        total_uncertainty = np.random.rand(len(x_test))
        epistemic_uncertainty = np.random.rand(len(x_test))
        aleatoric_uncertainty = np.random.rand(len(x_test))
    else:
        print(f"[Error] No implementation of unc_method {unc_method} for DF")

    return prediction, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, main_model


def get_likelyhood(model_ens, x_train, y_train, laplace_smoothing, a=0, b=0, log=False):
    likelyhoods  = []
    for estimator in model_ens.estimators_:
        if laplace_smoothing == 0 and a==0 and b==0:
            tree_prob_train = estimator.predict_proba(x_train) 
        else:
            pass

        likelyhoods.append(log_loss(y_train,tree_prob_train))
    likelyhoods = np.array(likelyhoods)
    likelyhoods = np.exp(-likelyhoods) # convert log likelihood to likelihood
    likelyhoods = likelyhoods / np.sum(likelyhoods) # normalization of the likelihood

    if log:
        print(f"<log>----------------------------------------[]")
        print(f"likelyhoods = {likelyhoods}")
    return np.array(likelyhoods)

def get_prob(model_ens, x_data, laplace_smoothing, a=0, b=0, log=False):
    prob_matrix  = []
    for estimator in model_ens.estimators_:
        if laplace_smoothing == 0 and a==0 and b==0:
            tree_prob = estimator.predict_proba(x_data) 
        else:
            pass
        prob_matrix.append(tree_prob)
    if log:
        print(f"<log>----------------------------------------[]")
        print(f"prob_matrix = {prob_matrix}")
    prob_matrix = np.array(prob_matrix)
    prob_matrix = prob_matrix.transpose([1,0,2]) # D1 = data index D2= ens tree index D3= prediction prob for classes
    return prob_matrix
