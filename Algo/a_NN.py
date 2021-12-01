import os
import numpy as np
import UncertaintyM as unc
import bays_opt as bo
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


def NN_run(x_train, x_test, y_train, y_test, pram, unc_method, seed, predict=True, opt_pram_list=None):
    np.random.seed(seed)
    us = unc_method.split('_')
    unc_method = us[0]
    if len(us) > 1:
        unc_mode = us[1] # spliting the active selection mode (_a _e _t) from the unc method because DF dose not work with that

    model = None
    model = RandomForestClassifier(bootstrap=True,
        criterion=pram['criterion'],
        max_depth=pram["max_depth"],
        max_features= pram["max_features"],
        n_estimators=pram["n_estimator_predict"],
        random_state=seed,
        verbose=0,
        warm_start=False)
    model.fit(x_train, y_train)
    if predict:
        prediction = model.predict(x_test)
    else:
        prediction = 0


    if "ent" == unc_method:
        porb_matrix = get_prob_matrix(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent(np.array(porb_matrix))
    elif "ent.levi" == unc_method:
        porb_matrix = get_prob_matrix(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_levi(np.array(porb_matrix))
    elif "rl" == unc_method:
        # print("normal rl")
        count_matrix = get_count_matrix(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_rl_avg(count_matrix)
    elif "rl.avgsup" == unc_method:
        # print("rl.avgsup")
        count_matrix = get_sub_count_matrix(model, x_test, pram["n_estimators"], pram["laplace_smoothing"]) # the counting shold be on subset on training data
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.unc_avg_sup_rl(count_matrix)
    elif "rl.score" == unc_method:
        count_matrix = get_sub_count_matrix(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.unc_rl_score(count_matrix)
    elif "rl.pro" == unc_method:
        x = unc.EpiAle_Averaged_Uncertainty_Preferences(model, x_train, y_train, x_test, unc_mode, pram["n_estimators"])
        x = np.array(x)
        total_uncertainty     = x
        epistemic_uncertainty = x 
        aleatoric_uncertainty = x
    elif "rl.alb" == unc_method:
        count_matrix = get_count_matrix(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_rl_ALB(count_matrix)
    elif "rl.int" == unc_method:
        count_matrix = get_int_count_matrix(model, x_train, x_test, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_rl_one(count_matrix)
    elif "rl.uni" == unc_method:
        count_matrix = get_uni_count_matrix(model, x_train, x_test, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_rl_one(count_matrix)
    elif "credal" == unc_method: # this is credal from the tree
        count_matrix = get_count_matrix(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_credal_tree_DF(count_matrix)
    elif "set14" == unc_method:
        porb_matrix = get_prob_matrix(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14(np.array(porb_matrix), pram["credal_size"])
    elif "set15" == unc_method:
        porb_matrix = get_prob_matrix(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set15(np.array(porb_matrix), pram["credal_size"])
    elif "setmix" == unc_method:
        porb_matrix = get_prob_matrix(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_setmix(np.array(porb_matrix))
    elif "set14.convex" == unc_method:
        porb_matrix = get_prob_matrix(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14_convex(porb_matrix, pram["credal_size"])
    elif "set15.convex" == unc_method:
        porb_matrix = get_prob_matrix(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set15_convex(porb_matrix, pram["credal_size"])
    elif "set18" == unc_method:
        likelyhoods = get_likelyhood(model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        # print(">>>> shape ", porb_matrix.shape)
        # exit()
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set18(porb_matrix, likelyhoods, pram["epsilon"])
    elif "set19" == unc_method:
        likelyhoods = get_likelyhood(model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set19(porb_matrix, likelyhoods, pram["epsilon"])

    elif "set20" == unc_method: # set20 [Random grid search] is about credal set with different hyper prameters. We get porb_matrix from different forests but use the same set18 method to have convexcity
        sample_acc_list = []
        credal_prob_matrix = []
        likelyhoods = []
        pram_smaple_list = []

        pram_grid = {
            "max_depth" :        np.arange(1,50),
            "min_samples_split": np.arange(2,10),
            "criterion" :        ["gini", "entropy"],
            "max_features" :     ["auto", "sqrt", "log2"],
            "n_estimators":      [pram["n_estimators"]]
        }

        opt = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=pram_grid, n_iter=pram["opt_iterations"], random_state=seed)
        # print(">>> y_train " , np.unique(y_train))
        opt_result = opt.fit(x_train, y_train)      

        # get ranking and params
        params_searched = np.array(opt_result.cv_results_["params"])
        params_rank = np.array(opt_result.cv_results_["rank_test_score"])
        params_score = np.array(opt_result.cv_results_["mean_test_score"])
        # sprt based on rankings
        sorted_index = np.argsort(params_rank, kind='stable') # sort based on rank
        params_searched = params_searched[sorted_index]
        params_rank = params_rank[sorted_index]
        params_score = params_score[sorted_index]
        # index = index[0]

        # select top K based on kmeans of 2 clusters, we select the cluster with good results to put in the credal set
        kmeans = KMeans(n_clusters=2, random_state=seed).fit(params_score.reshape(-1, 1))
        cluster_result = kmeans.labels_
        index = np.where(cluster_result == 1)[0][0]
        if index == 0:
            index = len(params_score)
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]
        
        # retrain with top K and get test_prob, likelihood values

        credal_prob_matrix = []
        likelyhoods = []

        for param in params_searched: # opt_pram_list: 
            model = None
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)
            test_prob = model.predict_proba(x_test)
            credal_prob_matrix.append(test_prob)
            train_prob = model.predict_proba(x_train)
            likelyhoods.append(log_loss(y_train,train_prob))

        likelyhoods = np.array(likelyhoods)
        likelyhoods = np.exp(-likelyhoods) # convert log likelihood to likelihood
        likelyhoods = likelyhoods / np.sum(likelyhoods) # normalization of the likelihood

        porb_matrix = np.array(credal_prob_matrix)
        porb_matrix = porb_matrix.transpose([1,0,2]) # convert to the format that uncertainty_set14 uses ## laplace smoothing has no effect on set20
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set18(porb_matrix, likelyhoods, pram["epsilon"])
    elif "set21" == unc_method: # Similar to set20
        sample_acc_list = []
        credal_prob_matrix = []
        likelyhoods = []
        pram_smaple_list = []

        pram_grid = {
            "max_depth" :        np.arange(1,50),
            "min_samples_split": np.arange(2,10),
            "criterion" :        ["gini", "entropy"],
            "max_features" :     ["auto", "sqrt", "log2"],
            "n_estimators":      [pram["n_estimators"]]
        }

        opt = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=pram_grid, n_iter=pram["opt_iterations"], random_state=seed)
        # print(">>> y_train " , np.unique(y_train))
        opt_result = opt.fit(x_train, y_train)      

        # get ranking and params
        params_searched = np.array(opt_result.cv_results_["params"])
        params_rank = np.array(opt_result.cv_results_["rank_test_score"])
        params_score = np.array(opt_result.cv_results_["mean_test_score"])
        # sprt based on rankings
        sorted_index = np.argsort(params_rank, kind='stable') # sort based on rank
        params_searched = params_searched[sorted_index]
        params_rank = params_rank[sorted_index]
        params_score = params_score[sorted_index]
        # index = index[0]

        # select top K based on kmeans of 2 clusters, we select the cluster with good results to put in the credal set
        kmeans = KMeans(n_clusters=2, random_state=seed).fit(params_score.reshape(-1, 1))
        cluster_result = kmeans.labels_
        index = np.where(cluster_result == 1)[0][0]
        if index == 0:
            index = len(params_score)
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]
        
        # retrain with top K and get test_prob, likelihood values

        credal_prob_matrix = []
        likelyhoods = []

        for param in params_searched: # opt_pram_list: 
            model = None
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)
            test_prob = model.predict_proba(x_test)
            credal_prob_matrix.append(test_prob)
            train_prob = model.predict_proba(x_train)
            likelyhoods.append(log_loss(y_train,train_prob))

        likelyhoods = np.array(likelyhoods)
        likelyhoods = np.exp(-likelyhoods) # convert log likelihood to likelihood
        likelyhoods = likelyhoods / np.sum(likelyhoods) # normalization of the likelihood

        porb_matrix = np.array(credal_prob_matrix)
        porb_matrix = porb_matrix.transpose([1,0,2]) # convert to the format that uncertainty_set14 uses ## laplace smoothing has no effect on set20
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set19(porb_matrix, likelyhoods, pram["epsilon"])

    elif "set22" == unc_method: # set22 [Bays opt] is about credal set with different hyper prameters. We get porb_matrix from different forests but use the same set18 method to have convexcity

        pram_grid = {
            "max_depth" :        np.arange(1,50),
            "min_samples_split": np.arange(2,10),
            "criterion" :        ["gini", "entropy"],
            "max_features" :     ["auto", "sqrt", "log2"],
            "n_estimators":      [pram["n_estimators"]]
        }

        opt = BayesSearchCV(estimator=RandomForestClassifier(), search_spaces=pram_grid, n_iter=pram["opt_iterations"], random_state=seed)
        print(">>> y_train " , np.unique(y_train))
        opt_result = opt.fit(x_train, y_train)      

        # get ranking and params
        params_searched = np.array(opt_result.cv_results_["params"])
        params_rank = np.array(opt_result.cv_results_["rank_test_score"])
        # sprt based on rankings
        sorted_index = np.argsort(params_rank, kind='stable') # sort based on rank
        params_searched = params_searched[sorted_index]
        params_rank = params_rank[sorted_index]
        # select top K
        params_searched = params_searched[: pram["credal_size"]]
        params_rank = params_rank[: pram["credal_size"]]
        # retrain with top K and get test_prob, likelihood values

        credal_prob_matrix = []
        likelyhoods = []

        for param in params_searched: # opt_pram_list: 
            model = None
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)
            test_prob = model.predict_proba(x_test)
            credal_prob_matrix.append(test_prob)
            train_prob = model.predict_proba(x_train)
            likelyhoods.append(log_loss(y_train,train_prob))

        porb_matrix = np.array(credal_prob_matrix)
        porb_matrix = porb_matrix.transpose([1,0,2]) # convert to the format that uncertainty_set14 uses ## laplace smoothing has no effect on set20
        print(porb_matrix.shape)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set18(porb_matrix, likelyhoods, pram["epsilon"])
    
    elif "set23" == unc_method: # set23 Levi 18 prune trees of forest with clustering based on likelihood  #One super big forest from multiple forests

        likelyhoods = get_likelyhood(model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        # print(">>>> before \n", porb_matrix)
        # print(">>>> before ", likelyhoods)

        # sort based on rankings
        sorted_index = np.argsort(-likelyhoods, kind='stable') # sort based on rank
        likelyhoods = likelyhoods[sorted_index]
        porb_matrix = porb_matrix[:,sorted_index,:]
        # print("------------------------------------")
        # print(">>>> after ", likelyhoods)
        # print(">>>> after \n", porb_matrix)

        # print("likelyhoods ", likelyhoods)
        # print("porb_matrix ", porb_matrix.shape)

        # select top K based on kmeans of 2 clusters, we select the cluster with good results to put in the credal set
        kmeans = KMeans(n_clusters=2, random_state=seed).fit(likelyhoods.reshape(-1, 1))
        cluster_result = kmeans.labels_
        # print("cluster_result ", cluster_result)
        index = np.where(cluster_result == 1)[0][0]
        if index == 0:
            index = len(likelyhoods)
        likelyhoods = likelyhoods[: index]
        likelyhoods = likelyhoods / np.sum(likelyhoods) # normalization after the cut

        # print(likelyhoods)
        porb_matrix = porb_matrix[:, :index,:]

        # print(">>>> porb_matrix.shape ", porb_matrix.shape)
        # exit()
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set18(porb_matrix, likelyhoods, pram["epsilon"])

    elif "set24" == unc_method: # One super big forest from multiple forests
        sample_acc_list = []
        credal_prob_matrix = []
        likelyhoods = []
        pram_smaple_list = []

        pram_grid = {
            "max_depth" :        np.arange(1,50),
            "min_samples_split": np.arange(2,10),
            "criterion" :        ["gini", "entropy"],
            "max_features" :     ["auto", "sqrt", "log2"],
            "n_estimators":      [pram["n_estimators"]]
        }

        opt = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=pram_grid, n_iter=pram["opt_iterations"], random_state=seed)
        # print(">>> y_train " , np.unique(y_train))
        opt_result = opt.fit(x_train, y_train)      

        # get ranking and params
        params_searched = np.array(opt_result.cv_results_["params"])
        params_rank = np.array(opt_result.cv_results_["rank_test_score"])
        params_score = np.array(opt_result.cv_results_["mean_test_score"])
        # sprt based on rankings
        sorted_index = np.argsort(params_rank, kind='stable') # sort based on rank
        params_searched = params_searched[sorted_index]
        params_rank = params_rank[sorted_index]
        params_score = params_score[sorted_index]
        # index = index[0]

        # select top K based on kmeans of 2 clusters, we select the cluster with good results to put in the credal set
        kmeans = KMeans(n_clusters=2, random_state=seed).fit(params_score.reshape(-1, 1))
        cluster_result = kmeans.labels_
        index = np.where(cluster_result == 1)[0][0]
        if index == 0:
            index = len(params_score)
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]
        print("cut index ", index)
        # retrain with top K and get test_prob, likelihood values

        credal_prob_matrix = []
        likelyhoods = []

        for param in params_searched: # opt_pram_list: 
            model = None
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)
            
            # test_prob = model.predict_proba(x_test)
            # credal_prob_matrix.append(test_prob)
            # train_prob = model.predict_proba(x_train)
            # likelyhoods.append(log_loss(y_train,train_prob))

            likelyhoods.append(get_likelyhood(model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"]))
            if credal_prob_matrix == []:
                credal_prob_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
            else:
                credal_prob_matrix = np.concatenate((credal_prob_matrix, get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])), axis=1)
            # credal_prob_matrix.append(get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"]))


        likelyhoods = np.array(likelyhoods)
        likelyhoods = likelyhoods.reshape(-1)

        # print("likelyhoods forests ", likelyhoods.shape)
        likelyhoods = np.exp(-likelyhoods) # convert log likelihood to likelihood
        likelyhoods = likelyhoods / np.sum(likelyhoods) # normalization of the likelihood

        porb_matrix = np.array(credal_prob_matrix)
        # porb_matrix = porb_matrix.reshape(porb_matrix.shape[1], -1,porb_matrix.shape[3])
        # print("------------------------------------")
        # print("porb_matrix forests ", porb_matrix.shape)
        # porb_matrix = porb_matrix.transpose([1,0,2]) # convert to the format that uncertainty_set14 uses ## laplace smoothing has no effect on set20
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set18(porb_matrix, likelyhoods, pram["epsilon"])

    elif "set25" == unc_method: # One super big forest from multiple forests
        sample_acc_list = []
        credal_prob_matrix = []
        likelyhoods = []
        pram_smaple_list = []

        pram_grid = {
            "max_depth" :        np.arange(1,50),
            "min_samples_split": np.arange(2,10),
            "criterion" :        ["gini", "entropy"],
            "max_features" :     ["auto", "sqrt", "log2"],
            "n_estimators":      [pram["n_estimators"]]
        }

        opt = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=pram_grid, n_iter=pram["opt_iterations"], random_state=seed)
        # print(">>> y_train " , np.unique(y_train))
        opt_result = opt.fit(x_train, y_train)      

        # get ranking and params
        params_searched = np.array(opt_result.cv_results_["params"])
        params_rank = np.array(opt_result.cv_results_["rank_test_score"])
        params_score = np.array(opt_result.cv_results_["mean_test_score"])
        # sprt based on rankings
        sorted_index = np.argsort(params_rank, kind='stable') # sort based on rank
        params_searched = params_searched[sorted_index]
        params_rank = params_rank[sorted_index]
        params_score = params_score[sorted_index]
        # index = index[0]

        # select top K based on kmeans of 2 clusters, we select the cluster with good results to put in the credal set
        kmeans = KMeans(n_clusters=2, random_state=seed).fit(params_score.reshape(-1, 1))
        cluster_result = kmeans.labels_
        index = np.where(cluster_result == 1)[0][0]
        if index == 0:
            index = len(params_score)
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]
        print("cut index ", index)
        # retrain with top K and get test_prob, likelihood values

        credal_prob_matrix = []
        likelyhoods = []

        for param in params_searched: # opt_pram_list: 
            model = None
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)
            
            # test_prob = model.predict_proba(x_test)
            # credal_prob_matrix.append(test_prob)
            # train_prob = model.predict_proba(x_train)
            # likelyhoods.append(log_loss(y_train,train_prob))

            likelyhoods.append(get_likelyhood(model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"]))
            if credal_prob_matrix == []:
                credal_prob_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
            else:
                credal_prob_matrix = np.concatenate((credal_prob_matrix, get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])), axis=1)
            # credal_prob_matrix.append(get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"]))


        likelyhoods = np.array(likelyhoods)
        likelyhoods = likelyhoods.reshape(-1)

        # print("likelyhoods forests ", likelyhoods.shape)
        likelyhoods = np.exp(-likelyhoods) # convert log likelihood to likelihood
        likelyhoods = likelyhoods / np.sum(likelyhoods) # normalization of the likelihood

        porb_matrix = np.array(credal_prob_matrix)
        # porb_matrix = porb_matrix.reshape(porb_matrix.shape[1], -1,porb_matrix.shape[3])
        # print("------------------------------------")
        # print("porb_matrix forests ", porb_matrix.shape)
        # porb_matrix = porb_matrix.transpose([1,0,2]) # convert to the format that uncertainty_set14 uses ## laplace smoothing has no effect on set20
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set19(porb_matrix, likelyhoods, pram["epsilon"])


    elif "out.tree" == unc_method:
        porb_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_outcome_tree(porb_matrix)
    elif "out" == unc_method:
        likelyhoods = get_likelyhood(model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_outcome(porb_matrix, likelyhoods, pram["epsilon"])
    elif "bays" == unc_method:
        model = None
        model = RandomForestClassifier(bootstrap=True,
            criterion=pram['criterion'],
            max_depth=pram["max_depth"],
            n_estimators=pram["n_estimators"],
            max_features= pram["max_features"],
            random_state=seed,
            verbose=0,
            warm_start=False)
        model.fit(x_train, y_train)
        likelyhoods = get_likelyhood(model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        # print("bays likelyhoods >>>>>>", likelyhoods)
        # accs = get_acc(model, x_train, y_train, pram["n_estimators"])
        porb_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_bays(porb_matrix, likelyhoods)
    elif "levi3" in unc_method:
        porb_matrix = get_prob_matrix(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        if "levi3.GH" == unc_method:
            total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14(porb_matrix, sampling_size=pram["credal_sample_size"], credal_size=pram["credal_size"])
        elif "levi3.ent" == unc_method:
            total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set15(porb_matrix, sampling_size=pram["credal_sample_size"], credal_size=pram["credal_size"])
        # elif "levi3.GH.conv" == unc_method:
        #     total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14_convex(porb_matrix, sampling_size=pram["credal_size"])
        # elif "levi3.ent.conv" == unc_method:
        #     total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set15_convex(porb_matrix, sampling_size=pram["credal_size"])
    
    elif "levidir" in unc_method:
        credal_prob_matrix = []
        # credal_likelihood_matrix = []
        for a in np.linspace(1,pram["credal_L"],pram["credal_size"]): #range(1, pram["credal_L"] + 1):
            for b in np.linspace(1,pram["credal_L"],pram["credal_size"]): #range(1, pram["credal_L"] + 1):
                # print(f"a {a} b {b}")
                porb_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"],a,b)
                likelyhoods = get_likelyhood(model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"],a,b)

                # print("likelyhoods  ", likelyhoods)
                # print("before porb_matrix\n ", porb_matrix)
                
                likelyhoods = np.reshape(likelyhoods, (1,-1,1))
                porb_matrix = porb_matrix * likelyhoods
                porb_matrix = np.sum(porb_matrix, axis=1)
                
                # porb_matrix = np.mean(porb_matrix, axis=1)


                # print("after porb_matrix\n ", porb_matrix)
                # print("porb_matrix.shape ", porb_matrix.shape)
                # print("likelyhoods.shape ", likelyhoods.shape)
                # print(aaa)
                credal_prob_matrix.append(porb_matrix)
                # credal_likelihood_matrix.append(likelyhoods)

        porb_matrix = np.array(credal_prob_matrix)
        porb_matrix = porb_matrix.transpose([1,0,2]) 

        if "levidir.GH" == unc_method:
            total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14(porb_matrix)
        elif "levidir.ent" == unc_method:
            total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set15(porb_matrix)
        elif "levidir.GH.conv" == unc_method:
            total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14_convex(porb_matrix)
        elif "levidir.ent.conv" == unc_method:
            total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set15_convex(porb_matrix)

    
    elif "levi" in unc_method:
        credal_prob_matrix = []
        for credal_index in range(pram["credal_size"]):
            model = None
            model = RandomForestClassifier(bootstrap=True,
                # criterion=pram['criterion'],
                max_depth=pram["max_depth"],
                n_estimators=pram["n_estimators"],
                # max_features= "sqrt",
                # min_samples_leaf= pram['min_samples_leaf'],
                random_state=(seed+100) * credal_index,
                verbose=0,
                warm_start=False)
            model.fit(x_train, y_train)

            credal_prob_matrix.append(get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"]))
        porb_matrix = np.array(credal_prob_matrix)
        porb_matrix = np.mean(porb_matrix, axis=2) # average all the trees in each forest
        porb_matrix = porb_matrix.transpose([1,0,2]) # convert to the format that uncertainty_set14 uses

        if "levi.GH" == unc_method:
            total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14(porb_matrix)
        elif "levi.ent" == unc_method:
            total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set15(porb_matrix)
        elif "levi.GH.conv" == unc_method:
            total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14_convex(porb_matrix)
        elif "levi.ent.conv" == unc_method:
            total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set15_convex(porb_matrix)

    elif "gs" == unc_method:
        porb_matrix, likelyhoods = ens_boot_likelihood(model, x_train, y_train, x_test, pram["n_estimators"], pram["credal_size"],pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_gs(porb_matrix, likelyhoods, pram["credal_size"])
    elif "random" == unc_method:
        total_uncertainty = np.random.rand(len(x_test))
        epistemic_uncertainty = np.random.rand(len(x_test))
        aleatoric_uncertainty = np.random.rand(len(x_test))
    else:
        print(f"[Error] No implementation of unc_method {unc_method} for DF")

    return prediction, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, model
