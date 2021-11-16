import os
import numpy as np
import UncertaintyM as unc
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans


def DF_run(x_train, x_test, y_train, y_test, pram, unc_method, seed, predict=True, opt_decision_model=True, log=False):
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
            # "min_samples_split": np.arange(0.01,0.1, 0.02),
            # "min_samples_leaf":  np.arange(0.01,0.1, 0.02),
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
            # print(params_searched)
            # print("------------------------------------params_rank")
            # print(params_rank)
            # print("------------------------------------params_score_mean")
            # print(params_score_mean)
            # print("------------------------------------params_score_std")
            # print(params_score_std)


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

    # if log:
    #     print("------------------------------------ tree depth")
    #     for estimator in main_model.estimators_:
    #         d = estimator.tree_.max_depth
    #         print(d)


    if predict:
        prediction = main_model.predict(x_test)
    else:
        prediction = 0


    if "ent" == unc_method:
        porb_matrix = get_prob_matrix(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent(np.array(porb_matrix))
    elif "ent.levi" == unc_method:
        porb_matrix = get_prob_matrix(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_levi(np.array(porb_matrix))
    elif "rl" == unc_method:
        # print("normal rl")
        count_matrix = get_count_matrix(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_rl_avg(count_matrix)
    elif "rl.avgsup" == unc_method:
        # print("rl.avgsup")
        count_matrix = get_sub_count_matrix(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"]) # the counting shold be on subset on training data
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.unc_avg_sup_rl(count_matrix)
    elif "rl.score" == unc_method:
        count_matrix = get_sub_count_matrix(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.unc_rl_score(count_matrix)
    elif "rl.pro" == unc_method:
        x = unc.EpiAle_Averaged_Uncertainty_Preferences(main_model, x_train, y_train, x_test, unc_mode, pram["n_estimators"])
        x = np.array(x)
        total_uncertainty     = x
        epistemic_uncertainty = x 
        aleatoric_uncertainty = x
    elif "rl.alb" == unc_method:
        count_matrix = get_count_matrix(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_rl_ALB(count_matrix)
    elif "rl.int" == unc_method:
        count_matrix = get_int_count_matrix(main_model, x_train, x_test, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_rl_one(count_matrix)
    elif "rl.uni" == unc_method:
        count_matrix = get_uni_count_matrix(main_model, x_train, x_test, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_rl_one(count_matrix)
    elif "credal" == unc_method: # this is credal from the tree
        count_matrix = get_count_matrix(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_credal_tree_DF(count_matrix)
    elif "set14" == unc_method:
        porb_matrix = get_prob_matrix(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14(np.array(porb_matrix), pram["credal_size"])
    elif "set15" == unc_method:
        porb_matrix = get_prob_matrix(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set15(np.array(porb_matrix), pram["credal_size"])
    elif "setmix" == unc_method:
        porb_matrix = get_prob_matrix(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_setmix(np.array(porb_matrix))
    elif "set14.convex" == unc_method:
        porb_matrix = get_prob_matrix(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14_convex(porb_matrix, pram["credal_size"])
    elif "set15.convex" == unc_method:
        porb_matrix = get_prob_matrix(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set15_convex(porb_matrix, pram["credal_size"])
    elif "set18" == unc_method:
        likelyhoods = get_likelyhood(main_model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        # replace likelyhoods with uniform - just for test
        # x0 = np.ones((likelyhoods.shape[0]))
        # x0_sum = np.sum(x0)
        # x0 = x0 / x0_sum
        # likelyhoods = x0
        # print(likelyhoods)
        porb_matrix = get_prob(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set18(porb_matrix, likelyhoods, pram["epsilon"])
    elif "set19" == unc_method:
        likelyhoods = get_likelyhood(main_model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = get_prob(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set19(porb_matrix, likelyhoods, pram["epsilon"])

    elif "set20" == unc_method: # set20 [Random grid search] is about credal set with different hyper prameters. We get porb_matrix from different forests but use the same set18 method to have convexcity
        sample_acc_list = []
        credal_prob_matrix = []
        likelyhoods = []
        pram_smaple_list = []

        # select top K based on kmeans of 2 clusters, we select the cluster with good results to put in the credal set
        kmeans = KMeans(n_clusters=2, random_state=seed).fit(params_score_mean.reshape(-1, 1))
        cluster_result = kmeans.labels_
        index = np.where(cluster_result == 1)[0][0]
        if index == 0:
            index = len(params_score_mean)
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

        # select top K based on kmeans of 2 clusters, we select the cluster with good results to put in the credal set
        kmeans = KMeans(n_clusters=2, random_state=seed).fit(params_score_mean.reshape(-1, 1))
        cluster_result = kmeans.labels_
        index = np.where(cluster_result == 1)[0][0]
        if index == 0:
            index = len(params_score_mean)
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

        likelyhoods = get_likelyhood(main_model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = get_prob(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
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

        # Confidance interval
        conf_int = params_score_mean[0] -  1 * params_score_std[0] # include SD which is 99%
        # print("conf_int ", conf_int)
        index = len(params_score_mean) - 1
        while params_score_mean[index] < conf_int:
            index -= 1
            # print(f"{index}     {params_score_mean[index]} < {conf_int}")
        if index == 0:
            index = 1
        
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]
        print("cut index ", index)
        # retrain with top K and get test_prob, likelihood values

        credal_prob_matrix = []
        likelyhoods = []

        for param in params_searched: # opt_pram_list: 
            model = None
            model = RandomForestClassifier(**param,random_state=42) # seed
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

        # Confidance interval
        conf_int = params_score_mean[0] -  1 * params_score_std[0] # include SD which is 99%
        # print("conf_int ", conf_int)
        index = len(params_score_mean) - 1
        while params_score_mean[index] < conf_int:
            index -= 1
            # print(f"{index}     {params_score_mean[index]} < {conf_int}")
        if index == 0:
            index = 1
        
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]
        print("cut index ", index)
        # retrain with top K and get test_prob, likelihood values

        credal_prob_matrix = []
        likelyhoods = []

        for param in params_searched: # opt_pram_list: 
            model = None
            model = RandomForestClassifier(**param,random_state=42)
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

    elif "set26" == unc_method: # Same as set24 but no clustering
        sample_acc_list = []
        credal_prob_matrix = []
        likelyhoods = []
        pram_smaple_list = []

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

    elif "set27" == unc_method: # Same as 24 but with confidance interval instead of clustering
        sample_acc_list = []
        credal_prob_matrix = []
        likelyhoods = []
        pram_smaple_list = []

        # Confidance interval
        conf_int = params_score_mean[0] -  1 * params_score_std[0] # include SD which is 99%
        # print("conf_int ", conf_int)
        index = len(params_score_mean) - 1
        while params_score_mean[index] < conf_int:
            index -= 1
            # print(f"{index}     {params_score_mean[index]} < {conf_int}")
        if index == 0:
            index = 1
        # print("------------------------------------")    
        # print("index ", index)
        # print("conf_int ", conf_int)
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]
        # print("cut index ", index)
        # retrain with top K and get test_prob, likelihood values

        credal_prob_matrix = []
        likelyhoods = []

        for param in params_searched: # opt_pram_list: 
            model = None
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)
            
            test_prob = model.predict_proba(x_test)
            credal_prob_matrix.append(test_prob)
            # train_prob = model.predict_proba(x_train)
            # likelyhoods.append(log_loss(y_train,train_prob))

            # likelyhoods.append(get_likelyhood(model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"]))
            # if credal_prob_matrix == []:
            #     credal_prob_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
            # else:
            #     credal_prob_matrix = np.concatenate((credal_prob_matrix, get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])), axis=1)


        # likelyhoods = np.array(likelyhoods)
        # likelyhoods = likelyhoods.reshape(-1)

        # likelyhoods = np.exp(-likelyhoods) # convert log likelihood to likelihood
        # likelyhoods = likelyhoods / np.sum(likelyhoods) # normalization of the likelihood

        porb_matrix = np.array(credal_prob_matrix)
        porb_matrix = porb_matrix.transpose([1,0,2]) # convert to the format that uncertainty_set14 uses ## laplace smoothing has no effect on set20
        # print("------------------------------------porb_matrix.shape")
        # print(porb_matrix.shape)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14(porb_matrix)

    elif "set28" == unc_method: # Ent version of set27
        sample_acc_list = []
        credal_prob_matrix = []
        likelyhoods = []
        pram_smaple_list = []

        # Confidance interval
        conf_int = params_score_mean[0] -  1 * params_score_std[0] # include SD which is 99%
        index = len(params_score_mean) - 1
        while params_score_mean[index] < conf_int:
            index -= 1
        print("index ", index)
        # print("conf_int ", conf_int)
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]
        # print("cut index ", index)
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


        # likelyhoods = np.array(likelyhoods)
        # likelyhoods = likelyhoods.reshape(-1)

        # likelyhoods = np.exp(-likelyhoods) # convert log likelihood to likelihood
        # likelyhoods = likelyhoods / np.sum(likelyhoods) # normalization of the likelihood

        porb_matrix = np.array(credal_prob_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set15(porb_matrix)
        
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
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)
            test_prob = model.predict_proba(x_test)
            credal_prob_matrix.append(test_prob)

        porb_matrix = np.array(credal_prob_matrix)
        porb_matrix = porb_matrix.transpose([1,0,2]) # convert to the format that uncertainty_set14 uses ## laplace smoothing has no effect on set20
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14(porb_matrix)
        # print("------------------------------------GH") # Eyke log
        # print(epistemic_uncertainty[0:5])
        # print("------------------------------------probs")
        # print(porb_matrix[0:5])

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
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)
            test_prob = model.predict_proba(x_test)
            credal_prob_matrix.append(test_prob)

        porb_matrix = np.array(credal_prob_matrix)
        porb_matrix = porb_matrix.transpose([1,0,2]) # convert to the format that uncertainty_set14 uses ## laplace smoothing has no effect on set20
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set15(porb_matrix)
    elif "set30.convex" == unc_method: # GH convex credal set from hyper forests
        # Confidance interval
        conf_int = params_score_mean[0] -  1 * params_score_std[0] # include SD which is 99%
        index = len(params_score_mean) - 1
        while params_score_mean[index] < conf_int:
            index -= 1
        if index == 0:
            index = 1
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]

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
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set30(porb_matrix, likelyhoods)

    elif "set31.convex" == unc_method: # Ent version of set30
        # Confidance interval
        conf_int = params_score_mean[0] -  1 * params_score_std[0] # include SD which is 99%
        index = len(params_score_mean) - 1
        while params_score_mean[index] < conf_int:
            index -= 1
        if index == 0:
            index = 1
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]

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
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set31(porb_matrix, likelyhoods)

    elif "set32" == unc_method: # GH convex credal set from trees of hyper forests
        # Confidance interval
        conf_int = params_score_mean[0] -  1 * params_score_std[0] # include SD which is 99%
        index = len(params_score_mean) - 1
        while params_score_mean[index] < conf_int:
            index -= 1
        if index == 0:
            index = 1
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]

        credal_prob_matrix = []
        likelyhoods = []

        for param in params_searched: # opt_pram_list: 
            model = None
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)
            
            likelyhoods.append(get_likelyhood(model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"]))
            if credal_prob_matrix == []:
                credal_prob_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
            else:
                credal_prob_matrix = np.concatenate((credal_prob_matrix, get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])), axis=1)

        porb_matrix = np.array(credal_prob_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set14(porb_matrix)
    elif "set32.convex" == unc_method: # GH convex credal set from trees of hyper forests
        # Confidance interval
        conf_int = params_score_mean[0] -  1 * params_score_std[0] # include SD which is 99%
        index = len(params_score_mean) - 1
        while params_score_mean[index] < conf_int:
            index -= 1
        if index == 0:
            index = 1
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]

        credal_prob_matrix = []
        likelyhoods = []

        for param in params_searched: # opt_pram_list: 
            model = None
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)
            
            likelyhoods.append(get_likelyhood(model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"]))
            if credal_prob_matrix == []:
                credal_prob_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
            else:
                credal_prob_matrix = np.concatenate((credal_prob_matrix, get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])), axis=1)

        porb_matrix = np.array(credal_prob_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_set30(porb_matrix,likelyhoods)

    elif "set24mix" == unc_method: # mix of 24 tree for epist and forest for ale. with conf int
        credal_prob_matrix = []
        likelyhoods = []
        credal_prob_matrix_f = []
        likelyhoods_f = []

        # Confidance interval
        conf_int = params_score_mean[0] -  1 * params_score_std[0] # include SD which is 99%
        index = len(params_score_mean) - 1
        while params_score_mean[index] < conf_int:
            index -= 1
        if index == 0:
            index = 1
        
        params_searched = params_searched[: index]
        params_rank = params_rank[: index]

        credal_prob_matrix = []
        likelyhoods = []

        for param in params_searched: # opt_pram_list: 
            model = None
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)
            
            test_prob = model.predict_proba(x_test)
            credal_prob_matrix_f.append(test_prob)
            train_prob = model.predict_proba(x_train)
            likelyhoods_f.append(log_loss(y_train,train_prob))

            likelyhoods.append(get_likelyhood(model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"]))
            if credal_prob_matrix == []:
                credal_prob_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
            else:
                credal_prob_matrix = np.concatenate((credal_prob_matrix, get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])), axis=1)

        likelyhoods = np.array(likelyhoods)
        likelyhoods = likelyhoods.reshape(-1)

        likelyhoods_f = np.array(likelyhoods_f)
        likelyhoods_f = likelyhoods_f.reshape(-1)

        # print("likelyhoods forests ", likelyhoods.shape)
        likelyhoods = np.exp(-likelyhoods) # convert log likelihood to likelihood
        likelyhoods = likelyhoods / np.sum(likelyhoods) # normalization of the likelihood
        likelyhoods_f = np.exp(-likelyhoods_f) # convert log likelihood to likelihood
        likelyhoods_f = likelyhoods_f / np.sum(likelyhoods_f) # normalization of the likelihood

        porb_matrix = np.array(credal_prob_matrix)
        porb_matrix_f = np.array(credal_prob_matrix_f)
        porb_matrix_f = porb_matrix_f.transpose([1,0,2]) # convert to the format that uncertainty_set14 uses ## laplace smoothing has no effect on set20
        _, epistemic_uncertainty, _ = unc.uncertainty_set18(porb_matrix, likelyhoods, pram["epsilon"])
        _, _, aleatoric_uncertainty = unc.uncertainty_set18(porb_matrix_f, likelyhoods_f, pram["epsilon"])
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

    elif "out.tree" == unc_method:
        porb_matrix = get_prob(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_outcome_tree(porb_matrix)
    elif "out" == unc_method:
        likelyhoods = get_likelyhood(main_model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        porb_matrix = get_prob(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_outcome(porb_matrix, likelyhoods, pram["epsilon"])
    elif "bays" == unc_method:
        # model = None
        # model = RandomForestClassifier(bootstrap=True,
        #     criterion=pram['criterion'],
        #     max_depth=pram["max_depth"],
        #     n_estimators=pram["n_estimators"],
        #     max_features= pram["max_features"],
        #     random_state=42, # seed
        #     verbose=0,
        #     warm_start=False)
        # model.fit(x_train, y_train)

        likelyhoods = get_likelyhood2(main_model, x_train, y_train, pram["laplace_smoothing"])
        # print("bays likelyhoods >>>>>>", likelyhoods)
        # accs = get_acc(main_model, x_train, y_train, pram["n_estimators"])
        porb_matrix = get_prob2(main_model, x_test, pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_bays(porb_matrix, likelyhoods)
    elif "bays2" == unc_method:
        model = None
        model = RandomForestClassifier(bootstrap=True,
            criterion=pram['criterion'],
            max_depth=pram["max_depth"],
            n_estimators=pram["n_estimators"],
            max_features= pram["max_features"],
            random_state=42, # seed
            verbose=0,
            warm_start=False)
        model.fit(x_train, y_train)
        likelyhoods = get_likelyhood(model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"])
        # print("bays likelyhoods >>>>>>", likelyhoods)
        # accs = get_acc(model, x_train, y_train, pram["n_estimators"])
        porb_matrix = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_bays2(porb_matrix, likelyhoods)
    
    elif "hyperbaysavg" == unc_method: # GH credal set from hyper forests
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

        total_list = []
        epistemic_list = []
        aliatoric_list = []

        for i, param in enumerate(params_searched): # opt_pram_list:
            if log: 
                print(f"Acc:{params_score_mean[i]:.4f} +-{params_score_std[i]:.4f} {param}")  # Eyke log
            model = None
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)

            likelyhoods = get_likelyhood2(model, x_train, y_train, pram["laplace_smoothing"])
            porb_matrix = get_prob2(model, x_test, pram["laplace_smoothing"])

            total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_ent_bays(porb_matrix, likelyhoods)

            # forest_avg
            total_list.append(total_uncertainty)
            epistemic_list.append(epistemic_uncertainty)
            aliatoric_list.append(aleatoric_uncertainty)
        
        # forest_avg
        total_uncertainty = np.array(total_list).mean(axis=0)
        epistemic_uncertainty = np.array(epistemic_list).mean(axis=0)
        aleatoric_uncertainty = np.array(aliatoric_list).mean(axis=0)

    elif "hyperbayshyper" == unc_method: # GH credal set from hyper forests
        credal_prob_matrix = []
        likelyhoods_hyper = []
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
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)

            # hyper forest
            test_prob = model.predict_proba(x_test)
            credal_prob_matrix.append(test_prob)
            train_prob = model.predict_proba(x_train)
            likelyhoods_hyper.append(log_loss(y_train,train_prob))
        
        # hyper forest
        likelyhoods_hyper = np.array(likelyhoods_hyper)
        likelyhoods_hyper = np.exp(-likelyhoods_hyper) # convert log likelihood to likelihood
        likelyhoods_hyper = likelyhoods_hyper / np.sum(likelyhoods_hyper) # normalization of the likelihood

        porb_matrix = np.array(credal_prob_matrix)
        porb_matrix = porb_matrix.transpose([1,0,2]) # convert to the format that uncertainty_set14 uses ## laplace smoothing has no effect on set20
        total_hyperforest, epistemic_hyperforest, aleatoric_hyperforest = unc.uncertainty_ent_bays(porb_matrix, likelyhoods_hyper)

        total_uncertainty = total_hyperforest
        epistemic_uncertainty = epistemic_hyperforest
        aleatoric_uncertainty = aleatoric_hyperforest
    elif "hyperbaysall" == unc_method: # GH credal set from hyper forests
        credal_prob_matrix_all = []
        likelyhoods_all = []
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
            model = RandomForestClassifier(**param,random_state=seed)
            model.fit(x_train, y_train)

            # all
            likelyhoods_all.append(get_likelyhood2(model, x_train, y_train, pram["laplace_smoothing"]))
            if credal_prob_matrix_all == []:
                credal_prob_matrix_all = get_prob(model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
            else:
                credal_prob_matrix_all = np.concatenate((credal_prob_matrix_all, get_prob2(model, x_test, pram["laplace_smoothing"])), axis=1)
        # all 
        porb_matrix_all = np.array(credal_prob_matrix_all)
        likelyhoods_all = np.array(likelyhoods_all)
        likelyhoods_all = likelyhoods_all.reshape(-1)
        likelyhoods_all = likelyhoods_all / np.sum(likelyhoods_all) # normalization of the likelihood
        total_all, epistemic_all, aleatoric_all = unc.uncertainty_ent_bays(porb_matrix_all, likelyhoods_all)

        total_uncertainty = total_all
        epistemic_uncertainty = epistemic_all
        aleatoric_uncertainty = aleatoric_all

    elif "levi3" in unc_method:
        porb_matrix = get_prob_matrix(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"])
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
                porb_matrix = get_prob(main_model, x_test, pram["n_estimators"], pram["laplace_smoothing"],a,b)
                likelyhoods = get_likelyhood(main_model, x_train, y_train, pram["n_estimators"], pram["laplace_smoothing"],a,b)

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
        porb_matrix, likelyhoods = ens_boot_likelihood(main_model, x_train, y_train, x_test, pram["n_estimators"], pram["credal_size"],pram["laplace_smoothing"])
        porb_matrix = np.array(porb_matrix)
        total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.uncertainty_gs(porb_matrix, likelyhoods, pram["credal_size"])
    elif "random" == unc_method:
        total_uncertainty = np.random.rand(len(x_test))
        epistemic_uncertainty = np.random.rand(len(x_test))
        aleatoric_uncertainty = np.random.rand(len(x_test))
    else:
        print(f"[Error] No implementation of unc_method {unc_method} for DF")

    return prediction, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty, main_model

def get_likelyhood(model_ens, x_train, y_train, n_estimators, laplace_smoothing, a=0, b=0, log=False):
    likelyhoods  = []
    for etree in range(n_estimators):
        if laplace_smoothing == 0 and a==0 and b==0:
            tree_prob_train = model_ens.estimators_[etree].predict_proba(x_train) 
        else:
            tree_prob_train = tree_laplace_corr(model_ens.estimators_[etree],x_train, laplace_smoothing, a, b)

        likelyhoods.append(log_loss(y_train,tree_prob_train))
    likelyhoods = np.array(likelyhoods)
    likelyhoods = np.exp(-likelyhoods) # convert log likelihood to likelihood
    likelyhoods = likelyhoods / np.sum(likelyhoods) # normalization of the likelihood

    if log:
        print(f"<log>----------------------------------------[{etree}]")
        print(f"likelyhoods = {likelyhoods}")
    return np.array(likelyhoods)

def get_likelyhood2(model_ens, x_train, y_train, laplace_smoothing, a=0, b=0, log=False):
    likelyhoods  = []
    for estimator in model_ens.estimators_:
        if laplace_smoothing == 0 and a==0 and b==0:
            tree_prob_train = estimator.predict_proba(x_train) 
        else:
            tree_prob_train = tree_laplace_corr(estimator,x_train, laplace_smoothing, a, b)

        likelyhoods.append(log_loss(y_train,tree_prob_train))
    likelyhoods = np.array(likelyhoods)
    likelyhoods = np.exp(-likelyhoods) # convert log likelihood to likelihood
    likelyhoods = likelyhoods / np.sum(likelyhoods) # normalization of the likelihood

    if log:
        print(f"<log>----------------------------------------[]")
        print(f"likelyhoods = {likelyhoods}")
    return np.array(likelyhoods)

def get_acc(model_ens, x_train, y_train, n_estimators, log=False):
    accs = []
    for etree in range(n_estimators):
        acc = model_ens.estimators_[etree].score(x_train, y_train)
        accs.append(acc)
    if log:
        print(f"<log>----------------------------------------[{etree}]")
        print(f"accs = {accs}")
    
    accs = np.array(accs)
    accs = accs / np.sum(accs)

    return accs

def get_prob(model_ens, x_data, n_estimators, laplace_smoothing, a=0, b=0, log=False):
    prob_matrix  = []
    for etree in range(n_estimators):
        if laplace_smoothing == 0 and a==0 and b==0:
            tree_prob = model_ens.estimators_[etree].predict_proba(x_data) 
        else:
            tree_prob = tree_laplace_corr(model_ens.estimators_[etree],x_data, laplace_smoothing,a,b)
        prob_matrix.append(tree_prob)
    if log:
        print(f"<log>----------------------------------------[{etree}]")
        print(f"prob_matrix = {prob_matrix}")
    prob_matrix = np.array(prob_matrix)
    prob_matrix = prob_matrix.transpose([1,0,2]) # D1 = data index D2= ens tree index D3= prediction prob for classes
    return prob_matrix
def get_prob2(model_ens, x_data, laplace_smoothing, a=0, b=0, log=False):
    prob_matrix  = []
    for estimator in model_ens.estimators_:
        if laplace_smoothing == 0 and a==0 and b==0:
            tree_prob = estimator.predict_proba(x_data) 
        else:
            tree_prob = tree_laplace_corr(estimator,x_data, laplace_smoothing,a,b)
        prob_matrix.append(tree_prob)
    if log:
        print(f"<log>----------------------------------------[]")
        print(f"prob_matrix = {prob_matrix}")
    prob_matrix = np.array(prob_matrix)
    prob_matrix = prob_matrix.transpose([1,0,2]) # D1 = data index D2= ens tree index D3= prediction prob for classes
    return prob_matrix

def tree_laplace_corr(tree, x_data, laplace_smoothing, a=0, b=0):
    tree_prob = tree.predict_proba(x_data)
    leaf_index_array = tree.apply(x_data)
    for data_index, leaf_index in enumerate(leaf_index_array):
        leaf_values = tree.tree_.value[leaf_index]
        leaf_samples = np.array(leaf_values).sum()
        for i,v in enumerate(leaf_values[0]):
            L = laplace_smoothing
            if a != 0 or b != 0:
                if i==0:
                    L = a
                else:
                    L = b
            # print(f"i {i} v {v} a {a} b {b} L {L} prob {(v + L) / (leaf_samples + (len(leaf_values[0]) * L))}")
            tree_prob[data_index][i] = (v + L) / (leaf_samples + (len(leaf_values[0]) * L))
    return tree_prob

def ens_boot_likelihood(ens, x_train, y_train, x_test, n_estimators, bootstrap_size, laplace_smoothing):
    # ens_prob = ens.predict_proba(x_test)
    n_estimator_index = list(range(n_estimators))

    prob_matrix = []
    likelihood_array = []
    for boot_seed in range(bootstrap_size):
        boot_index = resample(n_estimator_index, random_state=boot_seed) # bootstrap the n_estimator_index
        modle = ens
        for i, bi in enumerate(boot_index):
            modle.estimators_[i] = ens.estimators_[bi] # copying the estimator selected by the bootstrap to the new modle (new bootstraped ens)
        if laplace_smoothing == 0:
            boot_prob_train = modle.predict_proba(x_train) # get the prob predictions from the new bootstraped ens on the train data for likelihood estimation
        else:
            boot_prob_train = get_prob_matrix(modle,x_train, n_estimators, laplace_smoothing)
            boot_prob_train = np.mean(boot_prob_train, axis=1)
        likelihood_array.append(log_loss(y_train,boot_prob_train)) # calculation the likelihood of the bootstraped ens by logloss
        prob_matrix.append(modle.predict_proba(x_test)) # get the prob predictions from the new bootstraped ens on the test data for unc calculation
    prob_matrix = np.array(prob_matrix)
    likelihood_array = np.array(likelihood_array)
    # max_likelihood = np.amax(likelihood_array)
    # likelihood_array = likelihood_array / max_likelihood
    prob_matrix = prob_matrix.transpose([1,0,2]) # D1 = data index D2= ens boot index D3= prediction prob for classes
    return prob_matrix, likelihood_array


def get_prob_matrix(model, x_test, n_estimators, laplace_smoothing, log=False):
    porb_matrix = [[[] for j in range(n_estimators)] for i in range(x_test.shape[0])]
    for etree in range(n_estimators):
        # populate the porb_matrix with the tree_prob
        tree_prob = model.estimators_[etree].predict_proba(x_test)
        if laplace_smoothing > 0:
            leaf_index_array = model.estimators_[etree].apply(x_test)
            for data_index, leaf_index in enumerate(leaf_index_array):
                leaf_values = model.estimators_[etree].tree_.value[leaf_index]
                leaf_samples = np.array(leaf_values).sum()
                for i,v in enumerate(leaf_values[0]):
                    # tmp = (v + laplace_smoothing) / (leaf_samples + (len(leaf_values[0]) * laplace_smoothing))
                    # print(f">>>>>>>>>>>>>>> {tmp}  data_index {data_index} prob_index {i} v {v} leaf_samples {leaf_samples}")
                    tree_prob[data_index][i] = (v + laplace_smoothing) / (leaf_samples + (len(leaf_values[0]) * laplace_smoothing))
                # exit()

        for data_index, data_prob in enumerate(tree_prob):
            porb_matrix[data_index][etree] = list(data_prob)

        if log:
            print(f"----------------------------------------[{etree}]")
            print(f"class {model.estimators_[etree].predict(x_test)}  prob \n{tree_prob}")
    return porb_matrix

def get_count_matrix(model, x_test, n_estimators, laplace_smoothing=0):
    count_matrix = None #np.empty((len(x_test), n_estimators, 2))
    for etree in range(n_estimators):
        leaf_index_array = model.estimators_[etree].apply(x_test)
        tree_prob = model.estimators_[etree].tree_.value[leaf_index_array]
        tree_prob += laplace_smoothing 
        if etree == 0:
            count_matrix = tree_prob.copy()
        else:
            count_matrix = np.append(count_matrix , tree_prob, axis=1)

    # print(count_matrix)
    # exit()
    return count_matrix.copy()

def get_sub_count_matrix(model, x_test, n_estimators, laplace_smoothing=0):
    count_matrix = None #np.empty((len(x_test), n_estimators, 2))
    for etree in range(n_estimators):
        leaf_index_array = model.estimators_[etree].apply(x_test)
        tree_prob = model.estimators_[etree].tree_.value[leaf_index_array]
        tree_prob += laplace_smoothing 
        if etree == 0:
            count_matrix = tree_prob.copy()
        else:
            count_matrix = np.append(count_matrix , tree_prob, axis=1)

    # print(count_matrix)
    # exit()
    return count_matrix.copy()


def get_int_count_matrix(model, x_train, x_test, y_train, n_estimators, laplace_smoothing=0):
    leaf_index_matrix = []
    for etree in range(n_estimators):
        leaf_index_array_test  = model.estimators_[etree].apply(x_test) # get leaf index for each test data
        leaf_index_array_train = model.estimators_[etree].apply(x_train) # get leaf index for each train data

        tree_leaf_indexs = []
        for test_leaf_index in leaf_index_array_test:
            instances_in_leaf = np.where(leaf_index_array_train == test_leaf_index) # index of instances in the leaf that the test data is in
            tree_leaf_indexs.append(instances_in_leaf[0])
        leaf_index_matrix.append(tree_leaf_indexs)
    
    # find the intersection
    int_index_matrix = []
    for i in range(len(x_test)):
        int_i = leaf_index_matrix[0][i]
        for t in range(n_estimators):
            int_i = np.intersect1d(int_i,leaf_index_matrix[t][i])
        int_index_matrix.append(int_i)

    # get the labels
    int_labels = []
    for int_index_array in int_index_matrix:
        int_labels.append(y_train[int_index_array])
    count_matrix = np.zeros((len(x_test),len(np.unique(y_train))))
    labels = np.unique(y_train)
    for i,labs in enumerate(int_labels):
        v, c = np.unique(labs,return_counts=True)
        for j, class_value in enumerate(v):
            class_index = np.where(labels == class_value)
            count_matrix[i][class_index] = c[j]
    count_matrix = count_matrix + laplace_smoothing
    return count_matrix.copy()

def get_uni_count_matrix(model, x_train, x_test, y_train, n_estimators, laplace_smoothing=0):
    leaf_index_matrix = []
    for etree in range(n_estimators):
        leaf_index_array_test  = model.estimators_[etree].apply(x_test) # get leaf index for each test data
        leaf_index_array_train = model.estimators_[etree].apply(x_train) # get leaf index for each train data

        tree_leaf_indexs = []
        for test_leaf_index in leaf_index_array_test:
            instances_in_leaf = np.where(leaf_index_array_train == test_leaf_index) # index of instances in the leaf that the test data is in
            tree_leaf_indexs.append(instances_in_leaf[0])
        leaf_index_matrix.append(tree_leaf_indexs)
    
    # find the union
    uni_index_matrix = []
    for i in range(len(x_test)):
        uni_i = leaf_index_matrix[0][i]
        for t in range(n_estimators):
            uni_i = np.union1d(uni_i,leaf_index_matrix[t][i])
        uni_index_matrix.append(uni_i)

    # get the labels
    uni_labels = []
    for uni_index_array in uni_index_matrix:
        uni_labels.append(y_train[uni_index_array])

    # get counts of each class
    count_matrix = np.zeros((len(x_test),len(np.unique(y_train))))
    labels = np.unique(y_train)
    for i,labs in enumerate(uni_labels):
        v, c = np.unique(labs,return_counts=True)
        for j, class_value in enumerate(v):
            class_index = np.where(labels == class_value)
            count_matrix[i][class_index] = c[j]
    count_matrix = count_matrix + laplace_smoothing
    return count_matrix.copy()
