import os
import sys
import time
from ast import literal_eval
import numpy as np
import Data.data_provider as dp
import Data.data_generator as dg
import Algo.a_DF as df
# import Algo.a_eRF as erf
# import Algo.a_Tree as tree
import mysql.connector as db
import sklearn
import ray
from sklearn.model_selection import KFold

from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier

@ray.remote
def uncertainty_quantification(seed, features, target, prams, mode, algo, dir, opt_decision_model=True):
    s_time = time.time()
    np.random.seed(seed)
    classes = np.unique(target)
    selected_id = np.random.choice(classes,int(len(classes)/2),replace=False) # select id classes
    selected_id_index = np.argwhere(np.isin(target,selected_id)) # get index of all id instances
    selected_ood_index = np.argwhere(np.isin(target,selected_id,invert=True)) # get index of all not selected classes (OOD)

    target_id    = target[selected_id_index].reshape(-1)
    features_id  = features[selected_id_index].reshape(-1, features.shape[1])
    target_ood   = target[selected_ood_index].reshape(-1)
    features_ood = features[selected_ood_index].reshape(-1, features.shape[1])    
    
    x_train, x_test_id, y_train, y_test_id  = dp.split_data(features_id, target_id,   split=prams["split"], seed=seed)
    _      , x_test_ood, _     , y_test_ood = dp.split_data(features_ood, target_ood, split=prams["split"], seed=seed)

    y_test = np.concatenate((np.zeros(y_test_id.shape), np.ones(y_test_ood.shape)), axis=0)
    x_test = np.concatenate((x_test_id, x_test_ood), axis=0)
    # print("------------------------------------ y_test")
    # print(y_test)
    # print(y_test.shape)

    if algo == "DF":
        predictions , t_unc, e_unc, a_unc, model = df.DF_run(x_train, x_test, y_train, y_test, prams, unc_method, seed, opt_decision_model=opt_decision_model)
        probs = model.predict_proba(x_test)
    else:
        print("[ERORR] Undefined Algo")
        exit()
        
    # print(f"run {seed} score: train {model.score(x_train, y_train):0.2f} | test {model.score(x_test, y_test):0.2f}")

    # check for directories
    prob_dir = f"{dir}/prob"
    p_dir = f"{dir}/p"
    t_dir = f"{dir}/t"
    e_dir = f"{dir}/e"
    a_dir = f"{dir}/a"
    l_dir = f"{dir}/l"
    
    if not os.path.exists(p_dir):
        os.makedirs(prob_dir)
        os.makedirs(p_dir)
        os.makedirs(t_dir)
        os.makedirs(e_dir)
        os.makedirs(a_dir)
        os.makedirs(l_dir)

    # save the results
    np.savetxt(f"{prob_dir}/{seed}.txt", probs)
    np.savetxt(f"{p_dir}/{seed}.txt", predictions.astype(int))
    np.savetxt(f"{t_dir}/{seed}.txt", t_unc)
    np.savetxt(f"{e_dir}/{seed}.txt", e_unc)
    np.savetxt(f"{a_dir}/{seed}.txt", a_unc)
    np.savetxt(f"{l_dir}/{seed}.txt", y_test.astype(int))

    e_time = time.time()
    run_time = int(e_time - s_time)

    print(f"{seed} :{run_time}s")

@ray.remote
def uncertainty_quantification_cv(seed, x_train, x_test, y_train, y_test, prams, mode, algo, dir, opt_decision_model=True):
    s_time = time.time()
    
    if algo == "DF":
        predictions , t_unc, e_unc, a_unc, model = df.DF_run(x_train, x_test, y_train, y_test, prams, unc_method, seed, opt_decision_model=opt_decision_model)
        probs = model.predict_proba(x_test)
    else:
        print("[ERORR] Undefined Algo")
        exit()
        
    # check for directories
    prob_dir = f"{dir}/prob"
    p_dir = f"{dir}/p"
    t_dir = f"{dir}/t"
    e_dir = f"{dir}/e"
    a_dir = f"{dir}/a"
    l_dir = f"{dir}/l"
    
    if not os.path.exists(p_dir):
        os.makedirs(prob_dir)
        os.makedirs(p_dir)
        os.makedirs(t_dir)
        os.makedirs(e_dir)
        os.makedirs(a_dir)
        os.makedirs(l_dir)

    # save the results
    np.savetxt(f"{prob_dir}/{seed}.txt", probs)
    np.savetxt(f"{p_dir}/{seed}.txt", predictions.astype(int))
    np.savetxt(f"{t_dir}/{seed}.txt", t_unc)
    np.savetxt(f"{e_dir}/{seed}.txt", e_unc)
    np.savetxt(f"{a_dir}/{seed}.txt", a_unc)
    np.savetxt(f"{l_dir}/{seed}.txt", y_test.astype(int))

    e_time = time.time()
    run_time = int(e_time - s_time)

    print(f"cv {seed} :{run_time}s")


if __name__ == '__main__':
    # prameter init default
    job_id = 0 # for developement
    seed   = 1
    runs = 1
    data_name = "Jdata/fashionMnist" #  
    algo = "DF"
    unc_method = "bays"
    opt_decision_model = False
    prams = {
    'criterion'          : "entropy",
    'max_features'       : "auto",
    'max_depth'          : 2,
    'n_estimators'       : 3,
    'n_estimator_predict': 3,
    'opt_iterations'     : 20,
    'epsilon'            : 1.01,
    'credal_size'        : 999,
    'laplace_smoothing'  : 1,
    'split'              : 0.30,
    'run_start'          : 0,
    'cv'                 : 0,
    'opt_decision_model' : False,
    'ood_dataset'        : "Jdata/fashionMnist",
    }

    base_dir = os.path.dirname(os.path.realpath(__file__))
    dir = f"{base_dir[:-12]}/Database/DB_files/job_{job_id}"

    # get input from command line
    if len(sys.argv) > 1:
        job_id = int(sys.argv[1])
        mydb = db.connect(host="131.234.250.119", user="noctua", passwd="uncertainty", database="uncertainty")
        mycursor = mydb.cursor()
        mycursor.execute(f"SELECT dataset, prams, results, algo, runs, result_type FROM experiments Where id ={job_id}")
        results = mycursor.fetchone()
        data_name = results[0]
        prams = literal_eval(results[1])
        dir = f"{base_dir[:-12]}/Database/DB_files/job_{job_id}"
        algo = results[3]
        runs = results[4]
        unc_method = results[5]
        mycursor.execute(f"UPDATE experiments SET results='{dir}' Where id={job_id}")
        mydb.commit()

    features, target = dp.load_data(data_name)

    print(f"job_id {job_id} start")
    start = prams["run_start"]
    ray.init()
    ray_array = []
    if prams["cv"] == 0:
        for seed in range(start,runs+start):
            ray_array.append(uncertainty_quantification.remote(seed, features, target, prams, unc_method, algo, dir, prams["opt_decision_model"]))
    else:
        cv_outer = KFold(n_splits=prams["cv"], shuffle=True, random_state=1)
        seed = start
        for train_ix, test_ix in cv_outer.split(features):
            x_train, x_test = features[train_ix, :], features[test_ix, :]
            y_train, y_test = target[train_ix], target[test_ix]
            ray_array.append(uncertainty_quantification_cv.remote(seed, x_train, x_test, y_train, y_test, prams, unc_method, algo, dir, prams["opt_decision_model"]))
            seed += 1

    res_array = ray.get(ray_array)


    if len(sys.argv) > 1:
        mycursor.execute(f"UPDATE experiments SET status='done' Where id={job_id}")
        mydb.commit()