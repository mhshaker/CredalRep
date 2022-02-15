import os
import sys
import time
import numpy as np
import Data.data_provider as dp
import Data.data_generator as dg
import Algo.a_DF as df
import ray
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

######################################################################### Prameters
# unc_method = ["set18", "set19", "bays"]
# unc_method = ["set30","set31","bays"]
unc_method = ["bays"]
runs = 1
n_class = 2
# data_size_list = np.arange(100, 50000, 5000)
# data_size_list = [20000, 40000, 60000, 80000, 100000]
data_size_list = np.geomspace(100, 50000, num=10, dtype=int)
uncertaintymode_list = ["Total", "Epistemic", "Aleatoric"] 
algo = "DF"
prams = {
'criterion'          : "entropy",
'max_features'       : "auto",
'max_depth'          : 10,
'n_estimators'       : 5,
'n_estimator_predict': 5,
'opt_iterations'     : 20,
'epsilon'            : 2,
'laplace_smoothing'  : 1,
'split'              : 0.30,
'opt_decision_model' : False
}

plot_data = False

#########################################################################
@ray.remote
def uncertainty_quantification(seed, x_train, x_test, y_train, y_test, prams, unc_method, algo, opt_decision_model=True):
    if algo == "DF":
        _ , t_unc, e_unc, a_unc, model = df.DF_run(x_train, x_test, y_train, y_test, prams, unc_method, seed, opt_decision_model=opt_decision_model)
    else:
        print("[ERORR] Undefined Algo")
        exit()
    t_unc_avg = np.mean(t_unc)
    e_unc_avg = np.mean(e_unc)
    a_unc_avg = np.mean(a_unc)

    return [t_unc_avg, e_unc_avg, a_unc_avg]

if __name__ == '__main__':
    
    ray.init()
    # all_res = np.zeros((3,len(unc_method),len(data_size_list),runs))
    all_res = []
    ray_res = []

    for seed in range(0,runs):

        x, y = make_classification(
                class_sep=3,
                flip_y=0.3, 
                n_samples=100000, 
                n_features=10,
                n_informative=5, 
                n_redundant=0, 
                n_repeated=0, 
                n_classes=n_class, 
                n_clusters_per_class=1, 
                weights=None, 
                hypercube=True,
                shift=0.0, 
                scale=1.0, 
                shuffle=True,
                random_state=seed)
        x_train, x_test, y_train, y_test = dp.split_data(x, y, split=prams["split"], seed=seed)

        for unc_index, unc in enumerate(unc_method):

            for i, data_size in enumerate(data_size_list):
                x_train_run = x_train[0:data_size]
                y_train_run = y_train[0:data_size]
                x_train_run, x_test, y_train_run, y_test = dp.split_data(x_train_run, y_train_run, split=prams["split"], seed=seed)

                if plot_data:
                    plt.scatter(x_train_run[:,0], x_train_run[:,1], c= y_train_run, alpha=1)
                    plt.xlim(-10, 10)
                    plt.ylim(-10, 10)
                    plt.savefig(f"./pic/s_data/dataset_{i}.png")
                    plt.close()

                ray_res.append(uncertainty_quantification.remote(seed, x_train_run, x_test, y_train_run, y_test, prams, unc, algo, prams["opt_decision_model"]))
                # print("runing")
    for res in ray_res:
        all_res.append(ray.get(res))

    all_res = np.array(all_res)
    all_res = np.reshape(all_res, (runs, len(unc_method), len(data_size_list) ,3))
    all_res = np.mean(all_res, axis=0)

    xlabel      = "Number of Samples"
    ylabel      = "Uncertainty Value"
    fig, axs = plt.subplots(1,len(uncertaintymode_list))
    fig.set_figheight(3)
    fig.set_figwidth(16)

    for mode_index, mode in enumerate(uncertaintymode_list):
        for method_index, method in enumerate(unc_method):
            unc_value = all_res[method_index, :, mode_index]
            axs[mode_index].plot(data_size_list, unc_value) # linestyle=linestyle, color=color, label=legend, alpha=alpha
            axs[mode_index].set_title(mode)
            axs[mode_index].set_xlabel(xlabel)
            axs[mode_index].set_ylabel(ylabel)
    fig.legend(unc_method)
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(f"./pic/s_data/bays_test_flip03_sep3_geo_f10.png")