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
unc_method = ["set18", "set19", "bays"]
# unc_method = ["bays"]
runs = 10
seperation_list = np.arange(0, 5, 0.2)
uncertaintymode_list = ["Total", "Epistemic", "Aleatoric"] 
algo = "DF"
prams = {
'criterion'          : "entropy",
'max_features'       : "auto",
'max_depth'          : 10,
'n_estimators'       : 10,
'n_estimator_predict': 10,
'opt_iterations'     : 20,
'epsilon'            : 2,
'laplace_smoothing'  : 1,
'split'              : 0.30,
'opt_decision_model' : False
}

plot_data = False

#########################################################################
@ray.remote
def uncertainty_quantification(seed, features, target, prams, unc_method, algo, opt_decision_model=True):
    # s_time = time.time()
    x_train, x_test, y_train, y_test = dp.split_data(features, target, split=prams["split"], seed=seed)
    if algo == "DF":
        _ , t_unc, e_unc, a_unc, model = df.DF_run(x_train, x_test, y_train, y_test, prams, unc_method, seed, opt_decision_model=opt_decision_model)
    else:
        print("[ERORR] Undefined Algo")
        exit()
    # e_time = time.time()
    # run_time = int(e_time - s_time)
    # print(f"{seed} :{run_time}s")
    t_unc_avg = np.mean(t_unc)
    e_unc_avg = np.mean(e_unc)
    a_unc_avg = np.mean(a_unc)
    return [t_unc_avg, e_unc_avg, a_unc_avg]

if __name__ == '__main__':
    
    ray.init()
    all_res = np.zeros((3,len(unc_method),len(seperation_list)))

    for unc_index, unc in enumerate(unc_method):
        for i, sep in enumerate(seperation_list):

            test_runs = []
            for seed in range(0,runs):
                x, y = make_classification(
                        class_sep=sep,
                        flip_y=0.3, 
                        n_samples=1000, 
                        n_features=20,
                        n_informative=10, 
                        n_redundant=0, 
                        n_repeated=0, 
                        n_classes=2, 
                        n_clusters_per_class=1, 
                        weights=None, 
                        hypercube=True, 
                        shift=0.0, 
                        scale=1.0, 
                        shuffle=True,
                        random_state=seed)
                if plot_data:
                    plt.scatter(x[:,0], x[:,1], c= y, alpha=1)
                    plt.xlim(-10, 10)
                    plt.ylim(-10, 10)
                    plt.savefig(f"./pic/s_data/dataset_{i}.png")
                    plt.close()

                test_runs.append(ray.get(uncertainty_quantification.remote(seed, x, y, prams, unc, algo, prams["opt_decision_model"])))
            test_runs = np.array(test_runs)
            test_runs = test_runs.transpose()
            all_res[0][unc_index][i] = np.mean(test_runs[0,:]) # 0 is for total unc. average over the alearotic results (average of aleatoric uncertainty for the entier test dataset) for all runs
            all_res[1][unc_index][i] = np.mean(test_runs[1,:]) # 1 is for epist unc. average over the alearotic results (average of aleatoric uncertainty for the entier test dataset) for all runs
            all_res[2][unc_index][i] = np.mean(test_runs[2,:]) # 2 is for ale   unc. average over the alearotic results (average of aleatoric uncertainty for the entier test dataset) for all runs

# print(all_res)

xlabel      = "Seperation"
ylabel      = "Uncertainty Value"
fig, axs = plt.subplots(1,len(uncertaintymode_list))
fig.set_figheight(3)
fig.set_figwidth(16)

for mode_index, mode in enumerate(uncertaintymode_list):
    for method_index, method in enumerate(unc_method):
        unc_value = all_res[mode_index, method_index, :]
        # print(unc_value)
        axs[mode_index].plot(seperation_list, unc_value) # linestyle=linestyle, color=color, label=legend, alpha=alpha
        axs[mode_index].set_title(mode)
        axs[mode_index].set_xlabel(xlabel)
        axs[mode_index].set_ylabel(ylabel)
fig.legend(unc_method)
fig.subplots_adjust(bottom=0.15)
fig.savefig(f"./pic/s_data/Seperation_test_10_f20.png")