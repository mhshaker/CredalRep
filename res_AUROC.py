import sys
import mysql.connector as db
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import UncertaintyM as unc
import warnings
import pandas as pd

base_dir = os.path.dirname(os.path.realpath(__file__))
pic_dir = f"{base_dir}/pic/unc"
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

local          = False
job_id         = True

# data_list  = ["parkinsons","vertebral","breast","climate", "ionosphere", "blod", "bank", "QSAR", "spambase"] 
# data_list = ["parkinsons","vertebral","climate", "ionosphere", "blod"]
data_list = ["parkinsons"]
modes     = "eat"

for data in data_list:
    data_df = pd.DataFrame(columns=["job_id", "Method", "Parameter", "Epistemic", "e_sd", "Aleatoric", "a_sd", "Total", "t_sd"])
    data_df.set_index('job_id', inplace=True)
    print(f"-------------------------------------------------------------------------------------- {data}")
    # prameters ############################################################################################################################################

    run_name   = "s_delta_set24"
    # query       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND dataset='Jdata/{data}' AND run_name='{run_name}'"
    query       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND id=5604 OR id=5712 OR id=5703"

    ########################################################################################################################################################
    
    # get input from command line
    mydb = db.connect(host="131.234.250.119", user="root", passwd="uncertainty", database="uncertainty")
    mycursor = mydb.cursor()
    mycursor.execute(query) # run_name ='uni_exp'
    results = list(mycursor.fetchall())
    jobs = []
    for job in results:
        jobs.append(job)

    plot_list = []

    for job in jobs:
        dir = job[0]
        if dir[0] == ".":
            dir = base_dir + dir[1:]
        if local:
            dir = f"/home/mhshaker/Projects/Database/DB_files/job_{job[1]}"
            isFile = os.path.isdir(dir)
            if not isFile:
                print("[Error] file does not exist")
                print(dir)
                exit()
        plot_list.append(job[1])

        res_dir = dir + "/res"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        job_id_df = 0
        unc_method_df = "UNKNOWN"
        param_df = 0

        method_line = ""
        for mode_index, mode in enumerate(modes):
            dir_prob = dir + "/prob"
            dir_p = dir + "/p"
            dir_l = dir + "/l"
            dir_mode = dir + "/" + mode

            legend = ""
            if method_line == "":
                if job_id:
                    method_line += " " + str(job[1])
                    job_id_df = job[1]

                for text in job[3:]:
                    method_line += " " +str(text)
                    unc_method_df = str(text)


            # get the list of file names
            file_list = []
            for (dirpath, dirnames, filenames) in os.walk(dir_mode):
                file_list.extend(filenames)


            # read every file and append all to "all_runs"
            all_runs_prob = []
            for f in file_list:
                run_result = np.loadtxt(dir_prob+"/"+f)
                all_runs_prob.append(run_result)

            all_runs_unc = []
            for f in file_list:
                run_result = np.loadtxt(dir_mode+"/"+f)
                all_runs_unc.append(run_result)

            all_runs_p = []
            for f in file_list:
                run_result = np.loadtxt(dir_p+"/"+f)
                all_runs_p.append(run_result)

            all_runs_l = []
            for f in file_list:
                run_result = np.loadtxt(dir_l+"/"+f)
                all_runs_l.append(run_result)

            AUROC_mean, AUROC_std = unc.roc(all_runs_prob, all_runs_p,all_runs_l,all_runs_unc)
            method_line += f"   {mode} {AUROC_mean:.4f} +- {AUROC_std:.2f}"
            # print(f">>>>>>>>>>> {job_id_df} mode {mode}")
            if mode== "e":
                data_df.loc[job_id_df] = {"Method":unc_method_df, "Parameter": param_df, "Epistemic":AUROC_mean, "e_sd":AUROC_std}
            elif mode== "a":
                data_df.loc[job_id_df, ["Aleatoric"]] = AUROC_mean
                data_df.loc[job_id_df, ["a_sd"]] = AUROC_std
            elif mode== "t":
                data_df.loc[job_id_df, ["Total"]] = AUROC_mean
                data_df.loc[job_id_df, ["t_sd"]] = AUROC_std
            np.savetxt(f"{res_dir}/{mode}_ROC.txt", np.array([AUROC_mean, AUROC_std]))
        # print(method_line)
    # print("------------------------------------")
    print(data_df.head())
  