# import sys
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


def create_roc_table(data_list, query, pram_name="", modes="eat", epist_exp=False):
    df_res_list = []

    for data in data_list:
        data_df = pd.DataFrame(columns=["job_id", "Method", "Parameter", "Epistemic", "Aleatoric", "Total", "e_sd", "a_sd", "t_sd"])
        data_df.set_index('job_id', inplace=True)
        # print(f"-------------------------------------------------------------------------------------- {data}")
        
        # get input from command line
        mydb = db.connect(host="131.234.250.119", user="root", passwd="uncertainty", database="uncertainty")
        mycursor = mydb.cursor()
        mycursor.execute(query) # run_name ='uni_exp'
        results = list(mycursor.fetchall())
        jobs = []
        for job in results:
            jobs.append(job)


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

                if pram_name != "":
                    prams = str(job[2])
                    search_pram = f"'{pram_name}': "
                    v_index_s = prams.index(search_pram)
                    v_index_e = prams.index(",", v_index_s)
                    param_value = float(prams[v_index_s+len(search_pram) : v_index_e])
                    param_df = param_value


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

                if epist_exp == True:
                    AUROC_mean, AUROC_std = unc.roc_epist(all_runs_prob, all_runs_p,all_runs_l,all_runs_unc)
                    # print("Yes ", AUROC_mean)
                else:
                    AUROC_mean, AUROC_std = unc.roc(all_runs_prob, all_runs_p,all_runs_l,all_runs_unc)

                method_line += f"   {mode} {AUROC_mean:.4f} +- {AUROC_std:.2f}"

                if mode== "e":
                    data_df.loc[job_id_df] = {"Method":unc_method_df, "Parameter": param_df, "Epistemic":AUROC_mean, "e_sd":AUROC_std}
                elif mode== "a":
                    data_df.loc[job_id_df, ["Aleatoric"]] = AUROC_mean
                    data_df.loc[job_id_df, ["a_sd"]] = AUROC_std
                elif mode== "t":
                    data_df.loc[job_id_df, ["Total"]] = AUROC_mean
                    data_df.loc[job_id_df, ["t_sd"]] = AUROC_std

                np.savetxt(f"{res_dir}/{mode}_ROC.txt", np.array([AUROC_mean, AUROC_std]))
        data_df["Aleatoric"] = pd.to_numeric(data_df["Aleatoric"], downcast="float")
        data_df["a_sd"] = pd.to_numeric(data_df["a_sd"], downcast="float")
        data_df["Total"] = pd.to_numeric(data_df["Total"], downcast="float")
        data_df["t_sd"] = pd.to_numeric(data_df["t_sd"], downcast="float")
        data_df = data_df[["Method", "Parameter", "Epistemic", "Aleatoric", "Total", "e_sd", "a_sd", "t_sd"]]
        df_res_list.append(data_df)
    return df_res_list



def highlight_max_min(s): # style for the dataframe
    if s.dtype == object or s.name == "Parameter":
        is_max = [False for _ in range(s.shape[0])]
    else:
        if s.name == "e_sd" or s.name == "a_sd" or s.name == "t_sd":
            is_max = s == s.min()
        else:
            is_max = s == s.max()
    # print(is_max)
    return ['background: gray' if cell else '' for cell in is_max]
