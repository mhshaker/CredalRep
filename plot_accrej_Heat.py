import sys
import mysql.connector as db
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import UncertaintyM as unc
import warnings
import seaborn as sns

base_dir = os.path.dirname(os.path.realpath(__file__))
pic_dir = f"{base_dir}/pic/unc"
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

unc_value_plot = False
local          = False
vertical_plot  = False
single_plot    = False

color_correct  = True
job_id         = False
in_plot_legend = True
legend_flag    = False

kendalltau     = True 

data_list  = ["parkinsons","vertebral","breast","climate", "ionosphere", "QSAR"]  #, , "spambase" "blod", "bank"
# data_list = ["spambase"]
modes     = "eat"

for data in data_list:
    
    # prameters ############################################################################################################################################

    run_name   = "heat_map"
    result_type = "set19"
    plot_name = data + "_heat_" + result_type
    query       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND dataset='Jdata/{data}' AND status='done' AND run_name='{run_name}' AND result_type='{result_type}'"
    # query       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND  id=5964" 

    ########################################################################################################################################################

    # fix dataset official name
    data = data.replace("parkinsons", "Parkinsons")
    data = data.replace("vertebral", "Vertebral Column")
    data = data.replace("breast", "Breast Cancer Wisconsin (Diagnostic)")
    data = data.replace("climate", "Climate Model Simulation Crashes")
    data = data.replace("ionosphere", "Ionosphere")
    data = data.replace("blod", "Blood Transfusion Service Center")
    data = data.replace("bank", "Banknote Authentication")
    data = data.replace("QSAR", "QSAR biodegradation")
    data = data.replace("spambase", "Spambase")
    data = data.replace("iris", "Iris")
    data = data.replace("heartdisease", "Heart Disease")

    xlabel      = "Epistemic Rejection %"
    ylabel      = "Aleatoric Rejection %"


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
        # print(">>> ",job)
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
        flap = True

        # print(f"mode {mode} dir {dir}")
        dir_p = dir + "/p"
        dir_l = dir + "/l"
        dir_e = dir + "/e"
        dir_a = dir + "/a"

        legend = ""



        for text in job[3:]:
            legend += " " +str(text) 
        # legend += mode   

        # prams = str(job[2])
        # pram_name = "epsilon"
        # search_pram = f"'{pram_name}': "
        # v_index_s = prams.index(search_pram)
        # v_index_e = prams.index(",", v_index_s)
        # max_depth = prams[v_index_s+len(search_pram) : v_index_e]
        # legend += " delta: " + str(max_depth)

        if job_id:
            legend += " " + str(job[1])

        # get the list of file names
        file_list = []
        for (dirpath, dirnames, filenames) in os.walk(dir_e):
            file_list.extend(filenames)


        # read every file and append all to "all_runs"
        all_runs_unc_epist = []
        for f in file_list:
            run_result = np.loadtxt(dir_e+"/"+f)
            all_runs_unc_epist.append(run_result)

        all_runs_unc_ale = []
        for f in file_list:
            run_result = np.loadtxt(dir_a+"/"+f)
            all_runs_unc_ale.append(run_result)

        all_runs_p = []
        for f in file_list:
            run_result = np.loadtxt(dir_p+"/"+f)
            all_runs_p.append(run_result)

        all_runs_l = []
        for f in file_list:
            run_result = np.loadtxt(dir_l+"/"+f)
            all_runs_l.append(run_result)

        heat, rej_index = unc.unc_heat_map(all_runs_p, all_runs_l, all_runs_unc_epist, all_runs_unc_ale)
        heat = heat * 100
        
        sns_plot = sns.heatmap(heat, cbar_kws={'label': 'Accuracy'}, xticklabels=rej_index[0], yticklabels=rej_index[1])
        plt.xlabel("Epistemic Rejection %")
        plt.ylabel("Aleatoric Rejection %")
        plt.locator_params(nbins=10)
        sns_plot.figure.savefig(f"./pic/unc/{plot_name}.png",bbox_inches='tight')
        plt.close()
        print(f"Plot {plot_name} Done")