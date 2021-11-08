import sys
import mysql.connector as db
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import kendalltau
import UncertaintyM as unc
import warnings

base_dir = os.path.dirname(os.path.realpath(__file__))
pic_dir = f"{base_dir}/pic/unc"
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

job_id = True
local = False
color_correct = False
vertical_plot = True

# data_list  = ["parkinsons","vertebral","breast","climate", "ionosphere", "blod", "bank", "QSAR", "spambase", "wine_qw"] 
# data_list = ["breast", "climate", "ionosphere", "spambase"]
data_list = ["parkinsons"]
modes     = "eat"

for data in data_list:
    
    # prameters ############################################################################################################################################

    run_name  = "set31_31" # "ens_size_UAI"
    # query1       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND dataset='Jdata/{data}' AND run_name='{run_name}' AND result_type='bays'"
    # query2       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND dataset='Jdata/{data}' AND run_name='{run_name}' AND result_type='set18'"
    query1       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND  id=6101" 
    query2       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND  id=6102" 


    ########################################################################################################################################################
    plot_value1 = []
    legend_value1 = []

    # get input from command line
    mydb = db.connect(host="131.234.250.119", user="root", passwd="uncertainty", database="uncertainty")
    mycursor = mydb.cursor()
    mycursor.execute(query1) # run_name ='uni_exp'
    results = list(mycursor.fetchall())
    jobs = []
    for job in results:
        jobs.append(job)

    # print("first query ", len(jobs))
    plot_list = []
    legend_list = []

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

        job_plots = []
        job_legends = []
        for mode_index, mode in enumerate(modes):
            dir_mode = dir + "/" + mode

            legend = ""

            for text in job[3:]:
                legend += " " +str(text) 
            if job_id:
                legend += " (" + str(job[1]) + ")"

            # get the list of file names
            file_list = []
            for (dirpath, dirnames, filenames) in os.walk(dir_mode):
                file_list.extend(filenames)

            # read every file and append all to "all_runs"
            all_runs_unc1 = []
            for f in file_list:
                run_result = np.loadtxt(dir_mode+"/"+f)
                all_runs_unc1.append(run_result)
            job_plots.append(all_runs_unc1)
            job_legends.append(legend)

        plot_value1.append(job_plots)
        legend_value1.append(job_legends)



    # get input from command line
    mydb = db.connect(host="131.234.250.119", user="root", passwd="uncertainty", database="uncertainty")
    mycursor = mydb.cursor()
    mycursor.execute(query2) # run_name ='uni_exp'
    results = list(mycursor.fetchall())
    jobs = []
    for job in results:
        jobs.append(job)
    # print("second query ", len(jobs))
    plot_list = []
    legend_list = []
    for job, plot_value, legend_value in zip(jobs, plot_value1, legend_value1):
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

        kendalltau_e_t, p_et = unc.order_comparison(np.array(plot_value[0]), np.array(plot_value[2]))
        kendalltau_a_t, p_at = unc.order_comparison(np.array(plot_value[1]), np.array(plot_value[2]))
        kendalltau_e_a, p_ea = unc.order_comparison(np.array(plot_value[0]), np.array(plot_value[1]))

        print(f"{legend_value[0]} kendalltau -> e_t {kendalltau_e_t:.2f} p{p_et:.2f} | a_t {kendalltau_a_t:.2f} p{p_at:.2f} | e_a {kendalltau_e_a:.2f} p{p_ea:.2f}")

        for mode_index, mode in enumerate(modes):
            # print(f"mode {mode} dir {dir}")
            dir_mode = dir + "/" + mode

            legend = ""

            for text in job[3:]:
                legend += " " +str(text) 
            if job_id:
                legend += " (" + str(job[1]) + ")"

            # get the list of file names
            file_list = []
            for (dirpath, dirnames, filenames) in os.walk(dir_mode):
                file_list.extend(filenames)

            # read every file and append all to "all_runs"
            all_runs_unc2 = []
            for f in file_list:
                run_result = np.loadtxt(dir_mode+"/"+f)
                all_runs_unc2.append(run_result)
            comp_res, p_value = unc.order_comparison(np.array(plot_value[mode_index]), np.array(all_runs_unc2))
            print(f" {mode} {legend_value[mode_index]} to {legend} -> kendal tau {comp_res:.2f} p{p_value:.2f}")
