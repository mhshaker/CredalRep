import sys
import mysql.connector as db
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import UncertaintyM as unc
import warnings

base_dir = os.path.dirname(os.path.realpath(__file__))
pic_dir = f"{base_dir}/pic/unc"
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

unc_value_plot = False
local = False
color_correct = False
vertical_plot = True

# data_list  = ["parkinsons","vertebral","breast","climate", "ionosphere", "blod", "bank", "QSAR", "spambase", "wine_qw"] 
# data_list = ["breast", "climate", "ionosphere", "spambase"]
data_list = ["parkinsons"]
modes     = "eat"

for data in data_list:
    
    # prameters ############################################################################################################################################

    run_name  = "wuml21_2" # "ens_size_UAI"
    plot_name = "order_comp" + data # "RensEnt-" + data 
    # query       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND dataset='Jdata/{data}' AND run_name='{run_name}'"
    query1       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND dataset='Jdata/{data}' AND run_name='{run_name}' AND result_type='bays'"
    query2       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND dataset='Jdata/{data}' AND run_name='{run_name}' AND result_type='set18'"


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
        plot_value1.append(job_plots)



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
    for job, plot_value in zip(jobs, plot_value1):
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

        for mode_index, mode in enumerate(modes):
            # print(f"mode {mode} dir {dir}")
            dir_mode = dir + "/" + mode

            legend = ""

            # get the list of file names
            file_list = []
            for (dirpath, dirnames, filenames) in os.walk(dir_mode):
                file_list.extend(filenames)

            # read every file and append all to "all_runs"
            all_runs_unc2 = []
            for f in file_list:
                run_result = np.loadtxt(dir_mode+"/"+f)
                all_runs_unc2.append(run_result)
            comp_res = unc.order_comparison(np.array(plot_value[mode_index]), np.array(all_runs_unc2))
            print("comp_res ", comp_res)
