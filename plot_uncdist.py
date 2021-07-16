import sys
import mysql.connector as db
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import UncertaintyM as unc
import warnings
from sklearn import metrics

base_dir = os.path.dirname(os.path.realpath(__file__))
pic_dir = f"{base_dir}/pic/unc"
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

local = False
color_correct = True
legend_flag = False

data_list  = ["parkinsons","vertebral","breast","climate", "ionosphere", "blod", "bank", "QSAR", "spambase"] 
# data_list  = ["vertebral","breast", "ionosphere", "blod", "QSAR", "wine_qw"] 
# data_list = ["climate", "parkinsons", "spambase"]
# data_list = ["climate", "vertebral"]
# data_list = ["parkinsons"]
modes     = "eat"

for data in data_list:
    
    # prameters ############################################################################################################################################

    run_name  = "ROC_area"
    plot_name = data + "_dist"
    # query       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND dataset='Jdata/{data}' AND status='done' AND (run_name='{run_name}' AND result_type='set18' OR run_name='unc_out2' AND result_type='out')"
    query       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND dataset='Jdata/{data}' AND run_name='{run_name}'"
    # query       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND id>=4984 AND id<=4986"

    ########################################################################################################################################################

    # fix dataset official name
    # data = data.replace("parkinsons", "Parkinsons")
    # data = data.replace("vertebral", "Vertebral Column")
    # data = data.replace("breast", "Breast Cancer Wisconsin (Diagnostic)")
    # data = data.replace("climate", "Climate Model Simulation Crashes")
    # data = data.replace("ionosphere", "Ionosphere")
    # data = data.replace("blod", "Blood Transfusion Service Center")
    # data = data.replace("bank", "Banknote Authentication")
    # data = data.replace("QSAR", "QSAR biodegradation")
    # data = data.replace("spambase", "Spambase")
    # data = data.replace("iris", "Iris")
    # data = data.replace("heartdisease", "Heart Disease")

    xlabel      = "Uncertainty"
    ylabel      = "Frequency"

    



    # get input from command line
    mydb = db.connect(host="131.234.250.119", user="root", passwd="uncertainty", database="uncertainty")
    mycursor = mydb.cursor()
    mycursor.execute(query) # run_name ='uni_exp'
    results = list(mycursor.fetchall())
    jobs = []
    for job in results:
        jobs.append(job)

    fig, axs = plt.subplots(len(jobs),len(modes))
    fig.set_figheight(15)
    fig.set_figwidth(15)
    legend_list = []


    plot_list = []

    for job_index, job in enumerate(jobs):
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
        for mode_index, mode in enumerate(modes):
            # print(f"mode {mode} dir {dir}")
            dir_p = dir + "/p"
            dir_l = dir + "/l"
            dir_mode = dir + "/" + mode

            legend = ""

            for text in job[3:]:
                legend += " " +str(text) 
            # legend += mode   

            # get the list of file names
            file_list = []
            for (dirpath, dirnames, filenames) in os.walk(dir_mode):
                file_list.extend(filenames)


            # read every file and append all to "all_runs"
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

            correct, incorrect = unc.uncertainty_distribution(all_runs_p,all_runs_l,all_runs_unc)

            linestyle = '-'
            # if "set19" in legend:
            #     linestyle = ':'
            # if "set18" in legend:
            #     linestyle = ':'
            if "out" in legend:
                linestyle = '--'

            if color_correct:
                color = "black"
                if "ent" in legend:
                    color = "black"
                if "levi.GH" in legend:
                    color = "blue"
                if "levi.ent" in legend:
                    color = "red"
                if "levi3.GH" in legend:
                    color = "blue"
                if "levi3.ent" in legend:
                    color = "red"
                if "levidir.GH" in legend:
                    color = "blue"
                if "levidir.ent" in legend:
                    color = "red"
                if "gs" in legend:
                    color = "green"
                if "set14" in legend:
                    color = "blue"
                if "set15" in legend:
                    color = "red"
                if "set18" in legend:
                    color = "blue"
                if "set19" in legend:
                    color = "red"
                if "out.tree" in legend:
                    color = "yellow"
            else:
                color = None

            legend = legend.replace("set18", "Levi-GH")
            legend = legend.replace("set19", "Levi-Ent")
            legend = legend.replace("out", "Outcome")

            plot_legend = ""
            # if mode == "a":
            #     if "Levi-Ent" in legend:
            #         plot_legend = "AL"
            #     if "Levi-GH" in legend:
            #         plot_legend = "AD"
            #     if "Outcome" in legend:
            #         plot_legend = "AO"
            #     # make the exception for AU in mode==t
            # if mode == "e":
            #     if "Levi-GH" in legend:
            #         plot_legend = "EM"
            #     if "Levi-Ent" in legend:
            #         plot_legend = "EU"
            # if mode == "t":
            #     if "Levi-GH" in legend:
            #         # Exception for AU
            #         # axs[1].plot(steps, avg_acc, linestyle='--', color=color, label="AU" + f"(area {acc_rej_area:.2f})")
            #         axs[1].legend()

            #         plot_legend = "TA"
                    
            #     if "Outcome" in legend:
            #         plot_legend = "TO"
            # print(">>>>> ", plot_legend)

            # histogram with 10 bins
            n, bins, patches = axs[job_index][mode_index].hist(correct, bins=10, label="correct") # equal distance bins
            n, bins, patches = axs[job_index][mode_index].hist(incorrect, bins=10, label="incorrect") # equal distance bins
            # plt.savefig(f"./pic/unc/Hist.png",bbox_inches='tight')

            # axs[mode_index].plot(steps, avg_acc, linestyle=linestyle, color=color, label=plot_legend + f"(area {acc_rej_area:.2f})")
            axs[job_index][mode_index].legend()
            
            if mode == "a":
                mode_title = "AU"
            if mode == "e":
                mode_title = "EU"
            if mode == "t":
                mode_title = "TU"
            
            if flap:
                legend_list.append(legend)
                
            
            axs[job_index][mode_index].set_title(data + " " + legend + " " + mode_title)
            flap =False

    job_plots_list = list(axs.flat)

    for ax in job_plots_list:
        ax.set(xlabel=xlabel, ylabel=ylabel)
    
    # acc_lable_flag = True
    # for ax in job_plots_list:
    #     if acc_lable_flag:
    #         # ax.set(xlabel=xlabel, ylabel=ylabel)
    #         # ax.set(xlabel=xlabel)
    #         ax.set(ylabel=ylabel)
    #         acc_lable_flag = False

    if legend_flag:
        fig.legend(axs,labels=legend_list, loc="lower center", ncol=6)

    fig.savefig(f"./pic/unc/{plot_name}.png",bbox_inches='tight')
    # fig.close()
    print(f"Plot {plot_name} Done")