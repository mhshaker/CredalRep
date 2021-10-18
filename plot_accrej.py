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
local          = False
vertical_plot  = False
single_plot    = False

color_correct  = False
job_id         = True
in_plot_legend = True
legend_flag    = False

kendalltau     = True 

data_list  = ["parkinsons","vertebral","breast","climate", "ionosphere", "QSAR", "spambase"]  #, "blod", "bank"
# data_list  = ["parkinsons","vertebral","breast","climate", "ionosphere", "blod"] 
# data_list = ["climate", "parkinsons", "spambase"]
# data_list = ["parkinsons"]
modes     = "eat"

for data in data_list:
    
    # prameters ############################################################################################################################################

    run_name   = "bays_willam"
    plot_name = data + "_bays2"
    query       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND dataset='Jdata/{data}' AND status='done' AND run_name='{run_name}'"
    # query       = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND  id=5881 OR id=5883" 

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

    xlabel      = "Rejection %"
    ylabel      = "Accuracy %"

    if single_plot:
        fig, axs = plt.subplots(1,1)
        fig.set_figheight(5)
        fig.set_figwidth(5)
    else:
        if vertical_plot:
            fig, axs = plt.subplots(len(modes),1)
            fig.set_figheight(10)
            fig.set_figwidth(5)
        else:
            fig, axs = plt.subplots(1,len(modes))
            fig.set_figheight(3)
            fig.set_figwidth(15)
    legend_list = []
    
    uni_y_range = [100, -100] # to be changed by the real min and max of the range


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
        for mode_index, mode in enumerate(modes):
            # print(f"mode {mode} dir {dir}")
            dir_p = dir + "/p"
            dir_l = dir + "/l"
            dir_mode = dir + "/" + mode

            legend = ""



            for text in job[3:]:
                legend += " " +str(text) 
            # legend += mode   

            prams = str(job[2])
            pram_name = "epsilon"
            search_pram = f"'{pram_name}': "
            v_index_s = prams.index(search_pram)
            v_index_e = prams.index(",", v_index_s)
            max_depth = prams[v_index_s+len(search_pram) : v_index_e]
            legend += " delta: " + str(max_depth)

            # prams = str(job[2])
            # pram_name = "opt_iterations"
            # search_pram = f"'{pram_name}': "
            # v_index_s = prams.index(search_pram)
            # v_index_e = prams.index(",", v_index_s)
            # max_depth = int(prams[v_index_s+len(search_pram) : v_index_e])
            # legend += " opt: " + str(max_depth)

            # prams = str(job[2])
            # pram_name = "laplace_smoothing"
            # search_pram = f"'{pram_name}': "
            # v_index_s = prams.index(search_pram)
            # v_index_e = prams.index(",", v_index_s)
            # max_depth = int(prams[v_index_s+len(search_pram) : v_index_e])
            # legend += " L " + str(max_depth)

            if job_id:
                legend += " " + str(job[1])

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

            avg_acc, avg_min, avg_max, avg_random ,steps = unc.accuracy_rejection2(all_runs_p,all_runs_l,all_runs_unc, unc_value_plot)

            # print(">>>>>>>>", avg_acc)
            linestyle = '-'
            if "set19" in legend:
                linestyle = '--'
            if "set21" in legend:
                linestyle = '--'
            if "set25" in legend:
                linestyle = '--'
            # if "out" in legend:
            #     linestyle = ':'
            if "bays" in legend:
                linestyle = ':'
            
            alpha = 0.8
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
                    color = "blue"
                if "set18" in legend:
                    color = "blue"
                if "set19" in legend:
                    color = "red"
                if "set20" in legend:
                    color = "purple"
                    alpha = 0.3
                if "set21" in legend:
                    color = "purple"
                    alpha = 0.3
                if "set24" in legend:
                    color = "red"
                if "set25" in legend:
                    color = "red"
                if "out" in legend:
                    color = "black"
            else:
                color = None

            legend = legend.replace("levi.GH.conv", "Levi-GH-conv")
            legend = legend.replace("levi.ent.conv", "Levi-Ent-conv")
            legend = legend.replace("levi.GH", "Levi-GH")
            legend = legend.replace("levi.ent", "Levi-Ent")
            legend = legend.replace("levi3.GH", "Levi-GH-MANY")
            legend = legend.replace("levi3.ent", "Levi-Ent-MANY")
            legend = legend.replace("bays", "Bayes")
            legend = legend.replace("set14", "Levi-GH-boot")
            legend = legend.replace("set15", "Levi-Ent-boot")
            legend = legend.replace("set18", "Levi-GH")
            legend = legend.replace("set19", "Levi-Ent")
            legend = legend.replace("set20", "L-C2-GH")
            legend = legend.replace("set21", "L-C2-Ent")
            legend = legend.replace("set24", "L-C3-GH")  # it is C3 but just for the WUML21 presentation
            legend = legend.replace("set25", "L-C3-Ent")
            legend = legend.replace("gs", "GS")



            avg_acc = avg_acc * 100 # to have percentates and not decimals

            if mode == "a":
                mode_title = "AU"
            if mode == "e":
                mode_title = "EU"
            if mode == "t":
                mode_title = "TU"
                if "Ent" in legend: # do not plot the S^* for set 19 21 25 as they are the same for Levi GH
                    continue
            
            if single_plot:
                legend_list.append(legend + " " + mode_title)
            elif flap:
                legend_list.append(legend)
                
            
            if single_plot:
                axs.set_title(data)
            else:
                axs[mode_index].set_title(data + " " + mode_title)
            flap =False

            if single_plot:
                axs.plot(steps, avg_acc, linestyle=linestyle, color=color)
            else:
                axs[mode_index].plot(steps, avg_acc, linestyle=linestyle, color=color, label=legend, alpha=alpha)

                y_range = axs[mode_index].get_ylim() # code to find the min and max of y axis range
                if uni_y_range[0] == 100:
                    uni_y_range[0] = y_range[0]
                if y_range[1] > uni_y_range[1]:
                    uni_y_range[1] = y_range[1]

                if in_plot_legend:
                    handles, labels = axs[mode_index].get_legend_handles_labels()
                    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
                    axs[mode_index].legend(handles, labels)

    

    if single_plot == False:
        for mode_index, mode in enumerate(modes): # uniform y axis along all plots
            axs[mode_index].axis(ymin=uni_y_range[0],ymax=uni_y_range[1])

        acc_lable_flag = True
        job_plots_list = list(axs.flat)
        if vertical_plot:
            job_plots_list = reversed(list(axs.flat))
        for ax in job_plots_list:
            if acc_lable_flag:
                ax.set(xlabel=xlabel, ylabel=ylabel)
                # ax.set(xlabel=xlabel)
                ax.set(ylabel=ylabel)
                acc_lable_flag = False
            else:
                if vertical_plot:
                    ax.set(ylabel=ylabel)
                    # pass
                else:
                    ax.set(xlabel=xlabel)
                    pass
    # title = plot_list
    # fig.suptitle(data)
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)


    if legend_flag:
        # handles, _ = fig.get_legend_handles_labels()
        # labels, handles = zip(*sorted(zip(legend_list, handles), key=lambda t: t[0]))
        # print(">>>>",labels)
        
        # fig.legend(handles, labels, bbox_to_anchor=(0.9, 1), loc='upper left') # , loc="lower center" , ncol=7
        fig.legend(labels=legend_list, bbox_to_anchor=(0.9, 1), loc='upper left') # , loc="lower center" , ncol=7

    fig.savefig(f"./pic/unc/{plot_name}.png",bbox_inches='tight')
    # fig.close()
    print(f"Plot {plot_name} Done")