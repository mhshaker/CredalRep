import mysql.connector as db
import os

auto_run = False

# data_names     = ["Jdata/parkinsons", "Jdata/vertebral","Jdata/ionosphere", "Jdata/climate", "Jdata/breast", "Jdata/QSAR", "Jdata/spambase"] # , "Jdata/blod" ,"Jdata/bank"
# data_names     = ["Jdata/parkinsons", "Jdata/vertebral","Jdata/ionosphere", "Jdata/climate", "Jdata/blod"] # 
data_names     = ["Jdata/parkinsons"] 
algos          = ["DF"] # ,"LR"
# modes          = ["bays", "set18", "set19", "set24", "set25", "out"]
modes          = ["set32"] # , "set14.convex"
task           = "unc"
runs           = 20
prams = {
'criterion'          : "entropy",
'max_features'       : "auto",
'max_depth'          : 10,
'n_estimators'       : 10,
'n_estimator_predict': 10,
'opt_iterations'     : 20,
# 'epsilon'            : 1,
# 'credal_size'        : 999,
'laplace_smoothing'  : 1,
'split'              : 0.30,
'run_start'          : 0,
'cv'                 : 0,
'opt_decision_model' : True
}


for algo in algos:
    for data_name in data_names:
        for mode in modes:
            run_name       = "set30_31" #f"{mode}_{algo}" + "noctua_test" # if you want a specific name give it here
            description    = "acc_hist"

            mydb = db.connect(host="131.234.250.119", user="noctua", passwd="uncertainty", database="uncertainty")
            mycursor = mydb.cursor()

            mycursor.execute("SELECT id FROM experiments ORDER BY id DESC LIMIT 1") #get last id in DB
            results = mycursor.fetchall()
            job_id = results[0][0] + 1 # set new id
            result_address = f"/home/mhshaker/Projects/Database/DB_files/job_{job_id}" # set results address
            sqlFormula = "INSERT INTO experiments (id, task, run_name, algo, prams, dataset, runs, status, description, result_type, results) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            jobs = [(job_id, task, run_name, algo, str(prams), data_name, runs, "pending", description, mode, result_address)]
            mycursor.executemany(sqlFormula, jobs) # insert new job to DB
            mydb.commit() # save


if auto_run:
    if task == "unc":
        os.system("bash /home/mhshaker/projects/uncertainty/bash/run_unc.sh")
        os.system(f"python3 /home/mhshaker/projects/uncertainty/bash/plot_accrej.py auto_unc tea {job_id}")
    elif task == "sampling":
        os.system("bash /home/mhshaker/projects/uncertainty/bash/run_sampling.sh")
        os.system(f"python3 /home/mhshaker/projects/uncertainty/bash/plot_sampling.py auto_samp {job_id}")
