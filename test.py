run_name   = "epist_roc"
# data_list  = ["parkinsons","vertebral","breast","climate", "ionosphere", "blod", "bank", "QSAR", "spambase"] 
data_list = ["fashionMnist"]

import res_AUROC as roc

for data in data_list:
    query = f"SELECT results, id , prams, result_type FROM experiments Where task='epist' AND dataset='Jdata/{data}' AND run_name='{run_name}'"
    # query = f"SELECT results, id , prams, result_type FROM experiments Where task='unc' AND id=5604 OR id=5712 OR id=5703"

    res = roc.create_roc_table(data_list, query, "epsilon", epist_exp=True)
    print(f"-------------------------------------------------------------------------------------- {data}")
    # display(res[0].style.apply(roc.highlight_max_min))