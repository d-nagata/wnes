import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)

funcs = ["sphere", "ackley", "cigar", "ellipiside", "rastrigin", "rosenbrock", "tablet"]
# for func in funcs:
func = "sphere"
save_dir_path="/home/nagata/wnes/wnes_data/sphere/2024_01_26/dim2_pop100_eta0.1_mean3.0_sigma2.0_eta_update_rate0.1"#"/home/nagata/wnes/wnes_data/"+func+"/2023_11_15/dim10_pop100_eta0.1"
os.makedirs(save_dir_path+"/fig", exist_ok=True)

seeds = 100
is_log = False

no_data_count = 0
i=0
print(i)
os.makedirs(save_dir_path+"/seed"+str(i), exist_ok=True)
seed_data=loadPickle(save_dir_path+"/data/seed"+str(i))

#fitnessの情報
fitness_mins = []
fitness_maxs = []
fitness_ratios = []

fitness_raw_data = seed_data["fitness_raw_data"]
all_fitness_data = []
for gen_data_sol_and_fitness in fitness_raw_data:
    all_fitness_data.append([each_data[1] for each_data in gen_data_sol_and_fitness])
for gen_data in all_fitness_data:
    fitness_max = np.max(np.array(gen_data))
    fitness_min = np.min(np.array(gen_data))
    fitness_ratio = fitness_max/fitness_min
    
    fitness_mins.append(fitness_min)
    fitness_maxs.append(fitness_max)
    fitness_ratios.append(fitness_ratio)    

#分散共分散行列の固有値に関する計算
conv_eig_mins = []
conv_eig_maxs = []
conv_eig_ratios = []

Covs_raw_data = seed_data["Covs_raw_data"]
for conv in Covs_raw_data:
    w, _= np.linalg.eig(conv)
    cov_eig_max = np.max(w)
    cov_eig_min = np.min(w)
    cov_eig_ratio = cov_eig_max/cov_eig_min
    
    conv_eig_mins.append(cov_eig_min)
    conv_eig_maxs.append(cov_eig_max)
    conv_eig_ratios.append(cov_eig_ratio)
#平均ベクトルに関する計算
means = []
mean_abs = []
mean_raw_datas = seed_data["mean_raw_datas"]
for mean in mean_raw_datas:
    means.append(mean)
    mean_abs.append(np.linalg.norm(mean))
#G_c
G_cs = []
Covs_grad_raw_datas = seed_data["Covs_grad_raw_datas"]
for Covs_grad_raw_data in Covs_grad_raw_datas:
    G_cs.append(Covs_grad_raw_data)

#g_c
g_cs = []
Covs_nabra_raw_datas = seed_data["Covs_nabra_raw_datas"]
for Covs_nabra_raw_data in Covs_nabra_raw_datas:
    g_cs.append(Covs_nabra_raw_data)

start = 0
end = 100
csv_data = []
csv_data.append(["gen","fitness_max","fitness_min","fitness_ratio","conv_max","conv_min","conv_ratio","matrix", "mean", "mean_abs", "G_c", "g_c", "difference"])
for j in range(start, end):
    csv_data.append([j,fitness_maxs[j], fitness_mins[j], fitness_ratios[j], conv_eig_maxs[j], conv_eig_mins[j], conv_eig_ratios[j], Covs_raw_data[j], means[j], mean_abs[j], G_cs[j],g_cs[j], G_cs[j]+g_cs[j]])
with open(save_dir_path+"/seed"+str(i)+"/gen_data.csv", 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"no_data_count: {no_data_count}")
print("done")









