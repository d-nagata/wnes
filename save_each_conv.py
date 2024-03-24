import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)

funcs = ["sphere", "ackley", "cigar", "ellipiside", "rastrigin", "rosenbrock", "tablet"]
# for func in funcs:
func = "sphere"
save_dir_path="/home/nagata/wnes/wnes_data/sphere/2024_01_24/dim2_pop100_eta0.1_mean0.0_sigma1.0_eta_update_rate0.1"#"/home/nagata/wnes/wnes_data/"+func+"/2023_11_15/dim10_pop100_eta0.1"
os.makedirs(save_dir_path+"/cov_fig/png", exist_ok=True)
os.makedirs(save_dir_path+"/cov_fig/eps", exist_ok=True)
os.makedirs(save_dir_path+"/cov_fig/png_log", exist_ok=True)
os.makedirs(save_dir_path+"/cov_fig/eps_log", exist_ok=True)

seeds = 100
is_log = False

no_data_count = 0
for i in range(seeds):
    print(i)
    # try: 
    conv_eig_mins = []
    conv_eig_maxs = []
    conv_eig_ratios = []
    seed_data=loadPickle(save_dir_path+"/data/seed"+str(i))
    Covs_raw_data = seed_data["Covs_raw_data"]
    for conv in Covs_raw_data:
        w, _= np.linalg.eig(conv)
        cov_eig_max = np.max(w)
        cov_eig_min = np.min(w)
        cov_eig_ratio = cov_eig_max/cov_eig_min
        
        conv_eig_mins.append(cov_eig_min)
        conv_eig_maxs.append(cov_eig_max)
        conv_eig_ratios.append(cov_eig_ratio)

    start=0
    end=100#len(conv_eig_mins)
    
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(10, 6))
    plt.plot(range(start, end), conv_eig_mins[start:end], label='min', color='dodgerblue')#green
    plt.plot(range(start, end), conv_eig_maxs[start:end], label='max', color='green')#green
    plt.plot(range(start, end), conv_eig_ratios[start:end], label='ratio', color='red')#green

    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.subplots_adjust(left=0.13, right=0.93, top=0.9, bottom=0.1)

    plt.savefig(save_dir_path+'/cov_fig/png/'+func+'_seed'+str(i) +'.png')
    plt.savefig(save_dir_path+'/cov_fig/eps/'+func+'_seed'+str(i) +'.eps')
    plt.yscale('log')
    plt.savefig(save_dir_path+'/cov_fig/png_log/'+func+'_seed'+str(i) +'_log.png')
    plt.savefig(save_dir_path+'/cov_fig/eps_log/'+func+'_seed'+str(i) +'_log.eps')


    # except:
    #     no_data_count+=1
    #     print(f"no_data:{i} ")
print(f"no_data_count: {no_data_count}")
print("done")






# plt.plot(range(steps), means[:steps], label='wxNES', color='green')#
# plt.fill_between(range(steps), means[:steps] - errors[:steps], means[:steps] + errors[:steps], color='b', alpha=0.2)
# plt.axvline(x=stop_decline_point, linestyle='--', color='gray')
# plt.plot(range(steps), xnes_means[:steps], label='xNES', color='salmon')
# plt.fill_between(range(steps), xnes_means[:steps] - xnes_errors[:steps], xnes_means[:steps] + xnes_errors[:steps], color='r', alpha=0.2)












