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
save_dir_path="/home/nagata/wnes/wnes_data/sphere/2024_01_26/dim2_pop100_eta0.1_mean3.0_sigma2.0_eta_update_rate0.1"#"/home/nagata/wnes/wnes_data/"+func+"/2023_11_15/dim10_pop100_eta0.1"
os.makedirs(save_dir_path+"/seeds_fig/png", exist_ok=True)
os.makedirs(save_dir_path+"/seeds_fig/eps", exist_ok=True)
os.makedirs(save_dir_path+"/seeds_fig/png_log", exist_ok=True)
os.makedirs(save_dir_path+"/seeds_fig/eps_log", exist_ok=True)

seeds = 100
is_log = False

no_data_count = 0
for i in range(seeds):
    print(i)
    # try: 
    seed_data=loadPickle(save_dir_path+"/data/seed"+str(i))
    fitness_raw_data = seed_data["fitness_raw_data"]
    # min_fitness_data = [min(gen_data,key=lambda x:x[1])[1]  for gen_data in fitness_raw_data]
    all_fitness_data = []
    for gen_data in fitness_raw_data:
        all_fitness_data.append([each_data[1] for each_data in gen_data])
    mean_fitness_data = [np.mean(np.array(gen_fitness_data))  for gen_fitness_data in all_fitness_data]

    start = 0
    end = len(mean_fitness_data)
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(10, 6))
    plt.plot(range(start, end), mean_fitness_data[start:end], label='wNES', color='dodgerblue')#green
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.subplots_adjust(left=0.13, right=0.93, top=0.9, bottom=0.1)

    plt.savefig(save_dir_path+'/seeds_fig/png/'+func+'_seed'+str(i) +'.png')
    plt.savefig(save_dir_path+'/seeds_fig/eps/'+func+'_seed'+str(i) +'.eps')
    plt.yscale('log')
    plt.savefig(save_dir_path+'/seeds_fig/png_log/'+func+'_seed'+str(i) +'_log.png')
    plt.savefig(save_dir_path+'/seeds_fig/eps_log/'+func+'_seed'+str(i) +'_log.eps')
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












