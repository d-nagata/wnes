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
save_dir_path="/home/nagata/wnes/wnes_data/sphere/2024_01_19/dim10_pop100_eta0.01_mean3.0_sigma2.0_eta_update_rate0.1"#"/home/nagata/wnes/wnes_data/"+func+"/2023_11_15/dim10_pop100_eta0.1"
os.makedirs(save_dir_path+"/fig", exist_ok=True)

seeds = 100
is_log = False

no_data_count = 0
for i in range(seeds):
    if i==1:
        print(i)
        # try: 
        seed_data=loadPickle(save_dir_path+"/data/seed"+str(i))
        fitness_raw_data = seed_data["fitness_raw_data"]
        # min_fitness_data = [min(gen_data,key=lambda x:x[1])[1]  for gen_data in fitness_raw_data]
        all_fitness_data = []
        for gen_data in fitness_raw_data:
            all_fitness_data.append([each_data[1] for each_data in gen_data])
        mean_fitness_data = [np.mean(np.array(gen_fitness_data))  for gen_fitness_data in all_fitness_data]
        print(f"min: {np.min(mean_fitness_data)}")
        for j, fitness in enumerate(mean_fitness_data):
            print(f"{j} : {fitness}")

        fitness_mins = []
        fitness_maxs = []
        fitness_ratios = []
        seed_data=loadPickle(save_dir_path+"/data/seed"+str(i))
        # Covs_raw_data = seed_data["Covs_raw_data"]
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
        

        start = 200
        end = 480

        plt.rcParams.update({'font.size': 18})
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # ax1.plot(range(start, end), mean_fitness_data[start:end], label='fitness', color='dodgerblue')#green
        # ax1.set_xlabel('Steps')
        # ax1.set_ylabel('Value')
        # plt.legend()
        # ax1.set_yscale('log')

        ax2 = ax1.twinx()
        ax2.plot(range(start, end), fitness_mins[start:end], label='fitness_min', color='red')
        ax2.plot(range(start, end), fitness_maxs[start:end], label='fitness_max', color='green')
        ax2.plot(range(start, end), fitness_ratios[start:end], label='fitness_ratio', color='magenta')
        ax2.set_yscale('log')
        ax2.set_ylabel('outlier')
        plt.legend(loc='upper left')

        save_message = "check_outlier_fitness"
        plt.subplots_adjust(left=0.13, right=0.93, top=0.9, bottom=0.1)
        plt.savefig(save_dir_path+'/seed1/png_log/'+save_message +'_log.png')
        plt.savefig(save_dir_path+'/seed1/eps_log/'+save_message +'_log.eps')

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












