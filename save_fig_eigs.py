import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)

funcs = ["ackley", "cigar", "ellipiside", "rastrigin", "rosenbrock","sphere",  "tablet"]
func = "tablet"
save_dir_mean_path="/home/nagata/wnes/CEC_data/2dim/wNES_mean/tablet_dim2_pop100_eta0.01_mean3.0_sigma2.0_eta_update_rate0.1_mean_eig"
save_dir_min_path="/home/nagata/wnes/CEC_data/2dim/wNES_min/tablet_dim2_pop100_eta0.01_mean3.0_sigma2.0_eta_update_rate0.1_min"#"/home/nagata/wnes/wnes_data/"+func+"/2023_11_15/dim10_pop100_eta0.1"
xnes_save_dir_path="/home/nagata/wnes/CEC_data/2dim/xNES/tablet_dim2_pop100_eta0.01_mean3.0_sigma2.0"#"/home/nagata/wnes/xnes_data/"+func+"/2023_11_15/dim10_pop100_eta0.1"
os.makedirs(save_dir_min_path+"/fig", exist_ok=True)
parts = save_dir_min_path.split("/")

seeds = 100
steps = 1000

#minn_eig
all_mean_fitness_data_eig_min = []
no_data_count = 0
for i in range(seeds):
    print(i)
    try: 
        seed_data=loadPickle(save_dir_min_path+"/data/seed"+str(i))
        fitness_raw_data = seed_data["fitness_raw_data"]
        # min_fitness_data = [min(gen_data,key=lambda x:x[1])[1]  for gen_data in fitness_raw_data]
        all_fitness_data = []
        for gen_data in fitness_raw_data:
            all_fitness_data.append([each_data[1] for each_data in gen_data])
        mean_fitness_data = [np.mean(np.array(gen_fitness_data))  for gen_fitness_data in all_fitness_data]
        all_mean_fitness_data_eig_min.append(mean_fitness_data)
    except:
        no_data_count+=1
        print(f"no_data:{i} ")
print(f"no_data_count: {no_data_count}")
means_eig_min = np.median(np.array(all_mean_fitness_data_eig_min), axis=0)
errors = 1.96 * np.std(np.array(all_mean_fitness_data_eig_min), axis=0) / np.sqrt(seeds)

#mean_eig
all_mean_fitness_data_eig_mean = []
no_data_count = 0
for i in range(seeds):
    print(i)
    try: 
        seed_data=loadPickle(save_dir_mean_path+"/data/seed"+str(i))
        fitness_raw_data = seed_data["fitness_raw_data"]
        # min_fitness_data = [min(gen_data,key=lambda x:x[1])[1]  for gen_data in fitness_raw_data]
        all_fitness_data = []
        for gen_data in fitness_raw_data:
            all_fitness_data.append([each_data[1] for each_data in gen_data])
        mean_fitness_data = [np.mean(np.array(gen_fitness_data))  for gen_fitness_data in all_fitness_data]
        all_mean_fitness_data_eig_mean.append(mean_fitness_data)
    except:
        no_data_count+=1
        print(f"no_data:{i} ")
print(f"no_data_count: {no_data_count}")
means_eig_mean = np.median(np.array(all_mean_fitness_data_eig_mean), axis=0)
errors = 1.96 * np.std(np.array(all_mean_fitness_data_eig_mean), axis=0) / np.sqrt(seeds)


#xNES
xnes_all_mean = []
for i in range(seeds):
    seed_data=loadPickle(xnes_save_dir_path+"/data/seed"+str(i))
    fitness_raw_data = seed_data["fitness_raw_data"]
    all_fitness_data = []
    for gen_data in fitness_raw_data:
        all_fitness_data.append([each_data[1] for each_data in gen_data])
    mean_fitness_data = [np.mean(np.array(gen_fitness_data))  for gen_fitness_data in all_fitness_data]
    xnes_all_mean.append(mean_fitness_data)
xnes_means = np.median(np.array(xnes_all_mean), axis=0)
xnes_errors = 1.96 * np.std(np.array(xnes_all_mean), axis=0) / np.sqrt(seeds)
print("min")
print(f"min_eig: {min(means_eig_min)}")
print(f"mean_eig: {min(means_eig_mean)}")
print(f"xNES_eig: {min(xnes_means)}")
print("last")
print(f"min_last: {means_eig_min[-1]}")
print(f"mean_last: {means_eig_mean[-1]}")
print(f"xNES_last: {xnes_means[-1]}")



plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10, 6))


plt.plot(range(steps), means_eig_min[:steps], label='wNES(eig_min)', color='dodgerblue')#green
plt.plot(range(steps), means_eig_mean[:steps], label='wNES(eig_mean)', color='green')#
# plt.fill_between(range(steps), means[:steps] - errors[:steps], means[:steps] + errors[:steps], color='b', alpha=0.2)
# plt.axvline(x=stop_decline_point, linestyle='--', color='gray')
plt.plot(range(steps), xnes_means[:steps], label='xNES', color='salmon')
# plt.fill_between(range(steps), xnes_means[:steps] - xnes_errors[:steps], xnes_means[:steps] + xnes_errors[:steps], color='r', alpha=0.2)

plt.xlabel('generation')
plt.ylabel('fitness')
plt.legend()
plt.subplots_adjust(left=0.13, right=0.93, top=0.9, bottom=0.13)

plt.savefig(save_dir_min_path+'/fig/'+func+'_w_x_mean.png')
plt.savefig(save_dir_min_path+'/fig/'+func+'w_x_mean.eps')
plt.yscale('log')
plt.savefig(save_dir_min_path+'/fig/'+func+'_w_x_mean_log.png')
plt.savefig(save_dir_min_path+'/fig/'+func+'w_x_mean_log.eps')
print("done")

