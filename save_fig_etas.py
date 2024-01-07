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
save_dir_path1="/home/nagata/wnes/wnes_data/sphere/2023_11_30/dim10_pop100_eta0.1_mean3.0_sigma2.0_step0.1"#"/home/nagata/wnes/wnes_data/"+func+"/2023_11_15/dim10_pop100_eta0.1"
save_dir_path2="/home/nagata/wnes/wnes_data/sphere/2023_11_30/dim10_pop100_eta0.1_mean3.0_sigma2.0_step0.5"#"/home/nagata/wnes/wnes_data/"+func+"/2023_11_15/dim10_pop100_eta0.1"
save_dir_path3="/home/nagata/wnes/wnes_data/sphere/2023_11_30/dim10_pop100_eta0.1_mean3.0_sigma2.0_step0.9"#"/home/nagata/wnes/wnes_data/"+func+"/2023_11_15/dim10_pop100_eta0.1"
os.makedirs(save_dir_path1+"/fig", exist_ok=True)
os.makedirs(save_dir_path2+"/fig", exist_ok=True)

seeds = 100


all_mean_fitness_data = []
for i in range(seeds):
    seed_data=loadPickle(save_dir_path1+"/data/seed"+str(i))
    fitness_raw_data = seed_data["fitness_raw_data"]
    # min_fitness_data = [min(gen_data,key=lambda x:x[1])[1]  for gen_data in fitness_raw_data]
    all_fitness_data = []
    for gen_data in fitness_raw_data:
        all_fitness_data.append([each_data[1] for each_data in gen_data])
    mean_fitness_data = [np.mean(np.array(gen_fitness_data))  for gen_fitness_data in all_fitness_data]
    all_mean_fitness_data.append(mean_fitness_data)

    # print(min_fitness_data)
    # plt.plot(range(steps), min_fitness_data, label=f'Trial {i+1}')

    # Covs_raw_data = seed_data["Covs_raw_data"]
    # Cs = [data[0] for data in Covs_raw_data]
    # max_eigs = [max(np.linalg.eig(C)[0]) for C in  Cs]
    # print(max_eigs)
means = np.median(np.array(all_mean_fitness_data), axis=0)
errors = 1.96 * np.std(np.array(all_mean_fitness_data), axis=0) / np.sqrt(seeds)
steps = len(all_mean_fitness_data[0])
print(steps)

all_mean_2 = []
for i in range(seeds):
    seed_data=loadPickle(save_dir_path2+"/data/seed"+str(i))
    fitness_raw_data = seed_data["fitness_raw_data"]
    all_fitness_data = []
    for gen_data in fitness_raw_data:
        all_fitness_data.append([each_data[1] for each_data in gen_data])
    mean_fitness_data = [np.mean(np.array(gen_fitness_data))  for gen_fitness_data in all_fitness_data]
    all_mean_2.append(mean_fitness_data)
means_2 = np.median(np.array(all_mean_2), axis=0)
xnes_errors = 1.96 * np.std(np.array(all_mean_2), axis=0) / np.sqrt(seeds)

all_mean_3 = []
for i in range(seeds):
    seed_data=loadPickle(save_dir_path3+"/data/seed"+str(i))
    fitness_raw_data = seed_data["fitness_raw_data"]
    all_fitness_data = []
    for gen_data in fitness_raw_data:
        all_fitness_data.append([each_data[1] for each_data in gen_data])
    mean_fitness_data = [np.mean(np.array(gen_fitness_data))  for gen_fitness_data in all_fitness_data]
    all_mean_3.append(mean_fitness_data)
means_3 = np.median(np.array(all_mean_3), axis=0)
xnes_errors = 1.96 * np.std(np.array(all_mean_3), axis=0) / np.sqrt(seeds)

print(min(means))
print(min(means_2))
print(min(means_3))
stop_decline_point = 0
for i in range(len(means)):
    if (means[i]<means[i+1]):
        stop_decline_point=i
        break




plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10, 6))

plt.plot(range(steps), means[:steps], label='0.1', color='dodgerblue')
# plt.fill_between(range(steps), means[:steps] - errors[:steps], means[:steps] + errors[:steps], color='b', alpha=0.2)
# plt.axvline(x=stop_decline_point, linestyle='--', color='gray')
plt.plot(range(steps), means_2[:steps], label='0.5', color='salmon')
# plt.plot(range(steps), means_3[:steps], label='0.9', color='palegreen')
# plt.fill_between(range(steps), xnes_means[:steps] - xnes_errors[:steps], xnes_means[:steps] + xnes_errors[:steps], color='r', alpha=0.2)

plt.xlabel('Steps')
plt.ylabel('Value')
plt.legend()
plt.subplots_adjust(left=0.13, right=0.93, top=0.9, bottom=0.1)
plt.savefig(save_dir_path1+'/fig/'+func+'_w_x_mean.png')
plt.savefig(save_dir_path1+'/fig/'+func+'w_x_mean.eps')
print("done")









# all_data = loadPickle("/home/nagata/conjugate_nes/wnes_data/sphere/2023_11_04")
# print(len(all_data))

# for i in range(len(all_data)):
#     seed_data = all_data[i]
#     fitness_raw_data = seed_data["fitness_raw_data"]
#     min_fitness_data = [min(gen_data,key=lambda x:x[1])  for gen_data in fitness_raw_data for data in gen_data]
#     print(min_fitness_data)


# succeed_gens = []
# fail_count=0
# for i in range(len(all_data)):
#     seed_data = all_data[i]
#     if seed_data["success"]==True:
#         succeed_gens.append(seed_data["succeed_gen"])
#         print(seed_data["succeed_gen"])
#     else :
#         fail_coun+=1

# print("")
# print(min(succeed_gens))
# print(max(succeed_gens))

