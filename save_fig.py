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
save_dir_path="/home/nagata/wnes/wnes_data/sphere/2023_11_17/dim10_pop100_eta0.8"#"/home/nagata/wnes/wnes_data/"+func+"/2023_11_15/dim10_pop100_eta0.1"
xnes_save_dir_path="/home/nagata/wnes/xnes_data/sphere/2023_11_17/dim10_pop100_eta0.8"#"/home/nagata/wnes/xnes_data/"+func+"/2023_11_15/dim10_pop100_eta0.1"
os.makedirs(save_dir_path+"/fig", exist_ok=True)
os.makedirs(xnes_save_dir_path+"/fig", exist_ok=True)

seeds = 100


all_mean_fitness_data = []
for i in range(seeds):
    seed_data=loadPickle(save_dir_path+"/data/seed"+str(i))
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
means = np.mean(np.array(all_mean_fitness_data), axis=0)
errors = 1.96 * np.std(np.array(all_mean_fitness_data), axis=0) / np.sqrt(seeds)
steps = len(all_mean_fitness_data[0])

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
xnes_means = np.mean(np.array(xnes_all_mean), axis=0)
xnes_errors = 1.96 * np.std(np.array(xnes_all_mean), axis=0) / np.sqrt(seeds)


plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10, 6))

plt.plot(range(steps), means, label='wNES', color='dodgerblue')
plt.fill_between(range(steps), means - errors, means + errors, color='b', alpha=0.2)
plt.plot(range(steps), xnes_means, label='xNES', color='salmon')
plt.fill_between(range(steps), xnes_means - xnes_errors, xnes_means + xnes_errors, color='r', alpha=0.2)

plt.xlabel('Steps')
plt.ylabel('Value')
plt.legend()
plt.subplots_adjust(left=0.13, right=0.93, top=0.9, bottom=0.1)
plt.savefig(save_dir_path+'/fig/'+func+'_w_x_mean.png')
plt.savefig(save_dir_path+'/fig/'+func+'w_x_mean.eps')
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

