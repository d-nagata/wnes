import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)

save_dir_path="/home/nagata/wnes/wnes_data/sphere/2023_11_30/dim2_pop100_eta0.1_mean0.0_sigma2.0"
os.makedirs(save_dir_path+"/fig", exist_ok=True)



all_mean_fitness_data = []
# for seed in range(100):
seed_data=loadPickle(save_dir_path+"/data/seed"+str(11))
#check fitness, for check when fail to count
fitness_raw_data = seed_data["fitness_raw_data"]#gens, pop_size: (array, value)
all_fitness_data = []#gens, pop_size: value
for gen_data in fitness_raw_data:
    all_fitness_data.append([each_data[1] for each_data in gen_data])
mean_fitness_data = [np.mean(np.array(gen_fitness_data))  for gen_fitness_data in all_fitness_data]
    # print(f"#{seed}: {mean_fitness_data[-1]}")

for i,gen_mean_data in enumerate(mean_fitness_data):
    print(f"#{i}: {gen_mean_data}")

Covs_raw_data = seed_data["Covs_raw_data"]#gens: cov
for i,cov in enumerate(Covs_raw_data):
    if i<300:
        print(f"#{i}: {cov}")

Covs_grad_raw_datas = seed_data["Covs_grad_raw_datas"]
calc_Cov = Covs_raw_data[0]
for i,cov_grad in enumerate(Covs_grad_raw_datas):
    if i>0:
        calc_Cov+=cov_grad
        print(f"#{i}: {calc_Cov}")
        




# Covs_raw_data = seed_data["Covs_raw_data"]

# #grad
# Covs_grad_raw_datas = seed_data["Covs_grad_raw_datas"]
# grad_left_tops = []
# grad_right_tops = []
# grad_left_bottoms = []
# grad_right_bottoms = []
# for j, Cov in enumerate(Covs_grad_raw_datas):
#     grad_left_tops.append(Cov[0][0])
#     grad_right_tops.append(Cov[0][1])
#     grad_left_bottoms.append(Cov[1][0])
#     grad_right_bottoms.append(Cov[1][1])


# steps = 60
# plt.figure(figsize=(10, 6))

# plt.plot(range(steps), grad_left_tops[:steps], label='grad_left_tops', color='b')
# plt.plot(range(steps), grad_right_bottoms[:steps], label='grad_right_bottoms', color='k')
# plt.axhline(0, color='gray', linestyle='--')

# plt.title('seed0 Covs_grad')
# plt.xlabel('Steps')
# plt.ylabel('Value')
# plt.legend()
# plt.savefig(save_dir_path+'/fig/cov_grad_'+str(steps)+'.png')



# left_tops = []
# right_tops = []
# left_bottoms = []
# right_bottoms = []
# for j, Cov in enumerate(Covs_raw_data):
#     left_tops.append(Cov[0][0])
#     right_tops.append(Cov[0][1])
#     left_bottoms.append(Cov[1][0])
#     right_bottoms.append(Cov[1][1])

# steps = 20
# plt.figure(figsize=(10, 6))

# plt.plot(range(steps), left_tops[:steps], label='left_tops', color='b')
# plt.plot(range(steps), right_tops[:steps], label='right_tops', color='g')
# plt.plot(range(steps), left_bottoms[:steps], label='left_bottoms', color='r')
# plt.plot(range(steps), right_bottoms[:steps], label='right_bottoms', color='k')

# plt.title('seed0 Covs')
# plt.xlabel('Steps')
# plt.ylabel('Value')
# plt.legend()
# plt.savefig(save_dir_path+'/fig/cov_'+str(steps)+'.png')

#save grad vs cov
# steps = 20
# plt.figure(figsize=(10, 6))
# plt.plot(range(steps), left_tops[:steps], label='left_tops', color='b')
# plt.plot(range(steps), grad_left_tops[:steps], label='grad_left_tops', color='r')

# plt.title('seed0 grad vs cov')
# plt.xlabel('Steps')
# plt.ylabel('Value')
# plt.legend()
# plt.savefig(save_dir_path+'/fig/cov_vs_grad_'+str(steps)+'.png')