import numpy as np
from _xnes import XNES
from funcs import sphere, tablet, cigar, rosenbrock, ellipiside, ackley,rastrigin,six_hunp_camel
from omegaconf import DictConfig
from logging import getLogger
import hydra
import pickle
import datetime
import os
import shutil

@hydra.main(config_name="wnes", version_base=None, config_path="conf")
def main(cfg:DictConfig):
    logger = getLogger(__name__)
    logger.info("start!")

    func = cfg.func
    dims = cfg.dim
    mean = cfg.mean
    sigma = cfg.sigma
    eta = cfg.eta
    pop_size = cfg.pop_size
    seed_num = cfg.seed_num
    max_steps = cfg.max_steps
    update_mean = cfg.update_mean
    update_sigma = cfg.update_sigma

    current_date = datetime.datetime.now().strftime("%Y_%m_%d")
    save_date_path = "./xnes_data/"+func+"/"+current_date
    
    for dim in dims:
        save_dim_path=save_date_path+"/dim"+str(dim)+"_pop"+str(pop_size)+"_eta"+str(eta)+"_mean"+str(mean)+"_sigma"+str(sigma)+"/data"
        os.makedirs(save_dim_path, exist_ok=True)
        logger.info(f"dim:{dim}")
        for seed in range(seed_num):
            try:
                optimizer = XNES(mean=mean+np.zeros(dim),population_size=pop_size,eta=eta, sigma=sigma, seed=seed, update_mean=update_mean, update_sigma=update_sigma)
                seed_data = {}
                fitness_raw_datas = []
                mean_raw_datas=[]
                Covs_raw_datas = []
                success_gen = -1
                for generation in range(max_steps):
                    solutions = [] #solutions for one generation
                    for _ in range(optimizer.population_size):
                        # Ask a parameter
                        x = optimizer.ask()
                        if (func=="sphere"):
                            value = sphere(x)
                        elif (func=="tablet"):
                            value = tablet(x)
                        elif (func=="cigar"):
                            value = cigar(x)
                        elif (func=="ellipiside"):
                            value = ellipiside(x)
                        elif (func=="rosenbrock"):
                            value = rosenbrock(x)
                        elif (func=="ackley"):
                            value = ackley(x)
                        elif (func=="rastrigin"):
                            value = rastrigin(x)
                        else:
                            raise ValueError("cant use this func!!!")
                        solutions.append((x, value))
                        # print((x, value))
                    fitness_raw_datas.append(solutions)
                    values = [s[1] for s in solutions]
                    print(np.mean(np.array(values)))
                    mean_for_save,C = optimizer.tell(solutions)
                    if min(values)<1e-10 and success_gen==-1:
                        logger.info(f"seed:{seed} success!! gen:{generation}")
                        success_gen = generation
                    mean_raw_datas.append(mean_for_save)
                    Covs_raw_datas.append(C)
                else:
                    seed_data["fitness_raw_data"]=fitness_raw_datas
                    seed_data["mean_raw_datas"]=mean_raw_datas
                    seed_data["Covs_raw_data"]=Covs_raw_datas
                    if (success_gen==-1):
                        logger.info(f"seed{seed}:fail...couldn't find solution.")
                    else:
                        logger.info(f"seed{seed}:find solution! gen is {success_gen}")

                with open(save_dim_path+"/seed"+str(seed), mode="wb") as f:
                    pickle.dump(seed_data, f)
            except Exception as e:
                logger.info(f"error occourd.failed....: {e}")
    logger.info("end!!")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    if os.path.exists(save_dim_path+"/../output"):
            shutil.rmtree(save_dim_path+"/../output")
    shutil.copytree(hydra_cfg['runtime']['output_dir'], save_dim_path+"/../output")

if __name__ == '__main__':
    main()
