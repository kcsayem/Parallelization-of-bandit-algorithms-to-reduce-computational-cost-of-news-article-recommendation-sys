from thompson import ThompsonSampling
from linucb import LinUCB
from helper_functions import *
from helper_functions import print_example_banner as ptb
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import warnings
from datetime import datetime
import logging
from run_linucb import lazy_experiment as linLazy
from run_linucb import non_lazy_experiment as linNonLazy
import time
import os

SEED = 42


def sequential_experiment(path, articles, algos, aligned_time_steps, cumulative_rewards, aligned_ctr, t):
    f = open(path, "r")
    max_ = get_num_lines(path)
    iteration = 0
    mua = make_dict_arms()
    na = make_dict_arms()
    alpha = 0.05
    for line_data in tqdm(f, total=max_):
        final_arm = 0
        final_algos = []
        est_reward = -1

        # iteration+=1
        # if iteration==5000:
        #     break

        tim, articleID, click, user_features, pool_articles = parseLine(
            line_data)
        context = makeContext(pool_articles, user_features, articles)
        # print(context)
        # break
        p = np.random.choice([0, 1], 1, p=[1 - alpha, alpha])

        if p == 0:
            for i, algo in enumerate(algos):
                _, arm_index, specific_bandits = algo.pull(context)
                if mua[arm_index] >= est_reward:
                    est_reward = mua[arm_index]
                    final_arm = arm_index
                    if i not in final_algos:
                        final_algos.append(i)
        else:
            final_arm = np.random.choice([a for a in context.keys()])
        if final_arm == int(articleID):
            if p == 0:
                # Use reward information for the chosen arm to update
                algos[np.random.choice(final_algos)].update(click, context[final_arm])
            # For CTR calculation
            aligned_time_steps += 1
            cumulative_rewards += click
            mua[final_arm] = ((mua[final_arm] * na[final_arm]) + click) / (na[final_arm] + 1)
            na[final_arm] += 1
            aligned_ctr.append(cumulative_rewards / aligned_time_steps)
    f.close()
    return algos, aligned_time_steps, cumulative_rewards, aligned_ctr, t


def experiment(folder, p, lazy, savefile):
    articles = get_all_articles()
    v_s = [0.3]
    for v in v_s:
        v = float("{:.2f}".format(v))
        algos = [ThompsonSampling(len(articles) * 6 + 6, v, 1, SEED), LinUCB(len(articles), len(articles) * 6 + 6, v)]
        for algo in algos:
            algo.setup_bandits(articles)
        aligned_time_steps = 0
        cumulative_rewards = 0
        aligned_ctr = []
        t = 1
        for root, dirs, files in os.walk(folder):
            for filename in files:
                path = os.path.join(root, filename)
                algos, aligned_time_steps, cumulative_rewards, aligned_ctr, t = sequential_experiment(
                    path, articles, algos, aligned_time_steps, cumulative_rewards, aligned_ctr, t)
        label = f'Hybrid'
        plt.plot(aligned_ctr, label=label)
        logging.info(f"Total reward earned from {'Hybrid'}: {cumulative_rewards}")
        for algo in algos:
            algo.print_bandits()
    plt.ylabel("CTR ratio (Thompson Sampling)")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(f"Results/{savefile}.png")


if __name__ == '__main__':
    if os.path.isfile("experiment.log"):
        os.remove("experiment.log")
    logging.basicConfig(filename='experiment.log', level=logging.INFO, format='%(message)s')
    experiment("data/R6A_spec",-1,None,"hybrid")
