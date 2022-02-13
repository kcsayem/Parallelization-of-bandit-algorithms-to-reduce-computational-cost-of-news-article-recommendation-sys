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
import time
import os
import pickle

SEED = 42


def hybrid_experiment(path, articles, algos, aligned_time_steps, cumulative_rewards, aligned_ctr, t, alpha):
    f = open(path, "r")
    max_ = get_num_lines(path)
    iteration = 0
    mua = make_dict_arms()
    na = make_dict_arms()
    alpha = alpha
    algos_choosing_track = [0, 0, 0]
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
                        algos_choosing_track[i] += 1
        else:
            final_arm = np.random.choice([a for a in context.keys()])
            algos_choosing_track[2] += 1
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

    logging.info(f"Thompson Sampling Chosen: {(algos_choosing_track[0] * 100) / np.sum(algos_choosing_track)}% times")
    logging.info(f"LinUCB Chosen: {(algos_choosing_track[1] * 100) / np.sum(algos_choosing_track)}% times")
    logging.info(f"Randomly Chosen: {(algos_choosing_track[2] * 100) / np.sum(algos_choosing_track)}% times")
    return algos, aligned_time_steps, cumulative_rewards, aligned_ctr, t


def sequential_experiment(algo, filename, random=False):
    articles = get_all_articles()
    f = open(filename, "r")
    # Initiate policy
    # linucb_policy_object = LinUCB(len(articles), len(articles) * 6 + 6, alpha)
    # setup arms
    algo.setup_bandits(articles)
    aligned_time_steps = 0
    cumulative_rewards = 0
    aligned_ctr = []
    random_aligned_time_steps = 0
    random_cumulative_rewards = 0
    random_aligned_ctr = []
    t = 1
    max_ = get_num_lines(filename)
    # max_ = 5000
    for line_data in tqdm(f, total=max_):
        tim, articleID, click, user_features, pool_articles = parseLine(line_data)
        # 1st column: Logged data arm.
        # Integer data type
        context = makeContext(pool_articles, user_features, articles)

        _, arm_index, random_selection = algo.pull(context)
        if arm_index == int(articleID):
            # Use reward information for the chosen arm to update
            algo.update(click, context[arm_index])

            # For CTR calculation
            aligned_time_steps += 1
            cumulative_rewards += click
            aligned_ctr.append(cumulative_rewards / aligned_time_steps)

        if random:
            if random_selection == int(articleID):
                # For CTR calculation
                random_aligned_time_steps += 1
                random_cumulative_rewards += click
                random_aligned_ctr.append(random_cumulative_rewards / random_aligned_time_steps)

        # t += 1
        # if t == max_:
        #     break

    # algo.print_bandits()

    return (aligned_time_steps, cumulative_rewards, aligned_ctr, algo, random_aligned_ctr,
            random_cumulative_rewards) if random else (aligned_time_steps, cumulative_rewards, aligned_ctr, algo)


def experiment(folder, hybrid_alpha, dir):
    articles = get_all_articles()
    v_s = [0.3]
    for v in v_s:
        v = float("{:.2f}".format(v))

        # Hybrid Experiment
        print(f"Running Hybrid Experiment")
        logging.info(f"Running Hybrid Experiment")
        logging.info(f"alpha for Hybrid: {hybrid_alpha}")
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
                algos, aligned_time_steps, cumulative_rewards, aligned_ctr, t = hybrid_experiment(
                    path, articles, algos, aligned_time_steps, cumulative_rewards, aligned_ctr, t, hybrid_alpha)
        label = f'Hybrid ({str(cumulative_rewards)})'
        plt.plot(aligned_ctr, label=label)
        logging.info(f"Total reward earned from {'Hybrid'}: {cumulative_rewards}")
        for algo in algos:
            algo.print_bandits()
        hybrid_files = [algos, aligned_time_steps, cumulative_rewards, aligned_ctr]
        hybrid_files_dir = dir + '/hybrid_files.pickle'
        with open(hybrid_files_dir, 'wb') as handle:
            pickle.dump(hybrid_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Sequential Thompson Sampling Experiment
        print(f"Running Sequential Thompson Sampling Experiment")
        logging.info(f"Running Sequential Thompson Sampling Experiment")
        TS_policy_object = ThompsonSampling(len(articles) * 6 + 6, v, 1, SEED)
        for root, dirs, files in os.walk(folder):
            for filename in files:
                path = os.path.join(root, filename)
                aligned_time_steps, cumulative_rewards, aligned_ctr, policy_object = sequential_experiment(
                    TS_policy_object, path)
        plt.plot(aligned_ctr, label=f"Seq Thompson Sampling ({str(cumulative_rewards)})")
        logging.info(f"Total reward earned from {'Sequential Thompson Sampling'}: {cumulative_rewards}")
        policy_object.print_bandits()
        ts_files = [aligned_time_steps, cumulative_rewards, aligned_ctr, policy_object]
        ts_files_dir = dir + '/ts_files.pickle'
        with open(ts_files_dir, 'wb') as handle:
            pickle.dump(ts_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Sequential LinUCB Experiment
        print(f"Running Sequential LinUCB Experiment")
        logging.info(f"Running Sequential LinUCB Experiment")
        linucb_policy_object = LinUCB(len(articles), len(articles) * 6 + 6, v)
        for root, dirs, files in os.walk(folder):
            for filename in files:
                path = os.path.join(root, filename)
                aligned_time_steps, cumulative_rewards, aligned_ctr, linucb_policy_object, random_aligned_ctr, random_cumulative_rewards = sequential_experiment(
                    linucb_policy_object, path, random=True)
        plt.plot(aligned_ctr, label=f"Seq LinUCB ({str(cumulative_rewards)})")
        plt.plot(random_aligned_ctr, label=f"Random ({str(random_cumulative_rewards)})")
        logging.info(f"Total reward earned from {'Sequential LinUCB'}: {cumulative_rewards}")
        linucb_policy_object.print_bandits()
        logging.info(f"Total reward earned from {'Random'}: {random_cumulative_rewards}")

        linucb_files = [aligned_time_steps, cumulative_rewards, aligned_ctr, linucb_policy_object, random_aligned_ctr, random_cumulative_rewards]
        linucb_files_dir = dir + '/linucb_files.pickle'
        with open(linucb_files_dir, 'wb') as handle:
            pickle.dump(linucb_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.ylabel("CTR ratio")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(f"{dir}/Results.png")


if __name__ == '__main__':

    hybrid_alpha = 0.1
    dir = 'experiment_hybrid_alpha_' + str(hybrid_alpha)
    if not os.path.exists(dir): os.mkdir(dir)

    logfile = dir + '/experiment.log'
    if os.path.isfile(logfile):
        os.remove(logfile)
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(message)s')
    experiment("data/R6A_spec", 0.1, dir)
