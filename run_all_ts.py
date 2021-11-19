from thompson import LazyThompsonSampling, NonLazyThompsonSampling, ThompsonSampling
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
from run_linucb import experiment as linSeq
import time
SEED = 42


def run(processes):
    if os.path.isfile("experiment.log"):
        os.remove("experiment.log")
    if os.path.isfile("Results/parallel_seq_linucb.png"):
        os.remove("Results/parallel_seq_linucb.png")
    if os.path.isfile("Results/parallel_seq_thompson.png"):
        os.remove("Results/parallel_seq_thompson.png")
    logging.basicConfig(filename='experiment.log', level=logging.INFO, format='%(message)s')
    warnings.filterwarnings('ignore')
    np.random.seed(SEED)
    path = "data/R6A_spec"
    ptb("RUNNING LINUCB EXPERIMENTS")
    ptb("RUNNING SEQUENTIAL EXPERIMENTS")
    print("RUNNING LINUCB EXPERIMENTS")
    print("RUNNING SEQUENTIAL EXPERIMENTS")
    linSeq(path)
    # ptb("RUNNING LAZY EXPERIMENTS")
    # start = datetime.now()
    # linLazy(path, processes)
    # end = datetime.now()
    # ptb(f"DURATION FOR P = {processes[0]}: {end - start}")
    ptb("RUNNING NON-LAZY EXPERIMENTS")
    print("RUNNING NON-LAZY EXPERIMENTS")
    start = datetime.now()
    linNonLazy(path,processes)
    end = datetime.now()
    ptb(f"DURATION FOR P = {processes[0]}: {end - start}")


    time.sleep(10)


    ptb("RUNNING THOMPSON EXPERIMENTS")
    ptb("RUNNING SEQUENTIAL EXPERIMENTS")
    print("RUNNING THOMPSON EXPERIMENTS")
    print("RUNNING SEQUENTIAL EXPERIMENTS")
    experiment_seq(path)
    # ptb("RUNNING LAZY EXPERIMENTS")
    # for p in processes:
    #     ptb(f"TRYING WITH P = {p}")
    #     start = datetime.now()
    #     experiment(path, p, True)
    #     end = datetime.now()
    #     ptb(f"DURATION FOR P = {p}: {end - start}")
    # logging.info("\n\n")
    ptb("RUNNING NON-LAZY EXPERIMENTS")
    print("RUNNING NON-LAZY EXPERIMENTS")
    for p in processes:
        ptb(f"TRYING WITH P = {p}")
        start = datetime.now()
        experiment(path, p, False)
        end = datetime.now()
        ptb(f"DURATION FOR P = {p}: {end - start}")


def parallel_experiment(path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                     random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t, p):
    f = open(path, "r")
    max_ = get_num_lines(path)
    for iteration in tqdm(range(math.ceil(max_ / p))):
        # if iteration==10:
        #     break
        lines = []
        for i in range(p):
            lines.append(f.readline())
        clicks = np.array([])
        contexts = []
        article_ids = []
        for line in lines:
            if len(line) == 0: continue
            tim, articleID, click, user_features, pool_articles = parseLine(
                line)
            context = makeContext(pool_articles, user_features, articles)
            clicks = np.append(clicks, click)
            contexts.append(context)
            article_ids.append(int(articleID))
        arm_indexes = []
        for c in range(len(contexts)):
            x, arm_index, specific_bandits = ts.pull(contexts[c])
            arm_indexes.append(arm_index)
            if type(ts) is LazyThompsonSampling:
                ts.update_iteration(contexts[c][arm_index])
        for r in range(len(contexts)):
            # random_index = np.random.choice(specific_bandits).index
            if arm_indexes[r] == article_ids[r]:
                # Use reward information for the chosen arm to update

                # For CTR calculation
                aligned_time_steps += 1
                cumulative_rewards += clicks[r]
                aligned_ctr.append(cumulative_rewards / aligned_time_steps)
            else:
                clicks[r] = 0
        #ts.doubling_round()
        ts.update_batch(clicks, contexts)
        # if v == 0.01 and random_index == int(articleID):
        #     random_aligned_time_steps += 1
        #     random_cumulative_rewards += click
        #     random_aligned_ctr.append(
        #         random_cumulative_rewards / random_aligned_time_steps)
        t += 1
        # if t == 2:
        #     break
    # ts.updateV(t)
    f.close()
    return ts, aligned_time_steps, cumulative_rewards, aligned_ctr, random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t

def sequential_experiment(path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                     random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t):
    f = open(path, "r")
    max_ = get_num_lines(path)
    iteration = 0
    for line_data in tqdm(f, total=max_):
        # iteration+=1
        # if iteration==5000:
        #     break
        tim, articleID, click, user_features, pool_articles = parseLine(
            line_data)
        context = makeContext(pool_articles, user_features, articles)
        # print(context)
        # break
        x, arm_index, specific_bandits = ts.pull(context)
        # random_index = np.random.choice(specific_bandits).index
        if arm_index == int(articleID):
            # Use reward information for the chosen arm to update
            ts.update(click, context[arm_index])
            # For CTR calculation
            aligned_time_steps += 1
            cumulative_rewards += click
            aligned_ctr.append(cumulative_rewards / aligned_time_steps)
        # if v == 0.01 and random_index == int(articleID):
        #     random_aligned_time_steps += 1
        #     random_cumulative_rewards += click
        #     random_aligned_ctr.append(
        #         random_cumulative_rewards / random_aligned_time_steps)
        # t += 1
        # if t == 2:
        #     break
        # ts.updateV(t)
    f.close()
    return ts, aligned_time_steps, cumulative_rewards, aligned_ctr, random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t


def experiment(folder, p, lazy):
    articles = get_all_articles()
    v_s = [0.3]
    random_results = []
    for v in v_s:
        v = float("{:.2f}".format(v))
        ts = LazyThompsonSampling(len(articles) * 6 + 6, 0.0001, v, 1, SEED) if lazy else NonLazyThompsonSampling(
            len(articles) * 6 + 6, 0.0001, v, 1, SEED)
        ts.setUpBandits(articles)
        aligned_time_steps = 0
        random_aligned_time_steps = 0
        cumulative_rewards = 0
        random_cumulative_rewards = 0
        aligned_ctr = []
        random_aligned_ctr = []
        t = 1
        for root, dirs, files in os.walk(folder):
            for filename in files:
                path = os.path.join(root, filename)
                ts, aligned_time_steps, cumulative_rewards, aligned_ctr, random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t = parallel_experiment(
                    path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                    random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t, p)
        plt.plot(aligned_ctr, label=f"P = {p}, {'Lazy' if lazy else 'NonLazy'}")
        # if v == 0.01:
        #     # plt.plot(random_aligned_ctr, label=f"Random CTR")
        #     random_results = random_cumulative_rewards
        logging.info(f"Total reward earned from Thompson: {cumulative_rewards}")
        ts.printBandits()
    #logging.info(f"Total reward earned from Random: {random_results}")
    plt.ylabel("CTR ratio (Thompson Sampling)")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(f"Results/parallel_seq_thompson.png")


def experiment_seq(folder):
    plt.figure()
    articles = get_all_articles()
    v_s = [0.3]
    random_results = []
    for v in v_s:
        v = float("{:.2f}".format(v))
        ts = ThompsonSampling(len(articles) * 6 + 6, 0.0001, v)
        ts.setUpBandits(articles)
        aligned_time_steps = 0
        random_aligned_time_steps = 0
        cumulative_rewards = 0
        random_cumulative_rewards = 0
        aligned_ctr = []
        random_aligned_ctr = []
        t = 1
        for root, dirs, files in os.walk(folder):
            for filename in files:
                path = os.path.join(root, filename)
                ts, aligned_time_steps, cumulative_rewards, aligned_ctr, random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t = sequential_experiment(
                    path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                    random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t)
        plt.plot(aligned_ctr, label=f"Sequential")
        # if v == 0.01:
        #     plt.plot(random_aligned_ctr, label=f"Random CTR")
        #     random_results = random_cumulative_rewards
        logging.info(f"total reward earned from Thompson: {cumulative_rewards}")
        ts.printBandits()
    logging.info(f"total reward earned from Random: {random_results}")
    plt.ylabel("CTR ratio (Thompson Sampling)")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(f"Results/parallel_seq_thompson.png")

if __name__ == "__main__":
    run([5000])
