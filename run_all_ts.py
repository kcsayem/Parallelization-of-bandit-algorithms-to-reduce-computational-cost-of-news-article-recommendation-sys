from thompson import LazyThompsonSampling, NonLazyThompsonSampling
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

SEED = 42


def run_seq_par(processes, filename_ts, filename_ln):
    if os.path.isfile(f"Results/{filename_ts}.png"):
        os.remove(f"Results/{filename_ts}.png")
    if os.path.isfile(f"Results/{filename_ln}.png"):
        os.remove(f"Results/{filename_ln}.png")
    path = "data/R6A_spec"
    ptb("RUNNING LINUCB EXPERIMENTS")
    ptb("RUNNING SEQUENTIAL EXPERIMENTS")
    start = datetime.now()
    linNonLazy(path, [processes[0]], filename_ln)
    end = datetime.now()
    ptb(f"DURATION FOR P = {processes[0]}: {end - start}")
    ptb("RUNNING NONLAZY EXPERIMENTS")
    start = datetime.now()
    linNonLazy(path, [processes[1]], filename_ln)
    end = datetime.now()
    ptb(f"DURATION FOR P = {processes[1]}: {end - start}")

    time.sleep(10)
    plt.figure()

    ptb("RUNNING THOMPSON EXPERIMENTS")
    ptb("RUNNING SEQUENTIAL EXPERIMENTS")
    start = datetime.now()
    experiment(path, processes[0], False, filename_ts)
    end = datetime.now()
    ptb(f"DURATION FOR P = {processes[0]}: {end - start}")

    ptb("RUNNING NONLAZY EXPERIMENTS")
    start = datetime.now()
    experiment(path, processes[1], False, filename_ts)
    end = datetime.now()
    ptb(f"DURATION FOR P = {processes[1]}: {end - start}")


def run_par_par(processes, filename_ts, filename_ln):
    if os.path.isfile(f"Results/{filename_ts}.png"):
        os.remove(f"Results/{filename_ts}.png")
    if os.path.isfile(f"Results/{filename_ln}.png"):
        os.remove(f"Results/{filename_ln}")
    ptb("RUNNING LINUCB EXPERIMENTS")
    ptb("RUNNING LAZY EXPERIMENTS")
    path = "data/R6A_spec"
    start = datetime.now()
    linLazy(path, [processes[0]], filename_ln)
    end = datetime.now()
    ptb(f"DURATION FOR P = {processes[0]}: {end - start}")
    ptb("RUNNING NONLAZY EXPERIMENTS")
    start = datetime.now()
    linNonLazy(path, [processes[0]], filename_ln)
    end = datetime.now()
    ptb(f"DURATION FOR P = {processes[0]}: {end - start}")

    plt.figure()
    time.sleep(10)

    ptb("RUNNING THOMPSON EXPERIMENTS")
    ptb("RUNNING SEQUENTIAL EXPERIMENTS")
    start = datetime.now()
    experiment(path, processes[0], True, filename_ts)
    end = datetime.now()
    ptb(f"DURATION FOR P = {processes[0]}: {end - start}")
    ptb("RUNNING NONLAZY EXPERIMENTS")
    start = datetime.now()
    experiment(path, processes[0], False, filename_ts)
    end = datetime.now()
    ptb(f"DURATION FOR P = {processes[0]}: {end - start}")


def parallel_experiment(path, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                        random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t, p):
    f = open(path, "r")
    max_ = get_num_lines(path)
    for iteration in tqdm(range(math.ceil(max_ / p))):
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
            if arm_indexes[r] == article_ids[r]:
                aligned_time_steps += 1
                cumulative_rewards += clicks[r]
                aligned_ctr.append(cumulative_rewards / aligned_time_steps)
            else:
                clicks[r] = 0
        ts.update_batch(clicks, contexts)
        t += 1
    f.close()
    return ts, aligned_time_steps, cumulative_rewards, aligned_ctr, random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t


def experiment(folder, p, lazy, savefile):
    articles = get_all_articles()
    v_s = [0.3]
    for v in v_s:
        v = float("{:.2f}".format(v))
        ts = LazyThompsonSampling(len(articles) * 6 + 6, v, 1, SEED) if lazy else NonLazyThompsonSampling(
            len(articles) * 6 + 6, v, 1, SEED)
        ts.setup_bandits(articles)
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
                    path, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                    random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t, p)
        lazy_nonlazy = 'Lazy' if lazy else 'NonLazy'
        label = f'P = {p}, {lazy_nonlazy}' if p != 1 else 'Sequential'
        plt.plot(aligned_ctr, label=label)
        logging.info(f"Total reward earned from Thompson: {cumulative_rewards}")
        ts.print_bandits()
    plt.ylabel("CTR ratio (Thompson Sampling)")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(f"Results/{savefile}.png")


if __name__ == "__main__":
    if os.path.isfile("experiment.log"):
        os.remove("experiment.log")
    logging.basicConfig(filename='experiment.log', level=logging.INFO, format='%(message)s')
    warnings.filterwarnings('ignore')
    np.random.seed(SEED)
    run_seq_par([1, 1000], "seq_par_thompson", "seq_par_linucb")
    plt.figure()
    run_par_par([1000], "par_par_thompson", "par_par_linucb")
