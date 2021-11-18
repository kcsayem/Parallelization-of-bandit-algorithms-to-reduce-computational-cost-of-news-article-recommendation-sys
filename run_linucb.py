from linucb import LazyLinUCB, NonLazyLinUCB
from helper_functions import *
from helper_functions import print_example_banner as ptb
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import warnings
import datetime
import logging
import warnings
warnings.filterwarnings('ignore')


def yahoo_experiment(path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                     random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t, p):
    f = open(path, "r")
    max_ = get_num_lines(path)
    # for line_data in tqdm(f, total=max_):
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
        # ptb(context)
        # break
        arm_indexes = []
        for c in range(len(contexts)):
            arm_index, specific_bandits = ts.select_arm(contexts[c])
            ts.reward_update_iteration(contexts[c][arm_index])
            arm_indexes.append(arm_index)
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
        # Doubling Round Check
        ts.doubling_round()
        ts.reward_update_batch(clicks, contexts)
        # if v == 0.01 and random_index == int(articleID):
        #     random_aligned_time_steps += 1
        #     random_cumulative_rewards += click
        #     random_aligned_ctr.append(
        #         random_cumulative_rewards / random_aligned_time_steps)
        t += 1
        # if t == 2:
        #     break

    f.close()
    return ts, aligned_time_steps, cumulative_rewards, aligned_ctr, random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t

def lazy_experiment(folder, p):
    articles = get_all_articles()
    # alphas = np.arange(0.01, 0.5, 0.1)
    alphas = [0.3]
    # v_s = alphas[:1]
    lmd = 0.2
    ps = p
    random_results = []
    plt.figure()
    for v in alphas:
        for p in ps:
            v = float("{:.2f}".format(v))
            ptb("==================================================================================")
            ptb(f"Trying alpha = {v}, lambda = {lmd}, p = {p}")
            ptb("==================================================================================")
            ts = LazyLinUCB(len(articles), len(articles) * 6 + 6,v,lmd)
            ts.setup_arms(articles)
            aligned_time_steps = 0
            random_aligned_time_steps = 0
            cumulative_rewards = 0
            random_cumulative_rewards = 0
            aligned_ctr = []
            random_aligned_ctr = []
            t = 1
            # ptb('pass')
            for root, dirs, files in os.walk(folder):
                # print('pass 1')
                for filename in files:
                    # print('pass 2')
                    path = os.path.join(root, filename)
                    # print(path)
                    ts, aligned_time_steps, cumulative_rewards, aligned_ctr, random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t = yahoo_experiment(
                        path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                        random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t,p)
                    # print('pass 3')
                    break
            plt.plot(aligned_ctr, label=f"alpha = {v}, p = {p}, lamda = {lmd}")
            if v == 0.01:
                plt.plot(random_aligned_ctr, label=f"Random CTR")
                random_results = random_cumulative_rewards
            ptb(f"total reward earned from Linucb: {cumulative_rewards}" )
            ts.printBandits()
    ptb(f"total reward earned from Random: {random_results}")
    ct = datetime.datetime.now()
    ct = ct.strftime('%Y-%m-%d-%H-%M-%S')
    plt.ylabel("CTR ratio (Parallel lazy LinUCB and Random)")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(f"Results/parallel_lazy_linucb-{ct}.png")


def non_lazy_experiment(folder, p):
    articles = get_all_articles()
    # alphas = np.arange(0.01, 0.5, 0.1)
    alphas = [0.3]
    # v_s = alphas[:1]
    lmd = 0.2
    ps = p
    random_results = []
    plt.figure()
    for v in alphas:
        for p in ps:
            v = float("{:.2f}".format(v))
            ptb("==================================================================================")
            ptb(f"Trying alpha = {v}, lambda = {lmd}, p = {p}")
            ptb("==================================================================================")
            ts = NonLazyLinUCB(len(articles), len(articles) * 6 + 6,v,lmd)
            ts.setup_arms(articles)
            aligned_time_steps = 0
            random_aligned_time_steps = 0
            cumulative_rewards = 0
            random_cumulative_rewards = 0
            aligned_ctr = []
            random_aligned_ctr = []
            t = 1
            # ptb('pass')
            for root, dirs, files in os.walk(folder):
                # ptb('pass 1')
                for filename in files:
                    # ptb('pass 2')
                    path = os.path.join(root, filename)
                    # ptb(path)
                    ts, aligned_time_steps, cumulative_rewards, aligned_ctr, random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t = yahoo_experiment(
                        path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                        random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t,p)
                    # ptb('pass 3')
                    break
            plt.plot(aligned_ctr, label=f"alpha = {v}, p = {p}, lamda = {lmd}")
            if v == 0.01:
                plt.plot(random_aligned_ctr, label=f"Random CTR")
                random_results = random_cumulative_rewards
            ptb(f"total reward earned from Linucb: {cumulative_rewards}")
            ts.printBandits()
    ptb(f"total reward earned from Random: random_results")
    plt.ylabel("CTR ratio (Parallel Non-Lazy LinUCB and Random)")
    plt.xlabel("Time")
    plt.legend()
    ct = datetime.datetime.now()
    ct = ct.strftime('%Y-%m-%d-%H-%M-%S')
    plt.savefig(f"Results/parallel_non_lazy_linucb-{str(ct)}.png")

def run():
    ct = datetime.datetime.now()
    ct = ct.strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(filename='liucb'+ ct + '.log', level=logging.INFO, format='%(message)s')
    P = [1, 5, 10, 20, 30, 50, 100, 250, 500, 1000, 2000]
    path = "data/R6A_spec"
    ptb("================================================")
    ptb("RUNNING Non-LAZY EXPERIMENTS")
    ptb("================================================")
    start = datetime.datetime.now()
    non_lazy_experiment(path, P)
    end = datetime.datetime.now()
    ptb(f"DURATION FOR Non-Lazy LinUCB: {end - start}")
    ptb("================================================")
    ptb(" ")
    ptb(' ')
    ptb("================================================")
    ptb("RUNNING LAZY EXPERIMENTS")
    ptb("================================================")
    start = datetime.datetime.now()
    lazy_experiment(path, P)
    end = datetime.datetime.now()
    ptb(f"DURATION FOR Lazy LinUCB: {end - start}")
    ptb("================================================")



if __name__ == '__main__':
    run()