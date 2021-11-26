from linucb import LazyLinUCB, NonLazyLinUCB, LinUCB
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


def yahoo_experiment_non_lazy(path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                     random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t, p):
    f = open(path, "r")
    max_ = get_num_lines(path)
    # for line_data in tqdm(f, total=max_):
    for iteration in tqdm(range(math.ceil(max_ / p))):
        lines = []
        # if iteration>=100:
        #     break
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
        # print(context)
        # break
        arm_indexes = []
        for c in range(len(contexts)):
            arm_index, specific_bandits = ts.select_arm(contexts[c])
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
        ts.reward_update_batch(clicks, contexts)
        # Doubling round Check
       # ts.doubling_round()

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


def yahoo_experiment_lazy(path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                     random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t, p):
    f = open(path, "r")
    max_ = get_num_lines(path)
    # for line_data in tqdm(f, total=max_):
    for iteration in tqdm(range(math.ceil(max_ / p))):
        # if iteration==100:
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
        # ts.doubling_round()
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

def lazy_experiment(folder, p, savefile):
    articles = get_all_articles()
    # alphas = np.arange(0.01, 0.5, 0.1)
    alphas = [0.3]
    # v_s = alphas[:1]
    lmd = 1
    ps = p
    random_results = []
    # plt.figure()
    for v in alphas:
        for p in ps:
            v = float("{:.2f}".format(v))
            # ptb("==================================================================================")
            ptb(f"Trying alpha = {v}, lambda = {lmd}, p = {p}")
            # ptb("==================================================================================")
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
                    ts, aligned_time_steps, cumulative_rewards, aligned_ctr, random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t =yahoo_experiment_lazy(
                    path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                    random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t, p)
                    break
            plt.plot(aligned_ctr, label=f"P = {p}, Lazy")
            # if v == 0.01:
            #     plt.plot(random_aligned_ctr, label=f"Random CTR")
            #     random_results = random_cumulative_rewards
            ptb(f"total reward earned from Linucb: {cumulative_rewards}" )
            ts.printBandits()
    # ptb(f"total reward earned from Random: {random_results}")
    ct = datetime.datetime.now()
    ct = ct.strftime('%Y-%m-%d-%H-%M-%S')
    plt.ylabel("CTR ratio (LinUCB)")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(f"Results/{savefile}.png")


def non_lazy_experiment(folder, p,savefile):
    articles = get_all_articles()
    # alphas = np.arange(0.01, 0.5, 0.1)
    alphas = [0.3]
    # v_s = alphas[:1]
    lmd = 1
    ps = p
    random_results = []
    # plt.figure()
    for v in alphas:
        for p in ps:
            v = float("{:.2f}".format(v))
            # ptb("==================================================================================")
            ptb(f"Trying alpha = {v}, lambda = {lmd}, p = {p}")
            # ptb("==================================================================================")
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
                    ts, aligned_time_steps, cumulative_rewards, aligned_ctr, random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t = yahoo_experiment_non_lazy(
                        path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                        random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t,p)
                    # ptb('pass 3')
                    break
            label = f'P = {p}, NonLazy' if p != 1 else 'Sequential'
            plt.plot(aligned_ctr, label=label)
            # if v == 0.01:
            #     plt.plot(random_aligned_ctr, label=f"Random CTR")
            #     random_results = random_cumulative_rewards
            ptb(f"total reward earned from Linucb: {cumulative_rewards}")
            ts.printBandits()
    # ptb(f"total reward earned from Random: random_results")
    plt.ylabel("CTR ratio (LinUCB)")
    plt.xlabel("Time")
    plt.legend()
    ct = datetime.datetime.now()
    ct = ct.strftime('%Y-%m-%d-%H-%M-%S')
    plt.savefig(f"Results/{savefile}.png")

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

def sequential_experiment(path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                     random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t):
    f = open(path, "r")
    max_ = get_num_lines(path)
    iteration = 0
    for line_data in tqdm(f, total=max_):
        # iteration += 1
        # if iteration == 5000:
        #     break
        tim, articleID, click, user_features, pool_articles = parseLine(
            line_data)
        context = makeContext(pool_articles, user_features, articles)
        # print(context)
        # break
        arm_index, random_index = ts.select_arm(context)
        # random_index = np.random.choice(specific_bandits).index
        if arm_index == int(articleID):
            # Use reward information for the chosen arm to update
            ts.reward_update(click, context[arm_index])
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
def experiment(folder):
    plt.figure()
    articles = get_all_articles()
    v_s = [0.3]
    random_results = []
    for v in v_s:
        v = float("{:.2f}".format(v))
        ts = LinUCB(len(articles), len(articles) * 6 + 6,v)
        ts.setup_arms(articles)
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
        logging.info(f"total reward earned from LinUCB: {cumulative_rewards}")
        ts.printBandits()
    # print("total reward earned from Random:", random_results)
    plt.ylabel("CTR ratio (LinUCB)")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(f"Results/parallel_seq_linucb.png")

if __name__ == '__main__':
    run()