from __future__ import print_function, division
from builtins import range
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, multivariate_normal
import ast
import sys
import random
import math
import pandas as pd
from tqdm import tqdm
from helper_functions import *
from scipy.sparse.linalg import cg
# from numpy.random import multivariate_normal
import cProfile
from numpy.random import default_rng
import os
from datetime import datetime
import warnings

rng = default_rng()
SEED = 42


class ThompsonSampling:
    def __init__(self, contextDimension, R, c, lmb):
        self.d = contextDimension
        self.B = np.identity(self.d) * lmb
        self.B_inv = np.linalg.inv(self.B)
        v = c  # * R * math.sqrt(24 * self.d / 0.05 * math.log(1 / 0.05))
        self.R = R
        self.c = c
        self.v_squared = v ** 2
        self.f = np.zeros(self.d)
        self.theta_hat = np.zeros(self.d)
        self.theta_estimate = rng.multivariate_normal(
            self.theta_hat, self.v_squared * self.B_inv, method='cholesky')
        self.bandits = []
        self.regrets = []

    def setUpBandits(self, articleIds):
        self.bandits = [Bandit(articleIds[k]) for k in range(len(articleIds))]

    def sample(self):
        self.theta_estimate = random_sampling(
            self.theta_hat, self.v_squared * self.B_inv, self.d, 1, SEED)
        return self.theta_estimate

    def update_iteration(self, context):
        self.B += np.outer(context, context)
        self.B_inv = inverse(self.B_inv, np.outer(context, context))

    def update_batch(self, rewards, contexts):
        rc_sum = 0
        bc_sum = 0
        for i in range(len(contexts)):
            rc_sum += contexts[i][list(contexts[i].keys())[0]] * rewards[i]
            bc_sum += np.outer(contexts[i][list(contexts[i].keys())[0]], contexts[i][list(contexts[i].keys())[0]])
        self.f += rc_sum
        self.B += bc_sum
        self.B_inv = inverse(self.B_inv, bc_sum)
        self.theta_hat = np.dot(self.B_inv, self.f)

    def pull(self, context):
        theta_estimate = self.sample()
        specific_bandits = []
        for key in context:
            for bandit in self.bandits:
                if key == bandit.index:
                    specific_bandits.append(bandit)
                    break
        max_ = -1
        max_value = float('-inf')
        for b in specific_bandits:
            cur_value = np.dot(context[b.index].T, theta_estimate)
            if max_value < cur_value:
                max_value = cur_value
                max_ = b.index
        for b in self.bandits:
            if b.index == max_:
                b.update()
                break
        estimated_reward = max_value
        return estimated_reward, max_, specific_bandits

    def getEstimate(self):
        return self.theta_estimate

    def printBandits(self):
        print("num times selected each bandit:", [b.N for b in self.bandits])

    def getRegrets(self):
        return self.regrets

    def updateV(self, t):
        v = self.c * self.R * math.sqrt(24 * self.d / 0.05 * math.log(t / 0.05))
        self.v_squared = v ** 2


class Bandit:
    def __init__(self, index):
        self.regret = 0
        self.N = 0
        self.index = index

    def update(self):
        self.N += 1


def plot(bandits, trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label=f"real p: {b.p:.4f}, win rate = {b.a - 1}/{b.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()


def yahoo_experiment(path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                     random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t, p):
    f = open(path, "r")
    max_ = get_num_lines(path)
    for iteration in tqdm(range(math.ceil(max_ / p))):
        # if iteration==900:
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


def makeContext(pool_articles, user_features, articles):
    context = {}
    for article in pool_articles:
        if len(article) == 7 and len(user_features) == 6:
            all_zeros = np.zeros(len(articles) * 6 + 6)
            for i in range(len(articles)):
                if articles[i] == int(article[0]):
                    all_zeros[i * 6:i * 6 + 6] = user_features
                    break
            all_zeros[len(articles) * 6:] = article[1:]
            context[int(article[0])] = all_zeros
    return context


def experiment(folder, p):
    articles = get_all_articles()
    v_s = [0.3]
    random_results = []
    for v in v_s:
        v = float("{:.2f}".format(v))
        ts = ThompsonSampling(len(articles) * 6 + 6, 0.0001, v, 0.2)
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
                ts, aligned_time_steps, cumulative_rewards, aligned_ctr, random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t = yahoo_experiment(
                    path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                    random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t, p)
        plt.plot(aligned_ctr, label=f"P = {p}, v = {v}")
        if v == 0.01:
            # plt.plot(random_aligned_ctr, label=f"Random CTR")
            random_results = random_cumulative_rewards
        print("total reward earned from Thompson:", cumulative_rewards)
        ts.printBandits()
    print("total reward earned from Random:", random_results)
    plt.ylabel("CTR ratio (For Thompson Sampling and Random)")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(f"figure_real_ts_random_nonlazy_{p}.png")

# if __name__ == "__main__":
#     warnings.filterwarnings('ignore')
#     np.random.seed(SEED)
#     for p in [25,50,100]:
#         start = datetime.now()
#         experiment("data/R6A_spec",p)
#         end = datetime.now()
#         print(f"Duration: {end - start}")
