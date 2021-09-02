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

NUM_TRIALS = 2000

rng = default_rng()


class ThompsonSampling:
    def __init__(self, contextDimension, R, c):
        self.d = contextDimension
        self.B_inv = np.linalg.inv(np.identity(self.d))
        v = c * R * math.sqrt(24 * self.d / 0.05 * math.log(1 / 0.05))
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
        self.theta_estimate = rng.multivariate_normal(
            self.theta_hat, self.v_squared * self.B_inv, method='cholesky')
        return self.theta_estimate

    def update(self, reward, context):
        self.f += context * reward
        self.B_inv = inverse(self.B_inv, np.outer(context, context))
        self.theta_hat = np.dot(self.B_inv, self.f)

    def pull(self, context):
        theta_estimate = self.sample()
        specific_bandits = []
        for key in context:
            for bandit in self.bandits:
                if key == bandit.index:
                    specific_bandits.append(bandit)
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

    def plot_regret(self, figsize=[12, 6]):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        pd.Series(np.cumsum(self.regrets)).plot(ax=axes[0])
        axes[0].set_title(f"Cummulative Regret for {NUM_TRIALS} iteration")
        pd.Series(np.hstack(self.regrets).reshape(-1)).plot(ax=axes[1])
        axes[1].set_title(f"Individual Regret for {NUM_TRIALS} iteration")
        plt.suptitle(f"Regret for Thompson Sampling")
        plt.show()


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
                     random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t):
    f = open(path, "r")
    max_ = get_num_lines(path)
    for line_data in tqdm(f, total=max_):
        tim, articleID, click, user_features, pool_articles = parseLine(
            line_data)
        context = makeContext(pool_articles, user_features, articles)
        # print(context)
        # break
        x, arm_index, specific_bandits = ts.pull(context)
        random_index = np.random.choice(specific_bandits).index
        if arm_index == int(articleID):
            # Use reward information for the chosen arm to update
            ts.update(click, context[arm_index])

            # For CTR calculation
            aligned_time_steps += 1
            cumulative_rewards += click
            aligned_ctr.append(cumulative_rewards / aligned_time_steps)
        if v == 0.01 and random_index == int(articleID):
            random_aligned_time_steps += 1
            random_cumulative_rewards += click
            random_aligned_ctr.append(
                random_cumulative_rewards / random_aligned_time_steps)
        t += 1
        # if t == 2:
        #     break
        ts.updateV(t)
    f.close()


def makeContext(pool_articles, user_features, articles):
    context = {}
    for article in pool_articles:
        if len(article) == 7 and len(user_features) == 6:
            all_zeros = np.zeros(len(articles) * 6 + 6)
            for i in range(len(articles)):
                if articles[i] == int(article[0]):
                    all_zeros[i * 6:i * 6 + 6] = user_features
            all_zeros[len(articles) * 6:] = article[1:]
            context[int(article[0])] = all_zeros
    return context


def experiment(folder):
    articles = get_all_articles()
    v_s = np.arange(0.01, 0.5, 0.1)
    random_results = []
    for v in v_s:
        v = float("{:.2f}".format(v))
        print("==================================================================================")
        print(f"Trying c = {v}")
        print("==================================================================================")
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
                yahoo_experiment(path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                                 random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t)
        plt.plot(aligned_ctr, label=f"c = {v}")
        if v == 0.01:
            plt.plot(random_aligned_ctr, label=f"Random CTR")
            random_results = random_cumulative_rewards
        print("total reward earned from Thompson:", cumulative_rewards)
        ts.printBandits()
    print("total reward earned from Random:", random_results)
    plt.ylabel("CTR ratio (For Thompson Sampling and Random)")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(f"figure_real_ts_random.png")


if __name__ == "__main__":
    start = datetime.now()
    experiment("data/R6A")
    end = datetime.now()
    print(f"Duration: {end - start}")
