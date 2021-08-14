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
from helper_functions import inverse, get_num_lines
from scipy.sparse.linalg import cg
# from numpy.random import multivariate_normal
import cProfile
from numpy.random import default_rng

NUM_TRIALS = 2000

rng = default_rng()

def parseLine(line):
    line = line.split("|")

    tim, articleID, click = line[0].strip().split(" ")
    tim, articleID, click = int(tim), int(articleID), int(click)
    user_features = np.array([float(x.strip().split(':')[1])
                             for x in line[1].strip().split(' ')[1:]])

    pool_articles = [l.strip().split(" ") for l in line[2:]]
    pool_articles = np.array(
        [[int(l[0])] + [float(x.split(':')[1]) for x in l[1:]] for l in pool_articles])
    return tim, articleID, click, user_features, pool_articles


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
            self.theta_hat, self.v_squared * self.B_inv,method='cholesky')
        self.bandits = []
        self.regrets = []
        self.trueMean = 0

    def setUpBandits(self, articleIds):
        self.bandits = [Bandit(articleIds[k]) for k in range(len(articleIds))]

    def sample(self):
        self.theta_estimate = rng.multivariate_normal(
            self.theta_hat, self.v_squared * self.B_inv,method='cholesky')
        # self.theta_estimate = multivariate_normal(mean=self.theta_hat, cov=(self.v_squared * self.B_inv))
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

    def getTrueMean(self):
        return self.trueMean

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


def experiment():
    np.set_printoptions(suppress=True)
    data_path = "data/news_dataset.txt"
    num_of_bandits = 10
    # np.random.seed(1)
    dimension = 10
    R_values = ast.literal_eval(sys.argv[1])
    t = 1
    for R in R_values:
        print(f"Trying R = {R}")
        ts = ThompsonSampling(dimension, num_of_bandits, R)
        aligned_time_steps = 0
        cumulative_rewards = 0
        aligned_ctr = []
        f = open(data_path, "r")
        for line_data in tqdm(f):

            # 1st column: Logged data arm.
            # Integer data type
            data_arm = int(line_data.split()[0])

            # 2nd column: Logged data reward for logged chosen arm
            # Float data type
            data_reward = float(line_data.split()[1])
            covariate_string_list = line_data.split()[2:]
            data_x_array = np.array([float(covariate_elem)
                                    for covariate_elem in covariate_string_list])
            split_array = np.array_split(data_x_array, 10)
            context = np.hstack([np.reshape(p, (dimension, 1))
                                for p in split_array])
            x, arm_index = ts.pull(context)
            if arm_index + 1 == data_arm:
                # Use reward information for the chosen arm to update
                ts.update(data_reward, context[:, arm_index])

                # For CTR calculation
                aligned_time_steps += 1
                cumulative_rewards += data_reward
                aligned_ctr.append(cumulative_rewards / aligned_time_steps)
            t += 1
            ts.updateV(t)
        plt.plot(aligned_ctr, label=f"R = {R}")
        print("total reward earned:", cumulative_rewards)
        ts.printBandits()
    plt.ylabel("CTR ratio (For Thompson Sampling)")
    plt.xlabel("Time")
    plt.legend()
    plt.show()


def yahoo_experiment(filename):
    articles = [109498, 109509, 109508, 109473, 109503, 109502, 109501, 109492, 109495, 109494, 109484, 109506, 109510,
                109514, 109505, 109515, 109512, 109513, 109511, 109453, 109519, 109520, 109521, 109522, 109523, 109524,
                109525, 109526, 109527, 109528, 109529, 109530, 109534, 109532, 109533, 109531, 109535, 109536, 109417,
                109542, 109538, 109543, 109540, 109544, 109545, 109546, 109547, 109548, 109550, 109552]
    v_s = np.arange(0.01, 0.5, 0.1)
    random_results = []
    for v in v_s:
        v = float("{:.2f}".format(v))
        print("==================================================================================")
        print(f"Trying c = {v}")
        print("==================================================================================")
        f = open(filename, "r")
        ts = ThompsonSampling(306, 0.0001, v)
        ts.setUpBandits(articles)
        aligned_time_steps = 0
        random_aligned_time_steps = 0
        cumulative_rewards = 0
        random_cumulative_rewards = 0
        aligned_ctr = []
        random_aligned_ctr = []
        t = 1
        max_ = get_num_lines(filename)
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


def makeContext(pool_articles, user_features, articles):
    context = {}
    for article in pool_articles:
        if len(article) == 7 and len(user_features) == 6:
            all_zeros = np.zeros(306)
            for i in range(len(articles)):
                if articles[i] == int(article[0]):
                    all_zeros[i * 6:i * 6 + 6] = user_features
            all_zeros[300:] = article[1:]
            context[int(article[0])] = all_zeros
    return context


# def profile_run():
#     yahoo_experiment("data/data")

def num_articles(filename):
    f = open(filename, "r")
    articles = []
    for line_data in tqdm(f):
        tim, articleID, click, user_features, pool_articles = parseLine(
            line_data)
        for article in pool_articles:
            if article[0] not in articles:
                articles.append(article[0])
    return articles


if __name__ == "__main__":
    # np.set_printoptions(suppress=True)
    yahoo_experiment("data/data")
