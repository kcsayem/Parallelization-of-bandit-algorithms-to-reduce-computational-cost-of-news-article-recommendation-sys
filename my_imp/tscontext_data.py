from __future__ import print_function, division
from builtins import range
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, multivariate_normal
import ast, sys, random
import math
import pandas as pd
from tqdm import tqdm

NUM_TRIALS = 2000


class ThompsonSampling:
    def __init__(self, contextDimension, num_of_bandits, R):
        self.d = contextDimension
        self.B = np.identity(self.d)
        v = R * math.sqrt(24 * self.d / 0.05 * math.log(1 / np.random.random(1)))
        self.v_squared = v ** 2
        self.f = np.zeros(self.d)
        self.theta_hat = np.zeros(self.d)
        self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared * np.linalg.inv(self.B))
        self.bandits = [Bandit() for k in range(num_of_bandits)]
        self.regrets = []
        self.trueMean = 0

    def sample(self):
        self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared * np.linalg.inv(self.B))
        return self.theta_estimate

    def update(self, reward, context):
        self.B += np.outer(context, context)
        self.f += context * reward
        self.theta_hat = np.dot(np.linalg.inv(self.B), self.f)

    def pull(self, context):
        theta_estimate = self.sample()
        max_ = np.argmax([np.dot(context[:, t].T, theta_estimate) for t, b in enumerate(self.bandits)])
        self.bandits[max_].update()
        estimated_reward = context[:, max_].dot(theta_estimate)
        return estimated_reward, max_

    def getEstimate(self):
        return self.theta_estimate

    def getTrueMean(self):
        return self.trueMean

    def printBandits(self):
        print("num times selected each bandit:", [b.N for b in self.bandits])

    def getRegrets(self):
        return self.regrets

    def updateV(self, t):
        v = 0.001 * math.sqrt(24 * self.d / 0.05 * math.log(t / 0.1))
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
    def __init__(self):
        self.regret = 0
        self.N = 0

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
            data_x_array = np.array([float(covariate_elem) for covariate_elem in covariate_string_list])
            split_array = np.array_split(data_x_array, 10)
            context = np.hstack([np.reshape(p, (dimension, 1)) for p in split_array])
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
    plt.legend()
    plt.show()
if __name__ == "__main__":
    experiment()
