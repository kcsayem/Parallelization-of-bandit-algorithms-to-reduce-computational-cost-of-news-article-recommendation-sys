from __future__ import print_function, division
from builtins import range
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, multivariate_normal
import ast, sys, random
import math
import pandas as pd

NUM_TRIALS = 2000


class ThompsonSampling:
    def __init__(self, contextDimension, num_of_bandits):
        self.d = contextDimension
        self.B = np.identity(self.d)
        v = 0.001 * math.sqrt(24 * self.d / 0.05 * math.log(1 / 0.1))
        self.v_squared = v ** 2
        self.f = np.zeros(self.d)
        self.trueMean = np.random.random((self.d,1))
        self.theta_hat = np.zeros(self.d)
        self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared * np.linalg.inv(self.B))
        self.bandits = [Bandit() for k in range(num_of_bandits)]
        self.regrets = []
    def sample(self):
        self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared * np.linalg.inv(self.B))
        return self.theta_estimate

    def update(self, reward, context):
        self.B += np.outer(context, context)
        self.f += context * reward
        self.theta_hat = np.dot(np.linalg.inv(self.B), self.f)

    def pull(self, context):
        max_ = np.argmax([np.dot(context[:, t].T, self.sample()) for t, b in enumerate(self.bandits)])
        optimal = np.argmax([np.dot(context[:, t].T, self.trueMean) for t, b in enumerate(self.bandits)])
        self.bandits[max_].update()
        estimated_reward = context[:, max_].dot(self.trueMean)
        true_reward = context[:, optimal].dot(self.trueMean)
        regret = true_reward-estimated_reward
        self.regrets.append(regret)
        return estimated_reward, max_

    def getEstimate(self):
        return self.theta_estimate

    def getTrueMean(self):
        return self.trueMean

    def printBandits(self):
        print("num times selected each bandit:", [b.N for b in self.bandits])
    def getRegrets(self):
        return self.regrets

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
    num_of_bandits = 10
    # np.random.seed(1)
    dimension = 10
    ts = ThompsonSampling(dimension, num_of_bandits)
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = np.zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        # Thompson sampling
        context = np.hstack([np.random.random((dimension, 1)) for p in range(num_of_bandits)])
        x, max_ = ts.pull(context)

        # update rewards
        rewards[i] = x

        # update the distribution for the bandit whose arm we just pulled
        ts.update(x, context[:, max_])

    # print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    ts.printBandits()
    print("true values: ", ts.getTrueMean().flatten())
    print("estimated values:", ts.getEstimate().flatten())
    # print("regrets", ts.getRegrets())
    ts.plot_regret()

if __name__ == "__main__":
    experiment()
