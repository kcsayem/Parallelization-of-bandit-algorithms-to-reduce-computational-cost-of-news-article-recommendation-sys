from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
import ast
import sys

class Bandit:
    def __init__(self, m):
        self.m = m
        self.mean = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x



def run_experiment(BANDIT_PROBABILITIES, epsilons, N):
    cumulatives = []
    for eps in epsilons:
        data = np.empty(N)
        bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
        print(f"Trying with {eps}")
        for i in range(N):
            # epsilon greedy
            p = np.random.random()
            if p < eps:
                j = np.random.choice(len(BANDIT_PROBABILITIES))
            else:
                j = np.argmax([b.mean for b in bandits])
            x = bandits[j].pull()
            bandits[j].update(x)

            # for the plot
            data[i] = x
        cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
        cumulatives.append(cumulative_average)
        # plot moving average ctr
        plt.plot(cumulative_average,label=f"eps = {eps}")
        for i, p in enumerate(BANDIT_PROBABILITIES):
            plt.plot(np.ones(N) * p,label=f"m{i} = {p}")
            plt.xscale('log')
        plt.legend()
        plt.title("Log scale plot")
        plt.xlabel("Times", fontsize=12)
        plt.ylabel("Cumulative Rewards", fontsize=12)
        plt.show()
        for b in bandits:
            print(f"mean estimate: {b.mean}")
        print(f"Cumulative Average: {cumulative_average.sum()/N}")

    return cumulatives
def plot_res(res,epsilons,means,scale):
    for i in range(len(res)):
        plt.plot(res[i], label=f'EPS = {epsilons[i]}, Estimated Mean = {round(res[i].sum()/10000,3)}')
    plt.plot(np.ones(10000) * np.max(means), label=f"Max Mean = {np.max(means)}")
    plt.title(f"{str.upper(scale)} scale plot ( Trials = {10000} )")
    plt.xscale(scale)
    plt.xlabel("Trials", fontsize=12)
    plt.ylabel("Cumulative Rewards", fontsize=12)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    argv = sys.argv
    means = ast.literal_eval(argv[1])
    epsilons = ast.literal_eval(argv[2])
    res = run_experiment(means, epsilons, 10000)
    plot_res(res,epsilons,means,"log")
    # plot_res(res,epsilons,"linear")

