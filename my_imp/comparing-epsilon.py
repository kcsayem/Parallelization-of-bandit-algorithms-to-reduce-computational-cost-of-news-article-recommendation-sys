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
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    data = np.empty(N)
    cumulatives = []
    for eps in epsilons:
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
            plt.plot(np.ones(N) * p,label=f"p{i} = {p}")
            plt.xscale('log')
        plt.legend()
        plt.title("Log scale plot")
        plt.show()
        for b in bandits:
            print(f"mean estimate: {b.mean}")

    return cumulatives
def plot_res(res,epsilons,scale):
    for i in range(len(res)):
        plt.plot(res[i], label=f'eps = {epsilons[i]}')
    plt.title(f"{str.upper(scale)} scale plot")
    plt.xscale(scale)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    argv = sys.argv
    probabilities = ast.literal_eval(argv[1])
    epsilons = ast.literal_eval(argv[2])
    res = run_experiment(probabilities, epsilons, 100000)
    plot_res(res,epsilons,"log")
    # plot_res(res,epsilons,"linear")

