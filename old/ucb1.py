from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
import sys
import ast
NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p):
        # p: the win rate
        self.p = p
        self.p_estimate = 0.
        self.N = 0.  # num samples collected so far

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N


def ucb(mean, n, nj):
    return mean + np.sqrt(2 * np.log(n) / nj)


def run_experiment():
    argv = sys.argv
    BANDIT_PROBABILITIES = ast.literal_eval(argv[1])
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.empty(NUM_TRIALS)
    total_plays = 0

    # initialization: play each bandit once
    for j in range(len(bandits)):
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)

    for i in range(NUM_TRIALS):
        j = np.argmax([ucb(b.p_estimate, total_plays, b.N) for b in bandits])
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)

        # for the plot
        rewards[i] = x
    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

    # plot moving average ctr
    plt.xlabel("Trials", fontsize=12)
    plt.ylabel("Cumulative Rewards", fontsize=12)
    plt.plot(cumulative_average,label=f"Estimated Win Rate = {rewards.sum() / NUM_TRIALS}")
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES), label=f"Max Win Rate = {np.max(BANDIT_PROBABILITIES)}")
    plt.legend()
    plt.title(f"Log scale plot ( Trials = {NUM_TRIALS} )")
    plt.xscale('log')
    plt.show()

    # # plot moving average ctr linear
    # plt.plot(cumulative_average)
    # plt.title("Linear scale plot")
    # plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    # plt.show()

    for b in bandits:
        print(f"mean_estimate: {b.p_estimate}")

    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.N for b in bandits])

    return cumulative_average


if __name__ == '__main__':
    run_experiment()
