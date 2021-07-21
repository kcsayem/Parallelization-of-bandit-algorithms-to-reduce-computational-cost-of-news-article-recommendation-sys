from __future__ import print_function, division
from builtins import range
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
import ast,sys,random
import math
NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
  def __init__(self, contextDimension):
      self.d = contextDimension
      self.B = np.identity(self.d)
      v = 0.001 * math.sqrt(24 * self.d / 0.05 * math.log(1 / 0.1))
      self.v_squared = v**2
      self.f = np.zeros(self.d)
      self.trueMean = np.random.random((self.d,1))
      self.theta_hat = np.zeros(self.d)
      self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared * np.linalg.inv(self.B))
      # self.regret = abs(np.dot(np.transpose(self.theta_estimate),context)-self.trueMean)
      self.regret=0
      self.N = 0

  def pull(self,context):
      return context.dot(self.trueMean)

  def sample(self):
    self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared * np.linalg.inv(self.B))
    return self.theta_estimate

  def update(self, reward,context):
    self.B += np.outer(context,context)
    self.f += context*reward
    self.theta_hat = np.dot(np.linalg.inv(self.B),self.f)
    self.N += 1
  def getEstimate(self):
      return self.theta_estimate
def plot(bandits, trial):
  x = np.linspace(0, 1, 200)
  for b in bandits:
    y = beta.pdf(x, b.a, b.b)
    plt.plot(x, y, label=f"real p: {b.p:.4f}, win rate = {b.a - 1}/{b.N}")
  plt.title(f"Bandit distributions after {trial} trials")
  plt.legend()
  plt.show()


def experiment():
  BANDIT_PROBABILITIES = 10
  np.random.seed(42)
  bandits = [Bandit(10) for k in range(BANDIT_PROBABILITIES)]
  sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
  rewards = np.zeros(NUM_TRIALS)
  for i in range(NUM_TRIALS):
    # Thompson sampling
    context = np.hstack([np.random.random((10,1)) for p in range(BANDIT_PROBABILITIES)])
    j = np.argmax([np.dot(context[:,t].T,b.sample()) for t,b in enumerate(bandits)])
    # plot the posteriors
    # if i in sample_points:
    #   plot(bandits, i)

    # pull the arm for the bandit with the largest sample
    x = bandits[j].pull(context[:,j])

    # update rewards
    rewards[i] = x

    # update the distribution for the bandit whose arm we just pulled
    bandits[j].update(x,context[:,j])

  # print total reward
  print("total reward earned:", rewards.sum())
  print("overall win rate:", rewards.sum() / NUM_TRIALS)
  print("num times selected each bandit:", [b.N for b in bandits])
  print("true values: ", [b.trueMean for b in bandits])
  print("estimated values:",[b.getEstimate() for b in bandits])
  print("regrets",[b.regret for i,b in enumerate(bandits)])


if __name__ == "__main__":
  experiment()
