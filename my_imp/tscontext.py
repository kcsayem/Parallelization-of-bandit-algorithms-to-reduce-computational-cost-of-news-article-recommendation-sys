from __future__ import print_function, division
from builtins import range
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
import ast,sys,random

NUM_TRIALS = 20000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
  def __init__(self, contextDimension, v_squared, trueMean, context):
      self.d = contextDimension
      self.B = np.identity(self.d)
      self.v_squared = v_squared
      self.f = np.zeros(self.d)
      self.trueMean = np.dot(np.transpose(context),trueMean)
      self.theta_hat = np.zeros(self.d)
      self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared * np.linalg.inv(self.B))
      self.regret = abs(np.dot(np.transpose(self.theta_estimate),context)-self.trueMean)
      self.N = 0
      self.myContext = context

  def pull(self):
    if self.regret <= abs(np.dot(np.transpose(self.myContext),self.theta_estimate)-self.trueMean):
        return 0
    else:
        self.regret = abs(np.dot(np.transpose(self.myContext),self.theta_estimate)-self.trueMean)
        return 1
  def sample(self):
    self.theta_estimate = np.random.multivariate_normal(self.theta_hat, self.v_squared * np.linalg.inv(self.B))
    return np.dot(np.transpose(self.myContext),self.theta_estimate)

  def update(self, reward):
    self.B += np.outer(self.myContext,self.myContext)
    self.f += self.myContext*reward
    self.theta_hat = np.dot(np.linalg.inv(self.B),self.f)
    self.N += 1
  def getEstimate(self):
      return np.dot(np.transpose(self.myContext),self.theta_estimate)
def plot(bandits, trial):
  x = np.linspace(0, 1, 200)
  for b in bandits:
    y = beta.pdf(x, b.a, b.b)
    plt.plot(x, y, label=f"real p: {b.p:.4f}, win rate = {b.a - 1}/{b.N}")
  plt.title(f"Bandit distributions after {trial} trials")
  plt.legend()
  plt.show()


def experiment():
  BANDIT_PROBABILITIES = ast.literal_eval(sys.argv[1])
  BANDIT_CONTEXTS = []
  np.random.seed(42)
  for i in range(len(BANDIT_PROBABILITIES)):
      sample = np.ones(10)
      BANDIT_CONTEXTS.append(sample)
  bandits = [Bandit(10,25,np.ones(10)*p,BANDIT_CONTEXTS[k]) for k,p in enumerate(BANDIT_PROBABILITIES)]
  sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
  rewards = np.zeros(NUM_TRIALS)
  for i in range(NUM_TRIALS):
    # Thompson sampling
    j = np.argmax([b.sample() for b in bandits])

    # plot the posteriors
    # if i in sample_points:
    #   plot(bandits, i)

    # pull the arm for the bandit with the largest sample
    x = bandits[j].pull()

    # update rewards
    rewards[i] = x

    # update the distribution for the bandit whose arm we just pulled
    bandits[j].update(x)

  # print total reward
  print("total reward earned:", rewards.sum())
  print("overall win rate:", rewards.sum() / NUM_TRIALS)
  print("num times selected each bandit:", [b.N for b in bandits])
  print("true values: ", [b.trueMean for b in bandits])
  print("estimated values:",[b.getEstimate() for b in bandits])
  print("regrets",[b.regret for i,b in enumerate(bandits)])


if __name__ == "__main__":
  experiment()