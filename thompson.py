from helper_functions import *
from numpy.random import default_rng
import logging

logger = logging.getLogger(__name__)


class ThompsonSampling:
    def __init__(self, d, c, lmb=1, SEED=42):
        self.rng = default_rng()
        self.SEED = SEED
        self.d = d
        self.B = np.identity(self.d) * lmb
        self.B_inv = np.linalg.inv(self.B)
        self.B_previous = np.copy(self.B)
        self.alpha = c
        self.v_squared = c ** 2
        self.f = np.zeros(self.d)
        self.theta_hat = np.zeros(self.d)
        self.theta_estimate = self.rng.multivariate_normal(
            self.theta_hat, self.v_squared * self.B_inv, method='cholesky')
        self.doubling_rounds = 0
        self.bandits = []

    def setup_bandits(self, articleIds):
        self.bandits = [Bandit(articleIds[k]) for k in range(len(articleIds))]

    def sample(self):
        self.theta_estimate = random_sampling(
            self.theta_hat, self.v_squared * self.B_inv, self.d, 1, self.SEED)
        return self.theta_estimate

    def update(self, reward, context):
        self.f += context * reward
        self.B_inv = inverse(self.B_inv, np.outer(context, context))
        self.theta_hat = np.dot(self.B_inv, self.f)

    def doubling_round(self):
        if ispositivesemidifinate(2 * self.B_previous - self.B):
            self.B_previous = np.copy(self.B)
        else:
            self.doubling_rounds += 1
            self.B_previous = np.copy(self.B)

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

    def print_bandits(self):
        logging.info(f"alpha for Thompson Sampling: {self.alpha}")
        msg = f"Num times selected each bandit for Thompson Sampling: {[b.N for b in self.bandits]}"
        logger.info(msg)
        # msg = f"Doubling Rounds: {self.doubling_rounds}"
        # logger.info(msg)


class Bandit:
    def __init__(self, index):
        self.regret = 0
        self.N = 0
        self.index = index

    def update(self):
        self.N += 1


class LazyThompsonSampling(ThompsonSampling):
    def __init__(self, d, c, lmb, SEED):
        ThompsonSampling.__init__(self, d, c, lmb, SEED)

    def update_iteration(self, context):
        self.B += np.outer(context, context)
        self.B_inv = inverse(self.B_inv, np.outer(context, context))

    def update_batch(self, rewards, contexts):
        rc_sum = 0
        for i in range(len(contexts)):
            rc_sum += contexts[i][list(contexts[i].keys())[0]] * rewards[i]
        self.f += rc_sum
        self.theta_hat = np.dot(self.B_inv, self.f)


class NonLazyThompsonSampling(ThompsonSampling):
    def __init__(self, d, c, lmb, SEED):
        ThompsonSampling.__init__(self, d, c, lmb, SEED)

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
