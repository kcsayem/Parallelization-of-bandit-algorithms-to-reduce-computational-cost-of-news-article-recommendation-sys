import numpy as np
import matplotlib.pyplot as plt
import sys
import ast
from tqdm import tqdm
from helper_functions import *
import logging

logger = logging.getLogger(__name__)

# Create class object for a single linear ucb   arm
class linucb_arm():

    def __init__(self, arm_index, alpha):
        # Track arm index
        self.index = arm_index

        # Keep track of alpha
        self.alpha = alpha
        self.N = 0

    def calc_UCB(self, x_array, theta, A_inv):
        # Find A inverse for ridge regression
        # A_inv = np.linalg.inv(A)
        # # print("A:", A)
        #
        # # Reshape covariates input into (d x 1) shape vector
        # x = x_array.reshape([-1, 1])
        # # print("x:",x)
        # # Find ucb based on p formulation (mean + std_dev)
        # # p is (1 x 1) dimension vector
        # # print("Theta Shape", theta.shape)
        # # print("Context Shape", x_array.shape)
        # # print("A Shape", A.shape)
        x = x_array.reshape([-1, 1])
        # p = _calc_UCB(self.alpha, x_array, theta, A)
        p = np.dot(theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))
        # print('p:',p)
        return p

    def update(self):
        self.N += 1


class LinUCB:

    def __init__(self, K_arms, d, alpha, lmd=1):
        self.K_arms = K_arms
        self.linucb_arms = []
        self.d = d
        self.alpha = alpha
        self.chosen_arm = -1
        self.d = d
        self.theta = None
        # Random Arm Context Generation
        # self.arm_context = [np.random.random((self.d, 1)) for i in range(0,self.K_arms)]

        # A: (d x d) matrix = D_a.T * D_a + I_d.
        # The inverse of A is used in ridge regression
        self.A = np.identity(d)
        self.A_inv = np.linalg.inv(self.A)

        self.A = self.A * lmd
        self.A_previous = np.array(self.A, copy=True)
        self.doubling_rounds = 0

        # b: (d x 1) corresponding response vector.
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d, 1])

    def calc_theta(self):
        # A_inv = np.linalg.inv(self.A)
        self.theta = np.dot(self.A_inv, self.b)
        # self.theta = _calc_theta(self.A, self.b)
        return self.theta

    def setup_arms(self, article_ids):
        self.linucb_arms = [linucb_arm(article_ids[k], self.alpha) for k in range(len(article_ids))]

    def reward_update(self, reward, x):
        # Reshape covariates input into (d x 1) shape vector
        x = x.reshape([-1, 1])

        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)
        # update A_inv
        self.A_inv = inverse(self.A_inv, np.dot(x, x.T))
        # Update b which is (d x 1) vector
        # reward is scalar
        self.b += reward * x

    def printBandits(self):
        # print("num times selected each bandit:", [b.N for b in self.linucb_arms])
        msg = f"Num times selected each bandit: {[b.N for b in self.linucb_arms]}"
        logger.info(msg)
        msg = f"Doubling Rounds: {self.doubling_rounds}"
        logger.info(msg)

    def select_arm(self, context):
        # selecting arms for specific times
        specific_bandits = []
        for key in context:
            for bandit in self.linucb_arms:
                if key == bandit.index:
                    specific_bandits.append(bandit)

        # print(len(specific_bandits))

        # Initiate ucb to be 0
        highest_ucb = float('-inf')
        max_index = -1

        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []
        theta = self.calc_theta()

        for arm in specific_bandits:
            cur_value = arm.calc_UCB(context[arm.index], theta, self.A_inv)
            if highest_ucb < cur_value:
                # set new max ucb
                highest_ucb = cur_value

                # reset_candidate arms
                candidate_arms = [arm.index]

            # If there is a tie, append to candidate_arms
            if cur_value == highest_ucb:
                if arm.index not in candidate_arms:
                    candidate_arms.append(arm.index)

        # Choose based on candidate_arms randomly (tie breaker)
        # print('last step:', candidate_arms)
        chosen_arm = np.random.choice(candidate_arms)
        for bandit in specific_bandits:
            if bandit.index == chosen_arm:
                bandit.update()
        self.chosen_arm = chosen_arm

        # random choosen_arm
        random = np.random.choice(specific_bandits)
        return chosen_arm, random.index


class LazyLinUCB(LinUCB):
    def __init__(self, K_arms, d, alpha, lmd):
        super().__init__(K_arms, d, alpha, lmd)



    def doubling_round(self):
        if ispositivesemidifinate(2 * self.A_previous - self.A):
            self.A_previous = np.array(self.A, copy=True)
        else:
            self.doubling_rounds += 1
            self.A_previous = np.array(self.A, copy=True)

    def reward_update_batch(self, rewards, contexts):
        rc_sum = 0
        bc_sum = 0
        for i in range(len(contexts)):
            rc_sum+=contexts[i][list(contexts[i].keys())[0]]*rewards[i]
            bc_sum+=np.outer(contexts[i][list(contexts[i].keys())[0]], contexts[i][list(contexts[i].keys())[0]])
        self.b += rc_sum.reshape([-1,1])
        self.A = bc_sum
        self.A_inv = inverse(self.A_inv, bc_sum)
        self.calc_theta()

    def reward_update_iteration(self, x):
        # Reshape covariates input into (d x 1) shape vector
        x = x.reshape([-1, 1])

        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)
        # update A_inv
        self.A_inv = inverse(self.A_inv, np.dot(x, x.T))


class NonLazyLinUCB(LinUCB):
    def __init__(self, K_arms, d, alpha, lmd):
        super().__init__(K_arms, d, alpha, lmd)


    def doubling_round(self):
        if ispositivesemidifinate(2 * self.A_previous - self.A):
            self.A_previous = np.array(self.A, copy=True)
        else:
            self.doubling_rounds += 1
            self.A_previous = np.array(self.A, copy=True)


    def reward_update_batch(self, rewards, contexts):
        rc_sum = 0
        bc_sum = 0
        for i in range(len(contexts)):
            rc_sum+=contexts[i][list(contexts[i].keys())[0]]*rewards[i]
            bc_sum+=np.outer(contexts[i][list(contexts[i].keys())[0]], contexts[i][list(contexts[i].keys())[0]])
        self.b += rc_sum.reshape([-1,1])
        self.A = bc_sum
        self.A_inv = inverse(self.A_inv, bc_sum)
        self.calc_theta()


    def reward_update_iteration(self, x):
        # Reshape covariates input into (d x 1) shape vector
        x = x.reshape([-1, 1])

        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)
        # update A_inv
        self.A_inv = inverse(self.A_inv, np.dot(x, x.T))



