import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys
import ast
from tqdm import tqdm
from helper_functions import *
import math

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


class linucb_policy():

    def __init__(self, K_arms, d, alpha, lmd):
        self.K_arms = K_arms
        self.linucb_arms = []
        self.d = d
        self.alpha = alpha
        self.chosen_arm = -1
        self.d = d
        self.theta = None
        self.doubling_rounds = 0
    
        

        # A: (d x d) matrix = D_a.T * D_a + I_d.
        # The inverse of A is used in ridge regression
        self.A = np.identity(d) * lmd 
        self.A_previous = self.A
        self.A_inv = np.linalg.inv(self.A)

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

    def reward_update_iteration(self, x):
        # Reshape covariates input into (d x 1) shape vector
        x = x.reshape([-1, 1])

        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)
        # update A_inv
        self.A_inv = inverse(self.A_inv, np.dot(x, x.T))
        
    
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


    def printBandits(self):
        print("num times selected each bandit:", [b.N for b in self.linucb_arms])
        print('Doubling Round', self.doubling_rounds)
    
    def doubling_round(self):
        if ispositivesemidifinate(self.A - 2 * self.A_previous):
            self.doubling_rounds += 1
            self.A_previous = self.A
        else:
            self.A_previous = self.A

    def calculate_projection(self, u, v):
        v_norm = np.sqrt(sum(v ** 2))
        proj_on_v = (np.dot(u, v) / v_norm ** 2) * v
        return proj_on_v
    

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

        #random choosen_arm
        random = np.random.choice(specific_bandits)
        return chosen_arm, random.index

def yahoo_experiment(path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                     random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t,p):
    f = open(path, "r")
    max_ = get_num_lines(path)
    # for line_data in tqdm(f, total=max_):
    for iteration in tqdm(range(math.ceil(max_/p))):
        lines = []
        for i in range(p):
            lines.append(f.readline())
        clicks = np.array([])
        contexts = []
        article_ids = []
        for line in lines:
            if len(line)==0: continue
            tim, articleID, click, user_features, pool_articles = parseLine(
            line)
            context = makeContext(pool_articles, user_features, articles)
            clicks = np.append(clicks,click)
            contexts.append(context)
            article_ids.append(int(articleID))
        # print(context)
        # break
        arm_indexes = []
        for c in range(len(contexts)):
            arm_index, specific_bandits = ts.select_arm(contexts[c])
            arm_indexes.append(arm_index)
        for r in range(len(contexts)):
            #random_index = np.random.choice(specific_bandits).index
            if arm_indexes[r] == article_ids[r]:
                # Use reward information for the chosen arm to update

                # For CTR calculation
                aligned_time_steps += 1
                cumulative_rewards += clicks[r]
                aligned_ctr.append(cumulative_rewards / aligned_time_steps)
            else:
                clicks[r] = 0
        ts.reward_update_batch(clicks,contexts)
        # Doubling round Check
        ts.doubling_round()

        # if v == 0.01 and random_index == int(articleID):
        #     random_aligned_time_steps += 1
        #     random_cumulative_rewards += click
        #     random_aligned_ctr.append(
        #         random_cumulative_rewards / random_aligned_time_steps)
        t += 1
        # if t == 2:
        #     break
        
    f.close()
    return ts, aligned_time_steps, cumulative_rewards, aligned_ctr, random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t




def experiment(folder):
    articles = get_all_articles()
    # alphas = np.arange(0.01, 0.5, 0.1)
    alphas = [0.3]
    # v_s = alphas[:1]
    lmd = 0.2
    p = 50
    random_results = []
    for v in alphas:
        v = float("{:.2f}".format(v))
        print("==================================================================================")
        print(f"Trying alpha = {v}, lambda = {lmd}, p = {p}")
        print("==================================================================================")
        ts = linucb_policy(len(articles), len(articles) * 6 + 6,v,lmd)
        ts.setup_arms(articles)
        aligned_time_steps = 0
        random_aligned_time_steps = 0
        cumulative_rewards = 0
        random_cumulative_rewards = 0
        aligned_ctr = []
        random_aligned_ctr = []
        t = 1
        # print('pass')
        for root, dirs, files in os.walk(folder):
            # print('pass 1')
            for filename in files:
                # print('pass 2')
                path = os.path.join(root, filename)
                # print(path)
                ts, aligned_time_steps, cumulative_rewards, aligned_ctr, random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t = yahoo_experiment(
                    path, v, articles, ts, aligned_time_steps, cumulative_rewards, aligned_ctr,
                    random_aligned_time_steps, random_cumulative_rewards, random_aligned_ctr, t,p)
                # print('pass 3')
        plt.plot(aligned_ctr, label=f"alpha = {v}")
        if v == 0.01:
            plt.plot(random_aligned_ctr, label=f"Random CTR")
            random_results = random_cumulative_rewards
        print("total reward earned from Linucb:", cumulative_rewards)
        ts.printBandits()
    print("total reward earned from Random:", random_results)
    plt.ylabel("CTR ratio (For LinUCB and Random)")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(f"./parallel_non_lazy_linucb.png")

if __name__ == "__main__":
    start = datetime.now()
    experiment("data/R6A_spec")
    end = datetime.now()
    print(f"Duration: {end - start}")