import numpy as np
import matplotlib.pyplot as plt
import sys
import ast
from tqdm import tqdm
from helper_functions import inverse, get_num_lines, get_all_articles, makeContext, parseLine
# from numba import jit

# # Numba helper function for faster calculation
# @jit(nopython=True)
# def _calc_UCB(alpha, x, theta, A):
#     # Find A inverse for ridge regression
#     A_inv = np.linalg.inv(A)
#     # print("A:", A)
#
#     # Reshape covariates input into (d x 1) shape vector
#     # x = x_array.reshape([-1, 1])
#     # print("x:",x)
#     # Find ucb based on p formulation (mean + std_dev)
#     # p is (1 x 1) dimension vector
#     # print("Theta Shape", theta.shape)
#     # print("Context Shape", x_array.shape)
#     # print("A Shape", A.shape)
#     p = np.dot(theta.T, x) + alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))
#     # print('p:',p)
#     return p
# @jit(nopython=True)
# def _calc_theta(A, b):
#     A_inv = np.linalg.inv(A)
#     theta = np.dot(A_inv, b)
#     return theta



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
        p = np.dot(theta.T, x) + alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))
        # print('p:',p)
        return p

    def update(self):
        self.N += 1


class linucb_policy():

    def __init__(self, K_arms, d, alpha):
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
        print("num times selected each bandit:", [b.N for b in self.linucb_arms])

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


def ctr_simulator(K_arms, d, alpha, data_path):
    # Initiate policy
    linucb_policy_object = linucb_policy(K_arms=K_arms, d=d, alpha=alpha)

    # Instantiate trackers
    aligned_time_steps = 0
    cumulative_rewards = 0
    aligned_ctr = []
    unaligned_ctr = []  # for unaligned time steps

    # Open data
    with open(data_path, "r") as f:

        for line_data in tqdm(f):

            # 1st column: Logged data arm.
            # Integer data type
            data_arm = int(line_data.split()[0])

            # 2nd column: Logged data reward for logged chosen arm
            # Float data type
            data_reward = float(line_data.split()[1])

            # 3rd columns onwards: 100 covariates. Keep in array of dimensions (100,) with float data type
            covariate_string_list = line_data.split()[2:]
            data_x_array = np.array([float(covariate_elem) for covariate_elem in covariate_string_list])
            data_x_array = np.linalg.norm(data_x_array)
            # Find policy's chosen arm based on input covariates at current time step
            arm_index = linucb_policy_object.select_arm(data_x_array)

            # Check if arm_index is the same as data_arm (ie same actions were chosen)
            # Note that data_arms index range from 1 to 10 while policy arms index range from 0 to 9.
            if arm_index + 1 == data_arm:
                # calculating projection
                user_context = data_x_array
                arm_context = linucb_policy_object.arm_context[arm_index]
                projection = linucb_policy_object.calculate_projection(user_context, arm_context)
                # Use reward information for the chosen arm to update

                linucb_policy_object.reward_update(data_reward, projection)

                # For CTR calculation
                aligned_time_steps += 1
                cumulative_rewards += data_reward
                aligned_ctr.append(cumulative_rewards / aligned_time_steps)
    linucb_policy_object.printBandits()
    return aligned_time_steps, cumulative_rewards, aligned_ctr, linucb_policy_object


def yahoo_experiment(filename, alpha):
    # articles = [109498, 109509, 109508, 109473, 109503, 109502, 109501, 109492, 109495, 109494, 109484, 109506, 109510,
    #             109514, 109505, 109515, 109512, 109513, 109511, 109453, 109519, 109520, 109521, 109522, 109523, 109524,
    #             109525, 109526, 109527, 109528, 109529, 109530, 109534, 109532, 109533, 109531, 109535, 109536, 109417,
    #             109542, 109538, 109543, 109540, 109544, 109545, 109546, 109547, 109548, 109550, 109552]
    articles = get_all_articles()
    f = open(filename, "r")
    # Initiate policy
    linucb_policy_object = linucb_policy(K_arms=271, d=1632, alpha=alpha)
    # setup arms
    linucb_policy_object.setup_arms(articles)
    aligned_time_steps = 0
    cumulative_rewards = 0
    aligned_ctr = []
    random_aligned_time_steps = 0
    random_cumulative_rewards = 0
    random_aligned_ctr = []
    t = 1
    max_ = get_num_lines(filename)
    for line_data in tqdm(f, total=max_):
        tim, articleID, click, user_features, pool_articles = parseLine(line_data)
        # 1st column: Logged data arm.
        # Integer data type
        context = makeContext(pool_articles, user_features, articles)
        # for article in pool_articles:
        #     if len(article) == 7 and len(user_features) == 6:
        #         # context[int(article[0])] = np.reshape(np.concatenate((user_features, article[1:])), (12, 1))
        #         context[int(article[0])] = linucb_policy_object.calculate_projection(user_features, article[1:])
        #         # context[int(article[0])] = np.outer(article[1:], user_features).flatten()
        # print(context)

        # break
        arm_index, random_selection = linucb_policy_object.select_arm(context)
        if arm_index == int(articleID):
            # Use reward information for the chosen arm to update
            linucb_policy_object.reward_update(click, context[arm_index])

            # For CTR calculation
            aligned_time_steps += 1
            cumulative_rewards += click
            aligned_ctr.append(cumulative_rewards / aligned_time_steps)

        if random_selection == int(articleID):

            # For CTR calculation
            random_aligned_time_steps += 1
            random_cumulative_rewards += click
            random_aligned_ctr.append(random_cumulative_rewards / random_aligned_time_steps)

        # t += 1
        # if t == max_:
        #     break

    linucb_policy_object.printBandits()
    return aligned_time_steps, cumulative_rewards, aligned_ctr, linucb_policy_object, random_aligned_ctr, random_cumulative_rewards


if __name__ == "__main__":
    argv = sys.argv
    # alpha_inputs = ast.literal_eval(argv[1])
    # alpha_inputs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    alpha_inputs = [0.1]
    data_path = "data/R6A/data_2"
    for alpha in alpha_inputs:
        print(f"Trying with alpha = {alpha}")
        aligned_time_steps, cum_rewards, aligned_ctr, policy, random_aligned_ctr, random_cumulative_rewards = yahoo_experiment(data_path, alpha)
        print("Cumulative Reward: ", cum_rewards)
        plt.plot(aligned_ctr, label="alpha = " + str(alpha))
    plt.plot(random_aligned_ctr, label="For random selecton, CumR=" + str(random_cumulative_rewards))
    plt.ylabel("CTR ratio (For Single Theta)")
    plt.xlabel("Time")
    plt.legend()
    plt.show()
