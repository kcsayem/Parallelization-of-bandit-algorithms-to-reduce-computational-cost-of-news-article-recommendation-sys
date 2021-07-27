import numpy as np
import matplotlib.pyplot as plt
import sys
import ast
from tqdm import tqdm



# Create class object for a single linear ucb   arm
class linucb_arm():

    def __init__(self, arm_index, d, alpha):
        # Track arm index
        self.arm_index = arm_index

        # Keep track of alpha
        self.alpha = alpha
        self.N = 0

    def calc_UCB(self, x_array, theta, A):
        # Find A inverse for ridge regression
        A_inv = np.linalg.inv(A)
        # print("A:", A)

        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1, 1])
        # print("x:",x)
        # Find ucb based on p formulation (mean + std_dev)
        # p is (1 x 1) dimension vector
        p = np.dot(theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))
        # print('p:',p)
        return p

    def update(self):
        self.N += 1


class linucb_policy():

    def __init__(self, K_arms, d, alpha):
        self.K_arms = K_arms
        self.linucb_arms = [linucb_arm(arm_index=i, d=d, alpha=alpha) for i in range(K_arms)]
        self.chosen_arm = -1
        self.d = d
        self.theta = None
        self.arm_context = [np.random.random((self.d, 1)) for i in range(0,self.K_arms)]


        # A: (d x d) matrix = D_a.T * D_a + I_d.
        # The inverse of A is used in ridge regression
        self.A = np.identity(d)

        # b: (d x 1) corresponding response vector.
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d, 1])

    def calc_theta(self):
        A_inv = np.linalg.inv(self.A)
        self.theta = np.dot(A_inv, self.b)
        return self.theta

    def reward_update(self, reward, x_array):
        # Reshape covariates input into (d x 1) shape vector
        # split_array = np.array_split(x_array, 10)
        # x = split_array[self.chosen_arm][:].reshape([-1, 1])
        x = x_array
        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)

        # Update b which is (d x 1) vector
        # reward is scalar
        self.b += reward * x

    def printBandits(self):
        print("num times selected each bandit:", [b.N for b in self.linucb_arms])

    def calculate_projection(self, u, v):
        v_norm = np.sqrt(sum(v ** 2))
        proj_on_v = (np.dot(u, v) / v_norm ** 2) * v
        return proj_on_v

    def select_arm(self, x_array):
        # Initiate ucb to be 0
        highest_ucb = -10

        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []
        theta = self.calc_theta()

        for arm_index in range(self.K_arms):
            # Calculating projection of user on the arm
            user_context = x_array
            arm_context = self.arm_context[arm_index]
            projection = self.calculate_projection(user_context, arm_context)

            # Calculate ucb based on each arm using current covariates at time t

            arm_ucb = self.linucb_arms[arm_index].calc_UCB(projection, theta,
                                                           self.A)
            # If current arm is highest than current highest_ucb
            if arm_ucb > highest_ucb:
                # Set new max ucb
                highest_ucb = arm_ucb

                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [arm_index]

            # If there is a tie, append to candidate_arms
            if arm_ucb == highest_ucb:
                if arm_index not in candidate_arms:
                    candidate_arms.append(arm_index)

            # if len(candidate_arms) == 0:
            #     print('A', self.A)
            #     print('b', self.b)
            #     print('theta', self.theta)
            #     print('arm_ucb', arm_ucb)

        # Choose based on candidate_arms randomly (tie breaker)
        # print('last step:', candidate_arms)
        chosen_arm = np.random.choice(candidate_arms)
        self.linucb_arms[chosen_arm].update()
        self.chosen_arm = chosen_arm

        return chosen_arm



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


if __name__ == "__main__":
    argv = sys.argv
    # alpha_inputs = ast.literal_eval(argv[1])
    alpha_inputs = [0.1, 0.2, 0.4, 0.5, 0.9]
    data_path = "data/news_dataset.txt"
    for alpha in alpha_inputs:
        print(f"Trying with alpha = {alpha}")
        aligned_time_steps, cum_rewards, aligned_ctr, policy = ctr_simulator(K_arms=10, d=50, alpha=alpha,
                                                                             data_path=data_path)
        print("Cumulative Reward: ", cum_rewards)
        plt.plot(aligned_ctr, label="alpha = " + str(alpha))
    plt.ylabel("CTR ratio (For Single Theta)")
    plt.xlabel("Time")
    plt.legend()
    plt.show()
