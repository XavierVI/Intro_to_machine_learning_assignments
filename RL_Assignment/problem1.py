import numpy as np
from scipy.integrate import odeint
import pyprind

import matplotlib.pyplot as plt

import time

class Agent:
    """
    This class defines the agent for problem 1.
    """

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, state_space_size=21):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # state space and action space
        self.X = np.linspace(-5, 5, state_space_size)
        self.V = np.linspace(-5, 5, state_space_size)
        self.U = np.array([-5, -1, -0.1, -0.01, -0.001, -0.0001, 0,
                           0.0001, 0.001, 0.01, 0.1, 1, 5])

        # The indices for the Q table are retrieved from these dictionaries
        # this one can be used for either x or v
        self.s_to_index = {x_val: i for i, x_val in enumerate(self.X)}
        self.u_to_index = {u_val: i for i, u_val in enumerate(self.U)}

        self.Q = np.random.default_rng(1).uniform(
            size=(self.X.size, self.V.size, self.U.size)
        )

    
    def discretize_state(self, s):
        """
        Discretizes the state s = (x, v) to the nearest values in X and V.
        """
        # clip values to keep them within -5 and 5
        x = s[0]
        v = s[1]
        x_discrete = self.X[np.argmin(np.abs(self.X - x))]
        v_discrete = self.V[np.argmin(np.abs(self.V - v))]
        return (x_discrete, v_discrete)


    def get_best_action(self, s):
        """
        This function computes the best action for a given state s
        using the Q table.
        """
        # for the epsilon-greedy policy, we select a random action
        # with probability epsilon
        if np.random.default_rng(1).uniform() < self.epsilon:
            return np.random.choice(self.U)

        # transform the state into one of our discrete states
        s = self.discretize_state(s)
        x_idx = self.s_to_index[s[0]]
        v_idx = self.s_to_index[s[1]]
        q_values = self.Q[x_idx, v_idx, :]

        best_action_idx = np.argmax(q_values)
        best_action = self.U[best_action_idx]

        return best_action


    def update_Q(self, s, u, r, next_s):
        """
        This function updates the Q table using the update method
        from the slides.

        s: the state when the action was made
        u: the action taken
        r: the reward received
        next_s: the state after the action was taken
        """
        x_idx = self.s_to_index[s[0]]
        v_idx = self.s_to_index[s[1]]
        u_idx = self.u_to_index[u]
        next_x_idx = self.s_to_index[next_s[0]]
        next_v_idx = self.s_to_index[next_s[1]]

        # epsilon decay
        self.epsilon -= 0.01

        # get the maximum Q value for the next state
        max_next_s = np.max(
            self.Q[next_x_idx, next_v_idx, :]
        )
        
        # update the Q table
        self.Q[x_idx, v_idx, u_idx] += \
            self.alpha * (r + self.gamma * max_next_s - self.Q[x_idx, v_idx, u_idx])



class CarModel:
    """
    This class defines the car for problem 1.
    """

    def __init__(self):
        # size of each time step
        self.delta = 0.1
        self.step = np.linspace(0, self.delta)

    def model(self, s, t, u):
        dsdt = [ s[1], u  ]
        return dsdt

    def get_next_state(self, s, u):
        """
        This function computes the next state of the car given the
        current state and the control input.
        """
        # compute the next state
        y = odeint(self.model, s, self.step, args=(u,))
        return y[-1]


    def get_reward(self, s, u, next_s):
        x, v = s
        next_x, next_v = next_s

        reward = 0
        goal_reward = 10

        # Penalties for position, velocity, and control input
        # These penalize the agent for being away from the goal
        position_penalty = -0.2 * abs(next_x)
        velocity_penalty = -0.05 * abs(next_v)
        control_penalty = -0.01 * abs(u)

        # Reward for reaching the goal (using a threshold)
        if abs(next_x) < 0.1 and abs(next_v) < 0.1:
            reward += goal_reward

        reward += position_penalty + velocity_penalty + control_penalty

        # Encourage moving towards zero
        if x * next_x < 0:  # x changed sign (crossed zero)
            reward += 1
        if v * next_v < 0:  # v changed sign (crossed zero)
            reward += 0.5

        # Out-of-bounds penalty (softer)
        if abs(next_x) > 5 or abs(next_v) > 5:
            reward -= 5

        return reward


    def generate_trajectory(self, s0, T, agent: Agent):
        """
        This function generates a trajectory of states s = (x, v), using
        a Q table to select the control input at each time step.

        s0: the initial state
        T: the final time step such that s(t) is the terminal state
        Q: the Q-table
        """
        # let the trajectory be represented as a T x 3 array where
        # each entry is (s, u, r)
        # where s = (x, y) is the state,
        # u is the control input, and r is the reward
        trajectory = np.zeros((T, 4))
        prev_s = s0
        # compute the next control input
        prev_u = agent.get_best_action(prev_s)
        
        for i in range(0, T):
            next_s = self.get_next_state(prev_s, prev_u)
            # only used for the next iteration
            next_u = agent.get_best_action(next_s)

            # compute the reward using the current state and control input
            # and the next state
            R = self.get_reward(prev_s, prev_u, next_s)

            # add the new sequence to the trajectory
            trajectory[i] = [prev_s[0], prev_s[1], prev_u, R]

            # update the previous state and control input
            # for the next iteration
            prev_s = next_s
            prev_u = next_u

        return trajectory
            

def Q_learning(agent: Agent, env: CarModel, num_episodes=50, time_steps=10):
    """
    This function is an implementation of the Q-learning algorithm.

    num_episodes: the number of episodes to run
    
    """
    history = []
    pbar = pyprind.ProgBar(num_episodes, title="Optimizing policy...", width=40)

    for episode in range(num_episodes):
        # randomly select the initial state from X and V
        random_idx1 = np.random.default_rng(1).choice(agent.X.size)
        random_idx2 = np.random.default_rng(1).choice(agent.V.size)
        s_i = (agent.X[random_idx1], agent.V[random_idx2])

        final_reward = 0

        for t in range(time_steps):
            u = agent.get_best_action(s_i)
            next_s = agent.discretize_state(env.get_next_state(s_i, u))
            reward = env.get_reward(s_i, u, next_s)

            # print(f"Episode {episode}, Time step {t}: s = {s_i}, u = {u}, r = {reward}")
            agent.update_Q(s_i, u, reward, next_s)
            
            s_i = next_s
            final_reward += reward
        
        history.append(final_reward)
        pbar.update(1)

    pbar.stop()

    return history


def evaluate_policy(
        agent: Agent,
        env: CarModel,
        num_episodes=10,
        time_steps=10,
        plot_figure=True
    ):
    """
    This function evaluates the policy learned by the agent.
    It runs the agent for a number of episodes and returns the
    average reward.
    """
    # stores the average reward for each episode
    # and the number of steps taken
    # history = np.array([])
    pbar = pyprind.ProgBar(num_episodes, title="Evaluating policy...", width=40)

    # fig = plt.figure(figsize=(12, 6))

    # Top row - two subplots side by side
    # ax1 = fig.add_subplot(2, 2, 1)  # 2 rows, 2 columns, first subplot
    # ax2 = fig.add_subplot(2, 2, 2)  # 2 rows, 2 columns, second subplot

    # Bottom row - one subplot spanning the width
    # ax3 = fig.add_subplot(2, 1, 2)  # 2 rows, 1 column, second subplot (spanning)

    # average distance and reward for each episode
    stats = np.zeros((num_episodes, 4))

    for episode in range(num_episodes):
        # randomly select the initial state from X and V
        random_idx1 = np.random.default_rng(1).choice(agent.X.size)
        random_idx2 = np.random.default_rng(1).choice(agent.V.size)
        s0 = (agent.X[random_idx1], agent.V[random_idx2])
        # generate the trajectory
        trajectory = env.generate_trajectory(s0, time_steps, agent)

        # accumulating the distance from the desired state
        stats[episode, 0] = np.mean(np.abs(trajectory[:, 0]))
        stats[episode, 1] = np.std(trajectory[:, 0])
        stats[episode, 2] = np.mean(np.abs(trajectory[:, 3]))
        stats[episode, 3] = np.std(trajectory[:, 3])
        
        # plotting
        # ax1.plot(range(time_steps), trajectory[:, 0], label="episode " + str(episode))
        # ax2.plot(range(time_steps), trajectory[:, 1], label="episode " + str(episode))

        # ax3.plot(trajectory[:, 3], '-', label="episode " + str(episode))
                
        pbar.update(1)

    pbar.stop()

    # ax1.set_xlabel("t")
    # ax1.set_ylabel("x")
    # ax1.set_ylim(-5, 5)
    # ax1.set_title("Position over time")
    # ax1.grid()
    # ax1.legend(loc="lower left")
    
    # ax2.set_xlabel("t")
    # ax2.set_ylabel("v")
    # ax2.set_ylim(-5, 5)
    # ax2.set_title("Velocity over time")
    # ax2.grid()
    # ax2.legend(loc="lower left")

    # ax3.set_xlabel("Time step (t)")
    # ax3.set_ylabel("Reward (r)")
    # ax3.set_title("Reward history per episode")
    # ax3.legend(loc="lower left")
    # ax3.grid()

    # plt.tight_layout()
    # plt.savefig(fname=f"./problem1_figs/problem1_{file_header}.png", dpi=300)
    # if plot_figure:
    #     plt.show()

    return stats

def length_of_time_steps():
    time_steps = [20, 50, 100, 500]
    environment = CarModel()
    table = []

    agent = Agent(
        alpha=0.01,
        gamma=0.9,
        # use the best state space size
        state_space_size=51
    )
    _ = Q_learning(
        agent,
        environment,
        num_episodes=500,
        time_steps=200
    )

    for time_step in time_steps:
        
        start = time.time()
        stats = evaluate_policy(
            agent,
            environment,
            num_episodes=4,
            time_steps=time_step,
            plot_figure=False
        )
        end = time.time()

        for [avg_dist, std_dist, avg_reward, std_reward] in stats:
            table.append(
                f"{time_step} & {avg_dist:.2f} ± {std_dist:.2f} & {avg_reward:.2f} ± {std_reward:.2f} & {end - start:.2f} \\\\")
    print("Time steps Test Accuracy Table")
    print("=============================================")
    for row in table:
        print(row)


def test_episodes_and_time_steps():
    episodes_and_time_steps = [
        (100, 50),
        (200, 100),
        (500, 200),
    ]
    environment = CarModel()
    table = []

    for (episodes, time_steps) in episodes_and_time_steps:
        agent = Agent(
            alpha=0.01,
            gamma=0.9,
            # use the best state space size
            state_space_size=51
        )
        start = time.time()
        _ = Q_learning(
            agent,
            environment,
            num_episodes=episodes,
            time_steps=time_steps
        )
        end = time.time()
        stats = evaluate_policy(
            agent,
            environment,
            num_episodes=4,
            time_steps=20,
            plot_figure=False
        )
        
        for [avg_dist, std_dist, avg_reward, std_reward] in stats:
            table.append(f"{episodes} & {time_steps} & {avg_dist:.2f} ± {std_dist:.2f} & {avg_reward:.2f} ± {std_reward:.2f} & {end - start:.2f} \\\\")

    for row in table:
        print(row)


def test_state_space():
    state_space_sizes = [21, 51, 101]
    environment = CarModel()
    table = []

    # Changing the size of the state space
    for state_space_size in state_space_sizes:
        agent = Agent(
            alpha=0.01,
            gamma=0.9,
            state_space_size=state_space_size
        )
        _ = Q_learning(
            agent,
            environment,
            num_episodes=500,
            time_steps=200
        )
        stats = evaluate_policy(
            agent,
            environment,
            num_episodes=4,
            time_steps=20,
            plot_figure=False
        )
        for avg_dist, std_dist, avg_reward, std_reward in stats:
            table.append(
                f"{state_space_size} & {avg_dist:.2f} ± {std_dist:.2f} & {avg_reward:.2f} ± {std_reward:.2f} \\\\")

    print("State space table")
    print("=============================================")
    for row in table:
        print(row)


def test_hyperparams():
    alphas = [0.001, 0.1]
    gammas = [0.3, 0.9]
    environment = CarModel()
    table = []

    
    for a in alphas:
        for g in gammas:
            agent = Agent(
                alpha=a,
                gamma=g,
                state_space_size=51
            )
            _ = Q_learning(
                agent,
                environment,
                num_episodes=500,
                time_steps=200
            )
            stats = evaluate_policy(
                agent,
                environment,
                num_episodes=4,
                time_steps=20,
                plot_figure=False
            )
            for avg_dist, std_dist, avg_reward, std_reward in stats:
                table.append(
                    f"{a} & {g} & {avg_dist:.2f} ± {std_dist:.2f} & {avg_reward:.2f} ± {std_reward:.2f} \\\\")

    print("Hyperparameters table")
    print("=============================================")
    for row in table:
        print(row)


# warm up run
print('--------------------------------------')
print('WARM UP RUN')
print('--------------------------------------')
agent = Agent()
env = CarModel()
evaluate_policy(agent, env)
print('--------------------------------------')
print('END OF WARM UP RUN')
print('--------------------------------------')

length_of_time_steps()
test_episodes_and_time_steps()
test_state_space()
test_hyperparams()



