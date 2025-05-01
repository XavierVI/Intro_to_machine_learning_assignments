import numpy as np
from scipy.integrate import odeint
import pyprind

import matplotlib.pyplot as plt
"""

"""


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

        self.Q = np.zeros((self.X.size, self.V.size, self.U.size))

    
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
        if np.random.default_rng().uniform() < self.epsilon:
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

    def get_reward(self, s, u):
        """
        This function goes through a sequence of steps to accumulate
        the reward, such that we punish the agent when preventing the
        state from being (0, 0) and reward it when it is at or reaching
        (0, 0).
        """
        reward = 0
        x, v = s
        # next_x, next_v = next_s

        goal_reward = 10
        velocity_reward = 0.1 * abs(v)
        position_reward = 0.5 * abs(x)
        control_reward  = 0.01 * abs(u)

        # Reward for reaching the goal
        if x == 0.0 and v == 0.0:
            reward += goal_reward

        # encourage moving towards the goal
        if x != 0 and np.sign(x) != np.sign(v):
            reward += velocity_reward

        if v != 0 and np.sign(v) != np.sign(u):
            reward += control_reward

        if abs(x) < 0.1 and abs(v) < 0.:
            reward += 5

        # Punish for being very far away from the goal
        if abs(x) > 5:
            reward -= goal_reward

        # punish for moving away from the goal
        if x != 0 and np.sign(x) == np.sign(v):
            reward -= position_reward

        if x == 0 and (v != 0 or u != 0):
            reward -= position_reward

        if v != 0 and np.sign(v) == np.sign(u):
            reward -= control_reward
        
        if abs(x) < 0.5 and abs(u) > 0.5:
            reward -= control_reward

        return reward


    def generate_trajectory(self, s0, T, agent: Agent):
        """
        This function generates a trajectory of states s = (x, v), using
        a Q table to select the control input at each time step.

        s0: the initial state
        T: the final time step such that s(t) is the terminal state
        Q: the Q-table
        """
        prev_s = s0

        # compute the next control input
        prev_u = agent.get_best_action(prev_s)

        # compute the next reward outside of the loop
        R = self.get_reward(prev_s, prev_u)
        
        # let the trajectory be represented as a T x 3 array where
        # each entry is (s, u, r)
        # where s = (x, y) is the state,
        # u is the control input, and r is the reward
        trajectory = np.zeros((T, 4))
        trajectory[0] = (prev_s[0], prev_s[1], prev_u, R)

        for i in range(1, T):
            # compute the next state
            s_i = self.get_next_state(prev_s, prev_u)
            u_i = agent.get_best_action(s_i)
            R = self.get_reward(s_i, u_i)

            # add the new sequence to the trajectory
            trajectory[i] = [s_i[0], s_i[1], u_i, R]

            # early stopping condition
            if s_i[0] == 0 and s_i[1] == 0:
                # if the car is at the goal, break
                break

            elif abs(s_i[0]) >= 5 or abs(s_i[1]) >= 5:
                # if the car is out of bounds, break
                break

            # update the previous state and control input
            prev_s = s_i
            prev_u = u_i

        return trajectory
            

def Q_learning(agent: Agent, env: CarModel, num_episodes=50, time_steps=10):
    """
    This function is an implementation of the Q-learning algorithm.

    num_episodes: the number of episodes to run
    
    """
    history = np.zeros((num_episodes,))
    pbar = pyprind.ProgBar(num_episodes, title="Training", width=40)

    for episode in range(num_episodes):
        # randomly select the initial state from X and V
        random_idx1 = np.random.default_rng(1).choice(agent.X.size)
        random_idx2 = np.random.default_rng(1).choice(agent.V.size)
        s_i = (agent.X[random_idx1], agent.V[random_idx2])

        final_reward = 0

        for t in range(time_steps):
            u = agent.get_best_action(s_i)
            next_s = agent.discretize_state(env.get_next_state(s_i, u))
            reward = env.get_reward(s_i, u)

            # print(f"Episode {episode}, Time step {t}: s = {s_i}, u = {u}, r = {reward}")
            agent.update_Q(s_i, u, reward, next_s)
            
            s_i = next_s
            final_reward += reward

            if s_i[0] == 0 and s_i[1] == 0:
                # if the car is at the goal, break
                break

            elif abs(s_i[0]) >= 5 or abs(s_i[1]) >= 5:
                # if the car is out of bounds, break
                break
        
        history[episode] = final_reward
        pbar.update(1)

    pbar.stop()

    return history


def evaluate_policy(agent: Agent, env: CarModel, num_episodes=10, time_steps=10):
    """
    This function evaluates the policy learned by the agent.
    It runs the agent for a number of episodes and returns the
    average reward.
    """
    # stores the average reward for each episode
    # and the number of steps taken
    history = np.zeros((num_episodes,))
    pbar = pyprind.ProgBar(num_episodes, title="Evaluating", width=40)

    fig, axes = plt.subplots(1, 2, figsize=(10, 8))

    for episode in range(num_episodes):
        # randomly select the initial state from X and V
        random_idx1 = np.random.default_rng(1).choice(agent.X.size)
        random_idx2 = np.random.default_rng(1).choice(agent.V.size)
        s0 = (agent.X[random_idx1], agent.V[random_idx2])
        axes[0].plot(s0[1], s0[0], "bo", label="s0 episode" + str(episode))

        # generate the trajectory
        trajectory = env.generate_trajectory(s0, time_steps, agent)
        axes[0].plot(trajectory[:, 1], trajectory[:, 0], label="episode" + str(episode))
        
        history[episode] = np.sum(trajectory[:, 3])
        pbar.update(1)

    pbar.stop()

    axes[0].plot(0, 0, "ro", label="Goal")
    axes[0].set_xlabel("v")
    axes[0].set_ylabel("x")
    axes[0].set_title("Trajectory")
    # axes[0].legend(["episode" + str(i) for i in range(num_episodes)] + ["Goal"])
    axes[0].grid()

    axes[1].plot(history)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Reward History")
    axes[1].grid()

    plt.suptitle("Evaluation for Q Table", fontsize=16)
    plt.tight_layout()
    # plt.savefig(fname="./problem1_figs/problem1-3.png", dpi=300)
    plt.show()

    return history


        
time_steps = 100
num_episodes = 100

agent = Agent(
    alpha=0.1,
    gamma=0.9,
    epsilon=0.4
)
environment = CarModel()
history = Q_learning(
    agent,
    environment,
    num_episodes=num_episodes,
    time_steps=time_steps
)

# randomly select the initial state from X and V
random_idx1 = np.random.default_rng(1).choice(agent.X.size)
random_idx2 = np.random.default_rng(1).choice(agent.V.size)
s0 = (agent.X[random_idx1], agent.V[random_idx2])

eval_history = evaluate_policy(
    agent,
    environment,
    num_episodes=3,
    time_steps=50
)


print("Average reward per episode: ", eval_history[:])
