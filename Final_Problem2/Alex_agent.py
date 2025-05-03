import numpy as np
import Environment as E

class Agent:
    def __init__(self, environment, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.environment = environment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.zeros((environment.height, environment.width, 4))

    def choose_action(self, state, cur_pos):
        """
        Choose an action using epsilon - greedy policy.
        """
        row, col = E.get_cord(cur_pos, self.environment.inputmap)
        #print(f"row: {row}, col: {col}")

        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, 4)

       # print(f"qtable: {self.q_table[row, col]}")
        return np.argmax(self.q_table[row, col])

    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-table using the Q-learning update rule.
        """
        row, col = E.get_cord(state, self.environment.inputmap)
        #next_row, next_col = next_state
        #print(f"current state: {state}")
        #print(f"next state: {next_state}\n")

        td_target = reward + self.gamma * np.max(self.q_table[row, col][next_state])
        td_error = td_target - self.q_table[row, col, action]

        # Update Q-value
        self.q_table[row, col, action] += self.alpha * td_error








