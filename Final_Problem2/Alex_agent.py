import numpy as np

class Agent:
    def __init__(self, environment, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.environment = environment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.zeros((environment.height, environment.width, 4))

        def choose_action(self, state):
            """
            Choose an action using epsilon - greedy policy.
            """
            row, col = state

            if np.random.uniform() < self.epsilon:
                return np.random.randint(0, len(self.env.actions))

            return np.argmax(self.q_table[row, col])

        def update_q_table(self, state, action, reward, next_state):
            """
            Update Q-table using the Q-learning update rule.
            """
            row, col = state
            next_row, next_col = next_state

            td_target = reward + self.gamma * np.max(self.q_table[next_row, next_col, :])

            td_error = td_target - self.q_table[row, col, action]

            # Update Q-value
            self.q_table[row, col, action] += self.alpha * td_error








