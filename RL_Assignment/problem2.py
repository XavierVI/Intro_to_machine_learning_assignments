import pyprind

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image
import matplotlib.animation as animation

def load_maps(map):
    if map == 1:
        map_1 = Image.open(
            # 532 x 528
            "./maps/map1_compressed(48,48).bmp")
        bw_img1 = map_1.convert('1')  # Convert to 1-bit black & white
        grid1 = np.array(bw_img1)
        print(grid1.shape)
        return grid1

    map_2 = Image.open(
        # 532 x 528
        "./maps/map2_compressed(48,48).bmp")
    bw_img2 = map_2.convert('1')  # Convert to 1-bit black & white

    grid2 = np.array(bw_img2)

    print(grid2.shape)

    return grid2


class Environment:
    def __init__(self, grid, start, goal):
        # the map
        self.grid = grid
        # the starting position of the agent
        self.start = start
        # the goal position
        self.goal = goal
        # used to track the last 20 steps
        self.state_history = []

        self.state = start
        self.actions = {
            # up
            0: (-1, 0),
            # down
            1: (1, 0),
            # right
            2: (0, 1),
            # left
            3: (0, -1)
        }

    def reset(self):
        self.state = self.start
        self.state_history.clear()
        return self.state

    def get_reward(self, state, action, next_state):
        reward = 0
        x, y = state
        next_x, next_y = next_state
        goal_x, goal_y = self.goal

        # compute manhattan distances
        current_distance = abs(goal_x - x) + abs(goal_y - y)
        next_distance = abs(goal_x - next_x) + abs(goal_y - next_y)

        if next_distance < current_distance:
            reward += 1
        elif next_distance > current_distance:
            reward -= 1
        # elif next_distance == current_distance and next_state != state:
        #     reward -= 0.1

        if next_state == self.goal:
            reward += 10

        if next_state in self.state_history:
            reward -= 0.1

        return reward

    def step(self, action):
        dx, dy = self.actions[action]
        x, y = self.state
        delta_x = x + dx
        delta_y = y + dy
        done = False
        reward = -0.1
        

        # Check bounds and for wall
        if self.is_valid_move(delta_x, delta_y):
            reward += self.get_reward(self.state, (dx, dy), (delta_x, delta_y))
            self.state = (delta_x, delta_y)
            if self.state == self.goal:
                print('REACHED GOAL!')
                done = True
        else:
            reward -= 10
            done = True

        self.state_history.append(self.state)
        if len(self.state_history) > 20:
            self.state_history.pop(0)

        return self.state, reward, done

    def is_valid_move(self, x, y):
        in_bounds = 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]
        return in_bounds and self.grid[x, y] == 1  # 1 means free



class Agent:
    def __init__(self, environment: Environment, epsilon=0.9):
        self.environment = environment
        self.epsilon = epsilon

        self.q_table = np.zeros(
            (environment.grid.shape[0], environment.grid.shape[1], len(environment.actions)))

    def choose_action(self, state):
        """
        Choose an action using epsilon - greedy policy.
        """
        row, col = state

        if np.random.default_rng(1).uniform() < self.epsilon:
            return np.random.randint(0, len(self.environment.actions))

        return np.argmax(self.q_table[row, col, :])

    def step(self, alpha=0.01, gamma=0.4):
        # get the current state from the environment
        state = self.environment.state
        # choose the direction
        action = agent.choose_action(state)
        next_s, reward, done = env.step(action)

        agent.update_q_table(
            state=state,
            action=action,
            reward=reward,
            next_state=next_s,
            alpha=alpha,
            gamma=gamma
        )

        # reduce epsilon
        self.epsilon -= 0.01

        return next_s, reward, done

    def update_q_table(self,
        state,
        action,
        reward,
        next_state,
        alpha,
        gamma
    ):
        """
        Update Q-table using the Q-learning update rule.
        """
        row, col = state
        next_row, next_col = next_state

        td_target = reward + gamma * \
            np.max(self.q_table[next_row, next_col, :])

        td_error = td_target - self.q_table[row, col, action]

        # Update Q-value
        self.q_table[row, col, action] += alpha * td_error


def Q_learning(
        agent: Agent,
        env: Environment,
        num_episodes=50,
        max_steps=10,
        alpha=0.01,
        gamma=0.9
    ):
    """
    This function is an implementation of the Q-learning algorithm.

    num_episodes: the number of episodes to run
    
    """
    history = []
    trajectories = []
    pbar = pyprind.ProgBar(
        num_episodes, title="Optimizing policy...", width=40)

    for episode in range(num_episodes):
        # reset environment to initial state
        env.reset()
        rewards = []
        num_steps = 0
        trajectory = []
        done = False

        for step in range(max_steps):
            next_s, reward, done = agent.step(
                alpha=alpha,
                gamma=gamma
            )

            rewards.append(reward)
            num_steps += 1
            trajectory.append(next_s)

            # break out of loop if we reached the goal
            if done: break

        reached_goal = env.state == env.goal
        history.append((num_steps, reached_goal, np.mean(rewards), np.std(rewards)))
        trajectories.append(trajectory)
        pbar.update(1)

    pbar.stop()

    return history, trajectories


def animate_trajectory(grid, trajectory, start, goal):
    """
    Displays an animation of the agent moving through the map.

    Args:
        grid (numpy.ndarray): The environment grid.
        trajectory (list): A list of (row, column) tuples representing the agent's path.
        start (tuple): The starting position of the agent.
        goal (tuple): The goal position.
    """
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray')
    # the y-axis is from 0 - 48 top to bottom by default
    ax.plot(start[0], 48 - start[1], 'go', markersize=2, label='Start')
    ax.plot(goal[0], 48 - goal[1], 'ro', markersize=2, label='Goal')
    # ax.grid(True)
    ax.set_title("Agent's Trajectory")
    ax.legend()

    agent_path = ax.plot([], [], 'bo', markersize=2)[0]

    def update(frame):
        if frame < len(trajectory):
            x = [x for (x, y) in trajectory[:frame]]
            y = [48 - y for (x, y) in trajectory[:frame]]
            
            agent_path.set_xdata(x)
            agent_path.set_ydata(y)
        return (agent_path,)

    ani = animation.FuncAnimation(fig, update, frames=len(
        trajectory), interval=30, blit=True, repeat=False)
    plt.show()


map1 = load_maps(map=1)
env = Environment(map1, (4, 4), (30, 40))
agent = Agent(env, epsilon=0.85)
history, trajectories = Q_learning(
    agent=agent,
    env=env,
    num_episodes=1000,
    max_steps=200
)

# Animate the first trajectory generated
animate_trajectory(map1, trajectories[-1], (4, 4), (30, 40))

for steps, done, avg_reward, reward_std in history:
    print(f'{steps} & {done} & {avg_reward:.2f} ± {reward_std:.2f} \\\\')
