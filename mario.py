import numpy as np
import gym
from gym import spaces


class CustomGridWorld(gym.Env):
    metadata = {"render.modes": ["human", "console"]}

    def __init__(self):
        super(CustomGridWorld, self).__init__()
        self.n_rows, self.n_cols = 3, 3
        self.state_space = self.n_rows * self.n_cols  # 9 states
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right

        self.grid = np.arange(self.state_space).reshape(self.n_rows, self.n_cols)
        self.state = 0  # Start at state 0 (top-left corner)

        # Rewards and transitions
        self.rewards = np.zeros(self.state_space)  # Zero-based indexing
        self.rewards[2] = 1  # Reward 1 for state 2
        self.rewards[5] = -10  # Reward -10 for state 5

        self.actions = {
            0: -self.n_cols,  # up
            1: self.n_cols,  # down
            2: -1,  # left
            3: 1,
        }  # right

    def step(self, action):
        reward = self.rewards[self.state]
        if self.state == 5 and action == 0:
            if np.random.rand() < 0.2:
                new_state = 1
            else:
                new_state = 2

        # Calculate potential new state based on action
        else:
            new_state = self.state + self.actions[action]

        # Convert state index to row and column to check boundaries
        new_row, new_col = divmod(new_state, self.n_cols)
        if (
            new_row >= 0
            and new_row < self.n_rows
            and new_col >= 0
            and new_col < self.n_cols
        ):
            self.state = new_state

        done = False  # You can define termination condition here

        return self.state, reward, done, {}

    def reset(self):
        self.state = 0  # Reset to the initial state 0
        return self.state

    def render(self, mode="human"):
        if mode == "console":
            grid = np.zeros((self.n_rows, self.n_cols), dtype=int)
            row, col = divmod(self.state, self.n_cols)
            grid[row, col] = 1
            print("Grid World:\n", grid)


action_names = ["Up", "Down", "Left", "Right"]


def q_value_iteration(
    env, discount_factor=0.9, theta=0.0001, with_model=True, max_iter=None
):
    Q = np.zeros((env.state_space, env.action_space.n))  # Initialize Q-values
    policy = np.zeros(env.state_space, dtype=int)
    iter = 0
    while True:
        delta = 0
        current_Q = np.zeros((env.state_space, env.action_space.n))
        for state in range(env.state_space):  # Iterate over all states
            for action in range(env.action_space.n):  # Evaluate each action
                original_state = env.state
                env.state = state
                next_state, reward, done, _ = env.step(action)
                if with_model and state == 5 and action == 0:
                    future_q = max(0.2 * Q[1] + 0.8 * Q[2])
                    print("ahasd")
                    print(future_q)
                else:
                    future_q = max(Q[next_state])

                q = reward + discount_factor * future_q
                print("s, a, s', r, fq, q")
                print(
                    f"{state}, {action_names[action]}, {next_state}, {reward}, {future_q}, {q}"
                )
                delta = max(delta, abs(q - Q[state][action]))
                current_Q[state][action] = q
                Q[state][action] = q
        for action in range(env.action_space.n):
            print(f"Iteration number {iter}")
            print(f"Q-values for action '{action_names[action]}':")
            print(current_Q[:, action].reshape(env.n_rows, env.n_cols))
            print()

        iter += 1

        # Print Q-values in a 3x3 format for each action

        if iter == max_iter:
            break
        if delta < theta:  # Check if the improvement is small enough to stop
            break

    for state in range(env.state_space):
        policy[state] = np.argmax(Q[state])

    return Q, policy


mario = CustomGridWorld()
q_value_iteration(mario, max_iter=2)
