import numpy as np
import random
import time


class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1, q_table_size=100):
        """
        Initialize the Q-learning agent.

        Args:
            actions (list): List of actions the agent can take.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
            q_table_size (int): Number of states for the Q-table.
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((q_table_size, len(actions)))

    def load_q_table(self, filename='q_table.npy'):
        """
        Load the Q-table from a file.
        """
        try:
            self.q_table = np.load(filename)
            print(f"Q-table loaded from {filename}")
        except FileNotFoundError:
            print("No saved Q-table found. Starting with a fresh Q-table.")

    def save_q_table(self, filename='q_table.npy'):
        """
        Save the Q-table to a file.
        """
        np.save(filename, self.q_table)
        print(f"Q-table saved to {filename}")

    def choose_action(self, state):
        """
        Choose an action based on the epsilon-greedy strategy.

        Args:
            state (int): The current state.

        Returns:
            int: The index of the chosen action.
        """
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(range(len(self.actions)))
            #print(f"Exploring: Random action chosen: {self.actions[action]}")
        else:
            action = np.argmax(self.q_table[state])
            #print(f"Exploiting: Best action chosen: {self.actions[action]}")
        return action

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning formula.

        Args:
            state (int): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (int): Next state.
        """
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action]
        )
