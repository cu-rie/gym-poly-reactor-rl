import numpy as np


class RandomAgent():
    def __init__(self, state_dim, action_low: list, action_high: list):
        self.action_low = action_low
        self.action_high = action_high

    def get_action(self, state):
        action = np.random.uniform(low=self.action_low, high=self.action_high)
        return action
