import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from .learning_model import BaseLearningModel

class UCB(BaseLearningModel):
    """A simple tabular R-max agent."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        r_max: float = 1.0,
        m: int = 5,
        discount: float = 0.95,
        seed: int | None = None,
    ):
        """Initializes the R-max agent with the given parameters.

        Creates the necessary data structures for counting visits and transitions,
        as well as storing estimates for rewards and the value function.

        Args:
            num_states (int): Number of states in the environment.
            num_actions (int): Number of possible actions.
            r_max (float, optional): Maximum possible reward for unknown
                state-action pairs. Defaults to 1.0.
            m (int, optional): Minimum number of visits required to consider
                a state-action pair as known. Defaults to 5.
            discount (float, optional): Discount factor. Defaults to 0.95.
            seed (int | None, optional): Seed to ensure reproducibility. Defaults to None.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.m = m
        self.discount = discount
        self.obs_dim = (50, 50, 50)

        self.sa_counts = np.zeros((num_states, num_actions), dtype=np.int32)
        self.reward_sums = np.zeros((num_states, num_actions), dtype=np.float32)

        self.Q = np.full(
            (num_states, num_actions),
            dtype=np.float32,
            fill_value=r_max / (1 - discount),
        )
        np.random.seed(seed)
    
    def learn(self, state, action, reward):
        """Updates the internal model with a new experience.

        Increments visit counts for the given state-action pair, adds
        the observed reward, and updates transition counts.

        Args:
            action (int): Action taken in the current state.
            obs (int): Current state.
            reward (float): Reward received upon transitioning to next_state.
        """
        obs_idx = np.ravel_multi_index(state, self.obs_dim)

        if not self.is_known(obs_idx, action):
            self.sa_counts[obs_idx, action] += 1
            self.reward_sums[obs_idx, action] += reward

            if self.is_known(obs_idx, action):
                r_hat = self.get_reward_estimate(
                    obs=obs_idx, action=action
                )
                self.Q[obs_idx, action] = r_hat

    def is_known(self, obs, action):
        """Determines if a given state-action pair is known.

        A pair is considered known if it has been visited at least m times.

        Args:
            state (int): State of interest.
            action (int): Action of interest.

        Returns:
            bool: True if the pair is known, otherwise False.
        """
        return self.sa_counts[obs, action] >= self.m

    def get_reward_estimate(self, obs, action):
        """Computes the estimated reward for a state-action pair.

        If the pair is known, returns the average observed reward.
        Otherwise, returns r_max.

        Args:
            state (int): State of interest.
            action (int): Action of interest.

        Returns:
            float: Estimated reward for the given state-action pair.
        """
        if self.is_known(obs, action):
            return self.reward_sums[obs, action] / self.sa_counts[obs, action]
        else:
            return self.r_max

    def act(self, obs) -> int:
        """Selects an action based on the current value function.

        Computes the estimated Q-value for each possible action and
        returns the action that yields the highest value.

        Args:
            obs (int): Current state from which to choose an action.

        Returns:
            int: The action that maximizes the estimated Q-value.
        """
        obs_idx = np.ravel_multi_index(obs, self.obs_dim)
        self.last_obs = obs_idx

        q_values = self.Q[obs_idx]
        return np.random.choice(
            np.argwhere(q_values == np.max(q_values)).reshape((-1,))
        )