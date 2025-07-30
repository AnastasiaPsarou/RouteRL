import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from .learning_model import BaseLearningModel

class MBIE(BaseLearningModel):
    """A simple tabular MBIE agent."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        num_agents: int,
        beta: float,
        seed: int | None = None,
    ):
        """Initializes the MBIE agent with the given parameters.

        Creates the necessary data structures for counting visits and transitions,
        as well as storing estimates for rewards and the value function.

        Args:
            num_states (int): Number of states in the environment.
            num_actions (int): Number of possible actions.
            beta (float): Exploration bonus coefficient.
            seed (int | None, optional): Seed to ensure reproducibility. Defaults to None.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.obs_dim = tuple([num_agents] * num_actions)

        self.beta = beta

        self.sa_counts = np.zeros((num_states, num_actions), dtype=np.int32)
        self.reward_sums = np.zeros((num_states, num_actions), dtype=np.float32)

        self.Q = np.full(
            (num_states, num_actions),
            dtype=np.float32,
            fill_value=0,
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

        self.sa_counts[obs_idx, action] += 1
        self.reward_sums[obs_idx, action] += reward

        r_hat = self.get_reward_estimate(
            obs=obs_idx, action=action
        )

        stability_coeff = 0.001 # avoid division by zero
        exploration_bonus = 1 / np.sqrt(self.sa_counts[obs_idx, action] + stability_coeff) 
        self.Q[obs_idx, action] = r_hat + self.beta * exploration_bonus

    def get_reward_estimate(self, obs, action):
        """Computes the estimated reward for a state-action pair.

        Args:
            state (int): State of interest.
            action (int): Action of interest.

        Returns:
            float: Estimated reward for the given state-action pair.
        """
        return self.reward_sums[obs, action] / self.sa_counts[obs, action]


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