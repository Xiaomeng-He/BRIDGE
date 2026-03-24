import numpy as np
import torch

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from utils import levenshtein_similarity, one_hot_windowed_sequence


class SuffixEnvDLRewardOnlyWhenPredicted(gym.Env):
    def __init__(self, sequences, window_size, gt_suffixes, log_info):
        self.log_info = log_info

        self.n_activities = log_info["n_activities"]
        self.eoc = int(3)
        self.max_trace_length = log_info["max_trace_length"]

        self.sequences = sequences
        self.window_size = window_size

        self.action_space = spaces.Discrete(self.n_activities - 1)
        attributes = log_info["attributes"]
        observation_space_dict = {
            "activity_prefix": spaces.Box(low=0, high=self.n_activities, shape=(self.max_trace_length,),
                                          dtype=np.int32),
            "suffix": spaces.Box(low=0, high=1, shape=(self.window_size, self.n_activities)),
            "time_since_prev_prefix": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_trace_length,), dtype=np.float32),
            "time_since_start_prefix": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_trace_length,), dtype=np.float32),
        }
        for attribute in attributes:
            observation_space_dict[attribute + "_prefix"] = spaces.Box(low=0, high=log_info[f"n_{attribute}"],
                                                                  shape=(self.max_trace_length,), dtype=np.int32)

        self.observation_space = spaces.Dict(observation_space_dict)
        self.gt_suffixes = gt_suffixes


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.curr_idx = int(self.np_random.integers(0, len(self.sequences)))

        self.initial_observation = self.sequences[self.curr_idx]
        self.len_prefix = len(self.initial_observation)
        self.expected_suffix = self.gt_suffixes[self.curr_idx]

        self.predicted_suffix = []

        self.current_activity_prefix = self.initial_observation[0]
        self.time_since_prev_prefix = self.initial_observation[-2]
        self.time_since_start_prefix = self.initial_observation[-1]

        self.current_sequence = one_hot_windowed_sequence(
            self.current_activity_prefix, self.window_size, self.n_activities
        )

        obs = {"activity_prefix" : self.current_activity_prefix, 
                "suffix" : self.current_sequence,
                "time_since_prev_prefix" : self.time_since_prev_prefix, 
                "time_since_start_prefix" : self.time_since_start_prefix}

        for i, attribute in enumerate(self.log_info["attributes"]):
            obs[attribute + "_prefix"] = self.initial_observation[i+1]

        self.returned_obs = obs

        info = {}

        return obs, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _calculate_reward(self, predicted_suffix, expected_suffix, current_activity_prefix=None):
        reward = levenshtein_similarity(predicted_suffix, expected_suffix, self.eoc)[2]
        return reward

    def step(self, action):
        action = int(action) + 1 

        self.predicted_suffix.append(action)

        terminated = False
        truncated = False
        reward = 0.0
        info = {"predicted_suffix": self.predicted_suffix}

        if action == self.eoc:
            terminated = True
            reward = self._calculate_reward(self.predicted_suffix, self.expected_suffix)

        elif (len(self.predicted_suffix) + self.len_prefix) >= self.max_trace_length:
            truncated = True
            reward = self._calculate_reward(self.predicted_suffix, self.expected_suffix)

        else:
            pred = np.zeros((1, self.n_activities), dtype=np.float32)
            pred[0, action] = 1
            self.current_sequence = np.concatenate((self.current_sequence[1:], pred), axis=0)
            self.returned_obs["suffix"] = self.current_sequence

        return self.returned_obs, reward, terminated, truncated, info
