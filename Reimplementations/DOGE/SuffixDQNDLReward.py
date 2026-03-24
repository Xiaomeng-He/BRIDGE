import torch
import gymnasium as gym
from stable_baselines3 import PPO

from BatchedEvaluationCallback import BatchedEvaluationCallback
from SuffixEnvDLRewardOnlyWhenPredicted import SuffixEnvDLRewardOnlyWhenPredicted
from SimpleResourceExtractor import SimpleResourceExtractor
from utils import levenshtein_similarity, one_hot_windowed_sequence
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class SuffixDQNDLReward:
    def __init__(self, 
                 log_info, 
                 policy, 
                 epochs, 
                 window_size, 
                 reward_strategy, 
                 feature_extractor, 
                 checkpoint_path,
                 seed,
                 autoencoder_model=None, 
                 is_synthetic=False, 
                 check_compliance_testing=False):
        self.log_info = log_info
        self.policy = policy
        self.epochs = epochs
        self.window_size = window_size
        self.reward_strategy = reward_strategy
        self.checkpoint_path = checkpoint_path
        self.seed = seed
        self.pad = int(0)
        self.eoc = int(3)
        self.feature_extractor = feature_extractor
        self.autoencoder_model = autoencoder_model
        self.is_synthetic = is_synthetic
        self.check_compliance_testing = check_compliance_testing

    def train(self, X_train, y_train, X_val, y_val):
        num_timestamps = (len(X_train) * self.epochs) * 3
        total_timestamps = min(25_000_000, num_timestamps)
        
        if self.reward_strategy == "dl":
            SuffixEnvironment = SuffixEnvDLRewardOnlyWhenPredicted

        train_env = SuffixEnvironment(X_train, self.window_size, y_train, self.log_info)
        val_env = SuffixEnvironment(X_val, self.window_size, y_val, self.log_info)

        validation_environment = val_env

        train_env.seed(self.seed)

        policy = PPO

        eval_callback = BatchedEvaluationCallback(val_env, 
                                                X_val, 
                                                y_val, 
                                                self.log_info, 
                                                self.window_size, 
                                                self.checkpoint_path, 
                                                int(len(X_train) / 2), 
                                                validation_environment, 
                                                is_maskable=self.policy=="MaskablePPO", 
                                                is_recurrent=self.policy=="RecurrentPPO", 
                                                verbose=0, 
                                                is_synthetic=self.is_synthetic)

        policy_kwargs = {"features_dim" : 1, 
                         "log_info" : self.log_info, 
                         "window_size" : self.window_size}
        if self.feature_extractor == "flatten":
            feature_extractor_class = SimpleResourceExtractor

        arguments = {
                "features_extractor_class" : feature_extractor_class,
                "features_extractor_kwargs" : policy_kwargs
            }
        self.model = policy("MultiInputPolicy", 
                            env=train_env, 
                            verbose=0, 
                            policy_kwargs=arguments)
        self.model.learn(total_timesteps=total_timestamps, 
                         progress_bar=False, 
                         callback=eval_callback)

    def test(self, X_test, y_test, best_model_evaluation="reward"):
        print(f"Testing {best_model_evaluation}...")

        similarities = []
        mae_len = []
        all_predicted_suffixes = [] 

        self.model = self.model.load(os.path.join(self.checkpoint_path, 
                                                  f"best_model_{best_model_evaluation}.zip"))

        for i in range(len(X_test)):
            prefix_len = sum(1 for act in X_test[i][0] if act != self.pad)
            current_suffix = one_hot_windowed_sequence(
                X_test[i][0], self.window_size, self.log_info["n_activities"]
            )
            predicted_suffix = []
            action = None

            obs = {"activity_prefix" : X_test[i][0], 
                   "suffix" : current_suffix, 
                   "time_since_prev_prefix" : X_test[i][-2], 
                   "time_since_start_prefix" : X_test[i][-1]}
            for attr_id, attribute in enumerate(self.log_info["attributes"]):
                obs[f"{attribute}_prefix"] = X_test[i][attr_id+1]

            current_predicted_trace = X_test[i][0]

            while (len(predicted_suffix) + prefix_len) < self.log_info["max_trace_length"] and action != self.eoc:

                action, _states = self.model.predict(obs, deterministic=True)

                action += 1

                pred = np.array([[0] * self.log_info["n_activities"]])
                pred[0][action] = 1
                current_suffix = np.concatenate((current_suffix[1:], pred))
                obs["suffix"] = current_suffix
                predicted_suffix.append(action)

                current_predicted_trace = np.concatenate((current_predicted_trace[1:], [action]))

            all_predicted_suffixes.append(predicted_suffix)  # <-- store it
            pred_len, tgt_len, similarity = levenshtein_similarity(predicted_suffix, y_test[i], code_end=self.eoc)
            similarities.append(similarity)
            mae_len.append(abs(pred_len - tgt_len))

        print(f"Mean damerau {np.mean(similarities)}")

        return np.mean(similarities), np.mean(mae_len), all_predicted_suffixes

