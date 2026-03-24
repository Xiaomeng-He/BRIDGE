from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from utils import one_hot_windowed_sequence, levenshtein_similarity
import os
import torch

class BatchedEvaluationCallback(BaseCallback):
    def __init__(self, val_env, X_val, y_val, log_info, window_size, 
                 checkpoint_path, eval_freq, validation_environment, 
                 is_maskable, is_recurrent, verbose, is_synthetic):
        super(BatchedEvaluationCallback, self).__init__(verbose)
        self.val_env = val_env
        self.X_val = X_val
        self.log_info = log_info
        self.y_val = y_val
        self.window_size = window_size
        self.checkpoint_path = checkpoint_path
        self.eval_freq = eval_freq
        self.best_similarity = 0
        self.best_reward = 0
        self.is_maskable = is_maskable
        self.is_recurrent = is_recurrent
        self.eoc = int(3)

        self.validation_environment = validation_environment

        self.no_improve_count = 0
        self.patience_evals = 40

    def _on_step(self) -> bool:
        batch_size = 128
        if self.n_calls % self.eval_freq == 0:
            print("Beginning evaluation...")
            similarities = []
            rewards = []
            predicted_lens = []
            for i in range(0, len(self.X_val), batch_size):

                # Initialize the observation
                obs = {
                    "activity_prefix" : [],
                    "suffix" : [],
                    "time_since_prev_prefix" : [],
                    "time_since_start_prefix" : [],
                }
                for attr_id, attribute in enumerate(self.log_info["attributes"]):
                    obs[f"{attribute}_prefix"] = []

                for j in range(batch_size):
                    index = i + j
                    if index >= len(self.X_val):
                        break
                    current_suffix = one_hot_windowed_sequence(
                        self.X_val[index][0], self.window_size, self.log_info["n_activities"]
                    )[np.newaxis, :]
                    obs["suffix"].append(current_suffix)

                    activity_prefix = np.array(self.X_val[index][0])[np.newaxis, :]
                    obs["activity_prefix"].append(activity_prefix)
                    time_since_prev_prefix = np.array(self.X_val[index][-2])[np.newaxis, :]
                    obs["time_since_prev_prefix"].append(time_since_prev_prefix)
                    time_since_start_prefix = np.array(self.X_val[index][-1])[np.newaxis, :]
                    obs["time_since_start_prefix"].append(time_since_start_prefix)
                    for attr_id, attribute in enumerate(self.log_info["attributes"]):
                        obs[f"{attribute}_prefix"].append(np.array(self.X_val[index][attr_id + 1])[np.newaxis, :])


                obs["suffix"] = np.concatenate(obs["suffix"], axis=0)
                obs["activity_prefix"] = np.concatenate(obs["activity_prefix"], axis=0)
                obs["time_since_prev_prefix"] = np.concatenate(obs["time_since_prev_prefix"], axis=0)
                obs["time_since_start_prefix"] = np.concatenate(obs["time_since_start_prefix"], axis=0)
                for attr_id, attribute in enumerate(self.log_info["attributes"]):
                    obs[f"{attribute}_prefix"] = np.concatenate(obs[f"{attribute}_prefix"], axis=0)

                predicted_suffix = None
                action = None

                num_preds = 0

                current_predicted_trace = obs["activity_prefix"]

                while (num_preds < self.log_info["max_trace_length"]):
        
                    action, _states = self.model.predict(obs, deterministic=True)

                    action += 1
                    
                    pred = np.eye(self.log_info["n_activities"])[action]
                    obs["suffix"] = np.concatenate((obs["suffix"][:, 1:, :], pred[:, np.newaxis, :]), axis=1)
                    action = action[:, np.newaxis]
                    if predicted_suffix is None:
                        predicted_suffix = action
                    else:
                        predicted_suffix = np.concatenate((predicted_suffix, action), axis=1)

                    current_predicted_trace = np.concatenate((current_predicted_trace[:, 1:], action), axis=1)
                    num_preds = num_preds + 1


                for j in range(batch_size):
                    index = i + j
                    if index >= len(self.X_val):
                        break

                    suffix_predicted = predicted_suffix[j].reshape(-1).tolist()
                    ground_truth_suffix = list(self.y_val[index])
                    prefix = list(self.X_val[index][0])

                    _, _, similarity = levenshtein_similarity(list(predicted_suffix[j]), list(self.y_val[index]),
                                                              code_end=self.eoc)

                    reward = self.validation_environment._calculate_reward(suffix_predicted, ground_truth_suffix, prefix)

                    try:
                        predicted_len = list(predicted_suffix[j]).index(self.eoc)
                    except ValueError:
                        predicted_len = len(list(predicted_suffix[j]))
                    similarities.append(similarity)
                    rewards.append(reward)
                    predicted_lens.append(predicted_len + 1)

            mean_similarity = np.mean(similarities)
            mean_reward = np.mean(rewards)

            if mean_similarity > self.best_similarity:
                print(f"New best similarity: {mean_similarity}. Improved from {self.best_similarity}. Saving model.")
                self.best_similarity = mean_similarity
                self.no_improve_count = 0
                self.model.save(os.path.join(self.checkpoint_path, "best_model_dl.zip"))
            else:
                self.no_improve_count += 1
                print(f"Similarity {mean_similarity} did not improve from {self.best_similarity}")
                if self.no_improve_count >= self.patience_evals:
                    print("Early stopping: no similarity improvement.")
                    return False

            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save(os.path.join(self.checkpoint_path, "best_model_reward.zip"))

        return True