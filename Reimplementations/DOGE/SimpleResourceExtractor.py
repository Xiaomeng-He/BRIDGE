from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from torch import nn
import torch


def tensor_one_hot_windowed_sequence(current_sequence, n_activities, window_size):
    device = current_sequence.device

    trace_length = current_sequence.size(1)
    batch_size = current_sequence.size(0)
    seq_length = min(trace_length, window_size)
    current_sequence = current_sequence[:, -seq_length:].long()

    # Create a one hot tensor
    one_hot_sequence = torch.zeros((batch_size, seq_length, n_activities), device=device)

    mask = current_sequence != 0

    idx = torch.nonzero(mask)

    # Populate the one hot tensor using the current sequence
    one_hot_sequence[idx[:, 0], idx[:, 1], current_sequence[idx[:, 0], idx[:, 1]]] = 1

    return one_hot_sequence

class SimpleResourceExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: spaces.Dict, 
                 features_dim : int, 
                 log_info : dict, 
                 window_size : int):
        self.window_size = window_size
        self.log_info = log_info
        self.seq_length = min(window_size, log_info["max_trace_length"])
        features_dim = self.seq_length * self.log_info["n_activities"] + \
            window_size * self.log_info["n_activities"] + \
                2 * self.seq_length
        # activity prefix
        # activity suffix
        # time feature prefix
        for attribute in log_info["attributes"]:
            features_dim += self.seq_length * log_info[f"n_{attribute}"]

        super().__init__(observation_space, features_dim=features_dim)

        extractors = {}

        extractors["suffix"] = nn.Flatten()
        extractors["activity_prefix"] = nn.Identity()
        for attribute in log_info["attributes"]:
            extractors[attribute + "_prefix"] = nn.Identity()

        extractors["time_since_prev_prefix"] = nn.Identity()
        extractors["time_since_start_prefix"] = nn.Identity()

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations):
        output = []

        for key, extractor in self.extractors.items():
            if key != "suffix" and key != "time_since_prev_prefix" and key != "time_since_start_prefix":
                initial_tensor = observations[key]
                if key == "activity_prefix":
                    log_info_key = "n_activities"
                else:
                    log_info_key = "n_" + key.replace("_prefix", "")

                one_hot_tensor = tensor_one_hot_windowed_sequence(initial_tensor, 
                                                                  self.log_info[log_info_key], 
                                                                  self.window_size)
                extracted_output = extractor(one_hot_tensor)
                extracted_output = extracted_output.view(one_hot_tensor.shape[0], -1)
                output.append(extracted_output)
            elif key == "time_since_prev_prefix" or key == "time_since_start_prefix":
                extracted_output = observations[key][:, -self.window_size:]
                output.append(extracted_output)
            else:
                extracted_output = extractor(observations[key])
                output.append(extracted_output)

        concated_output = torch.cat(output, dim=-1)
        return concated_output
