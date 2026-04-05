from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from Preprocessing.create_tensor_pipeline import create_tensors
from Models.Markov.Markov_pipeline import fit_evaluate_Markov
from Models.LSTM.LSTM_pipeline import train_evaluate_LSTM
from Models.Transformer.Transformer_pipeline import train_evaluate_Transformer

# Supported options
Dataset = Literal["NASA", "Sepsis", "BPIC12", "BPIC12-W", "BPIC13", "BPIC17", "BPIC19", "BPIC20"]
Model = Literal["Markov", "LSTM", "Transformer"]
DecodingStrategy = Literal["bridge", "argmax", "beam_search", "top_p", "d_action", "prob_rank"]
Estimator = Literal["MC", "model_based"]
BridgeSampling = Literal["random", "top_k", "top_p"]

@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one experiment run."""

    dataset_name: Dataset = "NASA"              # Dataset to use
    model: Model = "Markov"                     # Predictive model
    seed: int = 42                              # Random seed

    decoding: DecodingStrategy = "bridge"       # Decoding strategy

    estimator: Estimator = "MC"                 # Estimator for BRIDGE
    n_candidate: int = 50                       # Size of candidate set
    n_sample: int = 50                          # Size of reference set
    diff: bool = False                          # Whether candidate/reference sets differ
    bridge_sampling: BridgeSampling = "random"  # Sampling strategy to create candidate set
    bridge_sampling_p: float = 0.9              # Used only if bridge_sampling == "top_p"
    bridge_sampling_k: int = 10                 # Used only if bridge_sampling == "top_k"

    beam_width: int = 5                         # Used only if decoding == "beam_search"
    p: float = 0.9                              # Used only if decoding == "top_p"


def run_experiment(config: ExperimentConfig):

    common_kwargs = dict(
        dataset_name=config.dataset_name,
        seed=config.seed,
        decoding=config.decoding,
        estimator=config.estimator,
        n_candidate=config.n_candidate,
        n_sample=config.n_sample,
        diff=config.diff,
        bridge_sampling=config.bridge_sampling,
        bridge_sampling_p=config.bridge_sampling_p,
        bridge_sampling_k=config.bridge_sampling_k,
        beam_width=config.beam_width,
        p=config.p,
    )

    if config.model == "Markov":
        return fit_evaluate_Markov(**common_kwargs)
    elif config.model == "LSTM":
        return train_evaluate_LSTM(**common_kwargs)
    elif config.model == "Transformer":
        return train_evaluate_Transformer(**common_kwargs)

    raise ValueError(f"Unsupported model: {config.model}")

config = ExperimentConfig() # Use all default settings from ExperimentConfig

# Example: override what you want to change
# config = ExperimentConfig(model="LSTM", dataset_name="Sepsis")

create_tensors(config.dataset_name)

test_dl_distance, test_mae_len = run_experiment(config)

print(f"Dataset           : {config.dataset_name}")
print(f"Model             : {config.model}")
print(f"Decoding          : {config.decoding}")
print(f"Test DL distance  : {test_dl_distance:.4f}")
print(f"Test MAE length   : {test_mae_len:.4f}")
