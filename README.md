# BRIDGE: A Bayes-Risk-Informed Process Decoder for Suffix Prediction 
This is the official repository for the paper **"BRIDGE: A Bayes-Risk-Informed Process Decoder for Suffix Prediction"**.

The repository includes the following materials:

- **Models**:  Code implementing **BRIDGE** and other baseline decoding strategies, including Argmax decoding, Beam Search, Top-k sampling, and the Daemon Action method, for:
  - Markov models
  - encoder-decoder LSTM
  - encoder-decoder Transformer

- **Reimplementations**:  Code reimplementing suffix prediction models from prior work.

- **Supplementary Materials**:  Supplementary materials, including:
  - an analysis comparing the performance of **BRIDGE** with a re-ranking method that randomly samples 50 suffixes and ranks them by likelihood ([link](<Supplementary Materials/BRIDGE vs Likelihood-Based Re-rank.md>))
  - detailed result tables (including means and standard deviations) comparing **BRIDGE** with four baseline decoding strategies
  - detailed result tables (including means and standard deviations) comparing **BRIDGE** (LSTM-based) with other suffix prediction models

- **Preprocessing**:  Code demonstrating the preprocessing steps.

- **Data**:  Datasets used in the paper.

## BRIDGE Configuration

BRIDGE can be implemented using two methods, depending on the estimator used to compute Bayes risk:

- **`mc_bridge(...)`**: BRIDGE with the Monte Carlo estimator
- **`mb_bridge(...)`**: BRIDGE with the model-based estimator

Choose the method according to the estimator you want to use. Both methods share the following configuration options:

- **Candidate/reference separation (`diff`)**: controls whether the candidate set and reference set are the same.  
  - `False` (default; used in the paper): the candidate set and reference set are identical  
  - `True`: the candidate set and reference set are sampled separately  

- **Set sizes**:
  - `n_sample`: size of the reference set  
  - `n_candidate`: size of the candidate set  
  - If `diff=False`, then `n_sample` must equal `n_candidate`.

- **Sampling method (`sampling`)**: controls how suffixes in candidate set are generated.  
  - `random` (default; used in the paper)  
  - `top-k` (if `top-k` is used, specify the value of `k` using `k=...`.)  
  - `top-p` (if `top-p` is used, specify the value of `p` using `p=...`.)

The model-based method **`mb_bridge(...)`** also includes:

- **Length normalization (`length_norm`)**:
  - `False` (default; used in the paper): disable length normalization  
  - `True`: enable length normalization
  
