# BRIDGE: A Bayes-Risk-Informed Process Decoder for Suffix Prediction 
This is the official repository for the paper **"BRIDGE: A Bayes-Risk-Informed Process Decoder for Suffix Prediction"**.

The repository includes the following materials:

- **`Models`**:  Code implementing **BRIDGE** and other baseline decoding strategies, including Argmax decoding, Beam Search, Top-k sampling, and the Daemon Action method, for:
  - Markov models
  - encoder-decoder LSTM
  - encoder-decoder Transformer

- **`Reimplementations`**:  Code reimplementing suffix prediction models from prior work.

- **`Supplementary Materials`**:  Supplementary materials, including:
  - an analysis comparing the performance of **BRIDGE** with a re-ranking method that samples 50 suffixes and ranks them by likelihood
  - detailed result tables (including means and standard deviations) comparing **BRIDGE** with four baseline decoding strategies
  - detailed result tables (including means and standard deviations) comparing **BRIDGE** (LSTM-based) with other suffix prediction models

- **`Preprocessing`**:  Code demonstrating the preprocessing steps.

- **`Data`**:  Datasets used in the paper.
  
