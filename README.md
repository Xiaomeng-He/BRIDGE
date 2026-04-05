# BRIDGE: A Bayes-Risk-Informed Process Decoder for Suffix Prediction 
This is the official repository for the paper **"BRIDGE: A Bayes-Risk-Informed Process Decoder for Suffix Prediction"**.

The repository includes the following materials:

- **Models**:  Code implementing **BRIDGE** and other baseline decoding strategies, including Argmax decoding, Beam Search, Top-p sampling, and the Daemon Action method, for:
  - Markov models
  - encoder-decoder LSTM
  - encoder-decoder Transformer

- **Reimplementations**:  Code reimplementing suffix prediction models from prior work.

- **Supplementary Materials**:  Supplementary materials, including:
  - an analysis comparing the performance of **BRIDGE** with a re-ranking method that randomly samples 50 suffixes and ranks them by likelihood ([link](<Supplementary Materials/BRIDGE vs Likelihood-Based Re-rank.md>))
  - detailed result tables (including means and standard deviations) comparing **BRIDGE** with four baseline decoding strategies ([link](<Supplementary Materials/Benchmark Decoding Strategy.md>))
  - detailed result tables (including means and standard deviations) comparing **BRIDGE** (LSTM-based) with other suffix prediction models ([link](<Supplementary Materials/Benchmark Suffix Prediction Models.md>))

- **Preprocessing**:  Code demonstrating the preprocessing steps.

- **Data**:  Datasets used in the paper.

## BRIDGE Configuration

BRIDGE is implemented by **`bridge(...)`**. The method supports the following configuration options:

- **Estimator (`estimator`)**: determines how Bayes risk is estimated.
  - `'MC'`: Monte Carlo estimator
  - `'model_based'`: model-based estimator

- **Candidate/reference separation (`diff`)**: controls whether the candidate set and reference set are the same.  
  - `False` (default; used in the paper): the candidate set and reference set are identical  
  - `True`: the candidate set and reference set are sampled separately  

- **Set sizes**:
  - `n_sample`: size of the reference set  
  - `n_candidate`: size of the candidate set  
  - If `diff=False`, then `n_sample` must equal `n_candidate`.

- **Sampling method (`sampling`)**: controls how suffixes in candidate set are generated.  
  - `random` (default; used in the paper)  
  - `top-k` (if `top-k` is used, specify the value of `k` using `k=...`)  
  - `top-p` (if `top-p` is used, specify the value of `p` using `p=...`)

## Reimplemented Suffix Prediction Models
  
The following table lists the reimplemented suffix prediction models, with the code locations in this repository and links to the original papers and repositories.

| | Folder Name | Paper | Original Repository |
|------:|-------------|------------|-----------|
| 1 | BEST           | [Rauch et al., 2025](https://link.springer.com/chapter/10.1007/978-3-032-02867-9_25) | [Code](https://github.com/lmu-dbs/BEST) |
| 2 | ProcessLSTM    | [Tax et al., 2017](https://link.springer.com/chapter/10.1007/978-3-319-59536-8_30) | [Code](https://github.com/verenich/ProcessSequencePrediction) |
| 3 | AccurateLSTM   | [Camargo et al., 2019](https://link.springer.com/chapter/10.1007/978-3-030-26619-6_19) | [Code](https://github.com/AdaptiveBProcess/GenerativeLSTM) |
| 4 | CRTP           | [Gunnarsson et al., 2023](https://ieeexplore.ieee.org/document/10045798) | [Code](https://github.com/bjornragu/CRTP-LSTM) |
| 5 | SuTraN         | [Wuyts et al., 2024](https://ieeexplore.ieee.org/document/10680671) | [Code](https://github.com/BrechtWts/SuffixTransformerNetwork) |
| 6 | DOGE           | [Rama-Maneiro et al., 2024](https://link.springer.com/chapter/10.1007/978-3-031-61057-8_13) | [Code](https://gitlab.citius.gal/efren.rama/rl-ppm)* |

\* The paper does not provide a link to an official repository. We therefore rely on a repository released by the author and reimplement the model based on that repository and the description in the paper.
