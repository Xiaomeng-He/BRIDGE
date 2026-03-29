import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from create_Markov import MarkovModel
from evaluate import evaluate
from Models.utils import read_config, set_seed

def fit_evaluate_Markov(dataset_name,
                        seed,
                        decoding,
                        estimator='MC',
                        n_candidate=50,
                        n_sample=50,
                        diff=False,
                        bridge_sampling='random',
                        bridge_sampling_p=0.9,
                        bridge_sampling_k=10,
                        beam_width=5,
                        p=0.9):
    parent_dir = Path(__file__).resolve().parent.parent.parent
    dir = parent_dir / f"{dataset_name}_data"
    tensor_dir = dir / "Tensors"

    log_info = read_config(dir / "log_info.yml")
    suffix_len = log_info["max_case_len"] + 1
    num_act = log_info["num_act"]

    # load data
    train_cate_prefix_path = tensor_dir / "train_cate_prefix_l.pt"
    train_act_suffix_path = tensor_dir / "train_act_suffix.pt"

    val_cate_prefix_path = tensor_dir / "val_cate_prefix_l.pt"
    val_act_suffix_path = tensor_dir / "val_cate_prefix_l.pt"

    test_cate_prefix_path = tensor_dir / "test_cate_prefix_l.pt"
    test_act_suffix_path = tensor_dir / "test_cate_prefix_l.pt"

    train_cate_prefix = torch.load(train_cate_prefix_path)
    train_act_suffix = torch.load(train_act_suffix_path)
    x1 = train_cate_prefix[:, -2, 0].clone()
    x2 = train_act_suffix[:, :2].clone()
    train_trigrams = torch.cat([x1.unsqueeze(1), x2], dim=1)

    val_cate_prefix = torch.load(val_cate_prefix_path)
    val_act_suffix = torch.load(val_act_suffix_path)
    x1 = val_cate_prefix[:, -2, 0].clone() # (N,)
    x2 = val_act_suffix[:, :2].clone()
    val_trigrams = torch.cat([x1.unsqueeze(1), x2], dim=1)

    trigrams = torch.cat([train_trigrams, val_trigrams], dim=0)

    test_cate_prefix = torch.load(test_cate_prefix_path)
    test_act_suffix = torch.load(test_act_suffix_path)
    test_prefixes = test_cate_prefix[:, -2:, 0].clone()
    test_tgt_act = test_act_suffix[:, 1:].clone()

    test_dataset = TensorDataset(test_prefixes, test_tgt_act)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    # fit model
    model = MarkovModel(suffix_len=suffix_len, 
                     vocab_size=num_act)
    model.fit(trigrams)

    # decoding
    set_seed(seed)

    test_dl_distance, test_mae_len = evaluate(model,
                                              test_dataloader,
                                              decoding=decoding,
                                              estimator=estimator,
                                              n_candidate=n_candidate,
                                              n_sample=n_sample,
                                              diff=diff,
                                              bridge_sampling=bridge_sampling,
                                              bridge_sampling_p=bridge_sampling_p,
                                              bridge_sampling_k=bridge_sampling_k,
                                              beam_width=beam_width,
                                              p=p)
    
    print('DL distance on test set:', test_dl_distance)
    print('Length MAE on test set:', test_mae_len)

    return test_dl_distance, test_mae_len


