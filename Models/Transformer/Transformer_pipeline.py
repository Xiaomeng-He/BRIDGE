import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import csv

from create_Transformer import ED_Transformer
from train_evaluate import training_trial, evaluate
from Models.utils import read_config, set_seed

def train_evaluate_Transformer(dataset_name,
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

    # define hyperparameters
    log_info = read_config(dir / "log_info.yml")
    prefix_len = log_info["max_case_len"] + 1
    suffix_len = log_info["max_case_len"] + 1
    num_act = log_info["num_act"]
    act_embed = round(1.6 * (num_act ** 0.56))
    cat_dims = log_info["cat_dims"]
    emb_dims = [min(600, round(1.6 * (cat_dim ** 0.56))) for cat_dim in cat_dims]
    num_nume_feature = log_info["num_nume_feature"]

    enc_input_size = act_embed + sum(emb_dims) + num_nume_feature
    dec_input_size = act_embed

    BASE_DIR = Path(__file__).resolve().parent # LSTM
    hp = read_config(BASE_DIR / "Transformer_hparams.yml")
    dataset_hparams = hp[dataset_name]

    d_model = 32
    dropout = 0.2
    num_heads = dataset_hparams["num_heads"]
    d_ff = dataset_hparams["d_ff"]
    num_layers = dataset_hparams["num_layers"]

    batch_size = 64
    lr = 0.0002
    num_epochs = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_cate_prefix_path = tensor_dir / "train_cate_prefix_r.pt"
    train_nume_prefix_path = tensor_dir / "train_nume_prefix_r.pt"
    train_act_suffix_path = tensor_dir / "train_act_suffix.pt"

    val_cate_prefix_path = tensor_dir / "val_cate_prefix_r.pt"
    val_nume_prefix_path = tensor_dir / "val_nume_prefix_r.pt"
    val_act_suffix_path = tensor_dir / "val_act_suffix.pt"

    test_cate_prefix_path = tensor_dir / "test_cate_prefix_r.pt"
    test_nume_prefix_path = tensor_dir / "test_nume_prefix_r.pt"
    test_act_suffix_path = tensor_dir / "test_act_suffix.pt"

    train_cate_prefix = torch.load(train_cate_prefix_path)
    train_nume_prefix = torch.load(train_nume_prefix_path)
    train_act_suffix = torch.load(train_act_suffix_path)
    train_dec_input_act = train_act_suffix[:, :-1].clone()
    train_dec_input_act[train_dec_input_act == 3] = 0
    train_tgt_act = train_act_suffix[:, 1:].clone()

    val_cate_prefix = torch.load(val_cate_prefix_path)
    val_nume_prefix = torch.load(val_nume_prefix_path)
    val_act_suffix = torch.load(val_act_suffix_path)
    val_dec_input_act = val_act_suffix[:, :-1].clone()
    val_dec_input_act[val_dec_input_act == 3] = 0
    val_tgt_act = val_act_suffix[:, 1:].clone()

    test_cate_prefix = torch.load(test_cate_prefix_path)
    test_nume_prefix = torch.load(test_nume_prefix_path)
    test_act_suffix = torch.load(test_act_suffix_path)
    test_dec_input_act = test_act_suffix[:, :-1].clone()
    test_dec_input_act[test_dec_input_act == 3] = 0
    test_tgt_act = test_act_suffix[:, 1:].clone()

    train_dataset = TensorDataset(train_cate_prefix, train_nume_prefix,
                                train_dec_input_act, train_tgt_act)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(val_cate_prefix, val_nume_prefix,
                                val_dec_input_act, val_tgt_act)

    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    test_dataset = TensorDataset(test_cate_prefix, test_nume_prefix,
                                test_dec_input_act, test_tgt_act)

    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # training
    set_seed(seed)
    parameter_path = dir / 'Transformer_parameters.pt'
    model = ED_Transformer(prefix_len, suffix_len,
                        act_embed, cat_dims, emb_dims,
                        enc_input_size, dec_input_size,
                        num_act, 
                        d_model, num_heads, d_ff, dropout,
                        num_layers).to(device)

    results = training_trial(model,
                        lr,
                        num_epochs,
                         train_dataloader,
                         device,
                         val_dataloader,
                         parameter_path)
    
    csv_file = dir / 'Transformer_loss.csv'
    csv_columns = ['epoch', 'train_loss', 'val_loss', 'val_dl_distance', 'val_mae_len']
    try:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(results)
        print(f"Metrics saved to {csv_file}")
    except IOError as e:
        print("I/O error", e)
    
    # evaluation
    model = ED_Transformer(prefix_len, suffix_len,
                        act_embed, cat_dims, emb_dims,
                        enc_input_size, dec_input_size,
                        num_act, 
                        d_model, num_heads, d_ff, dropout,
                        num_layers).to(device)
    state_dict = torch.load(parameter_path)
    model.load_state_dict(state_dict)

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