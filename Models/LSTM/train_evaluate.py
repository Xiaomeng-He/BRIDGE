import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance as norm_dl_distance

def train(model, 
          dataloader,
          optimizer,
          device):
    
    model.train() 

    epoch_loss = 0.0

    for batch in dataloader:

        # load data
        cate_prefix, nume_prefix, dec_input_act, tgt_act = batch
        
        cate_prefix = cate_prefix.long().to(device)
        dec_input_act = dec_input_act.long().to(device)
        tgt_act = tgt_act.long().to(device)
        nume_prefix = nume_prefix.float().to(device)
        
        # set the gradient to zero
        optimizer.zero_grad()

        # run a forward pass and obtain predictions
        act_logits = model(cate_prefix, nume_prefix, dec_input_act) # (batch_size, suffix_len, num_act)

        # ensure predictions are stored in device
        act_logits = act_logits.to(device)
        
        # calculate loss
        loss = loss_function(act_logits, tgt_act)

        # backpropagation
        loss.backward()
        optimizer.step()

        # sum up losses from all batches
        epoch_loss += loss.item()
    
    # compute average losses over the batch
    avg_loss = epoch_loss / len(dataloader)
    
    return avg_loss

def validate(model,
            dataloader,
            device):

    model.eval()

    epoch_loss = 0.0
    epoch_dl_distance, epoch_mea_len = 0.0, 0.0

    with torch.no_grad():

        for batch in dataloader:

            # load data
            cate_prefix, nume_prefix, dec_input_act, tgt_act = batch
            
            cate_prefix = cate_prefix.long().to(device)
            dec_input_act = dec_input_act.long().to(device)
            tgt_act = tgt_act.long().to(device)
            nume_prefix = nume_prefix.float().to(device)

            # run a forward pass and obtain predictions
            act_logits, pred_list, pred_len = model.argmax(cate_prefix, nume_prefix, dec_input_act)

            # ensure predictions are stored in device
            act_logits = act_logits.to(device)

            # calculate loss
            loss = loss_function(act_logits, tgt_act)
            
            # calculate performance metrics
            dl_distance, mae_len = performance_metrics(pred_list, pred_len, tgt_act)
            
            # sum up losses and metrics from all batches
            epoch_loss += loss.item()

            epoch_dl_distance += dl_distance
            epoch_mea_len += mae_len.item()

    # compute average losses and metrics over the batch
    avg_loss = epoch_loss / len(dataloader)

    avg_dl_distance = epoch_dl_distance / len(dataloader)
    avg_mae_len = epoch_mea_len / len(dataloader)
    
    return avg_loss, avg_dl_distance, avg_mae_len

def evaluate(model,
            dataloader,
            device,
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

    model.eval()

    epoch_dl_distance, epoch_mea_len = 0.0, 0.0

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):

        for batch in dataloader:

            # load data
            cate_prefix, nume_prefix, dec_input_act, tgt_act = batch
            
            cate_prefix = cate_prefix.long().to(device)
            dec_input_act = dec_input_act.long().to(device)
            tgt_act = tgt_act.long().to(device)
            nume_prefix = nume_prefix.float().to(device)

            # run a forward pass and obtain predictions
            if decoding == 'bridge':
                pred_list, pred_len = model.bridge(cate_prefix, nume_prefix, dec_input_act, 
                                                n_candidate=n_candidate, 
                                                n_sample=n_sample, 
                                                estimator=estimator, 
                                                diff=diff,
                                                sampling=bridge_sampling,
                                                k=bridge_sampling_k,
                                                p=bridge_sampling_p)
            elif decoding == 'argmax':
                _, pred_list, pred_len = model.argmax(cate_prefix, nume_prefix, dec_input_act)
            
            elif decoding == 'beam_search':
                pred_list, pred_len = model.beam_search(cate_prefix, nume_prefix, dec_input_act,
                                                beam_width=beam_width)
                
            elif decoding == 'top_p':
                pred_list, pred_len = model.top_p_sampler(cate_prefix, nume_prefix, dec_input_act, 
                                                            p=p)

            elif decoding == 'd_action':
                pred_list, pred_len = model.d_action(cate_prefix, nume_prefix, dec_input_act)
                
            elif decoding == 'prob_rank':
                pred_list, pred_len = model.prob_rank(cate_prefix, nume_prefix, dec_input_act,
                                                n_sample=n_sample)
                                                
            else:
                raise ValueError(f"Unknown decoding mode: {decoding}")
            
            # calculate performance metrics
            dl_distance, mae_len = performance_metrics(pred_list, pred_len, tgt_act)
            
            # sum up metrics from all batches
            epoch_dl_distance += dl_distance
            epoch_mea_len += mae_len.item()

    # compute average metrics over the batch
    avg_dl_distance = epoch_dl_distance / len(dataloader)
    avg_mae_len = epoch_mea_len / len(dataloader)
    
    return avg_dl_distance, avg_mae_len

def loss_function(act_logits, 
                  tgt_act):

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    act_logits = act_logits.view(-1, act_logits.size(-1)) # shape: (batch_size * suffix_len, num_act)
    tgt_act = tgt_act.view(-1) # shape: (batch_size * suffix_len,)
    loss = criterion(act_logits, tgt_act)

    return loss

def lens_till_eoc(x: torch.Tensor, eoc_index: int) -> torch.Tensor:
    
        B, T = x.shape
        is_eoc = (x == eoc_index) # (batch_size, suffix_len) bool
        has_eoc = is_eoc.any(dim=1) # (batch_size, ) bool

        first_eoc = is_eoc.int().argmax(dim=1) # (batch_size, ) long

        lengths = torch.where(has_eoc, first_eoc + 1, torch.full_like(first_eoc, T))

        return lengths

def performance_metrics(pred_list,
                        pred_len, 
                        tgt_act,
                        eoc_index = int(3)):
    """
    Compute three performance metrics for suffix prediction:  
    - dl_distance: Normalized Damerau-Levenshtein distance between predicted and 
      ground truth activity label suffixes.
    - mae: Mean Absolute Error between predicted suffix length and ground truth suffix length.

    """
    tgt_act = tgt_act.detach().cpu()
    pred_len = pred_len.detach().cpu()
    batch_size = tgt_act.shape[0]

    total_dl_distance = 0.0
    
    tgt_len  = lens_till_eoc(tgt_act, eoc_index)       # (batch_size,)

    for i in range(batch_size):
        p = pred_list[i]
        t = tgt_act[i, :int(tgt_len[i])].tolist()
        total_dl_distance += norm_dl_distance(p, t)

    # compute average metrics over the batch
    dl_distance = total_dl_distance / batch_size
    mae_len = torch.abs(pred_len - tgt_len).float().mean()
    
    return dl_distance, mae_len

def training_trial(model,
                   lr,
                   num_epochs,
                   train_dataloader,
                   device,
                   val_dataloader,
                   model_state_path,
                   best_val_dl_distance = float("inf")):

    results = []

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer,
                                  mode="min",
                                  factor=0.5,        # multiply LR by 0.5 on plateau
                                  patience=15,        # epochs with no improvement before reducing
                                  threshold=1e-3,    
                                  threshold_mode="abs",
                                  min_lr=1e-5)

    early_stopper = EarlyStopper(patience=20)

    for epoch in tqdm.tqdm(range(num_epochs)):

        train_loss = train(model, 
                           train_dataloader,
                           optimizer,
                           device)
        
        val_loss, val_dl_distance, val_mae_len= validate(model,
                                                   val_dataloader,
                                                   device)
        print(f"\tTrain Loss: {train_loss:7.3f} | Val Loss: {val_loss:7.3f}")
        print(f"\tVal DL Distance: {val_dl_distance:7.3f} | Val MAE length: {val_mae_len:7.3f}")

        if val_dl_distance < (best_val_dl_distance - 0.001):
            best_val_dl_distance = val_dl_distance
            torch.save(model.state_dict(), model_state_path)

        # Store metrics in the results list as a dictionary
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dl_distance': val_dl_distance,
            'val_mae_len': val_mae_len
            })
        
        # learning rate scheduler
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_dl_distance)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"LR reduced: {old_lr:.6g} → {new_lr:.6g} (epoch {epoch+1})")
        
        # Empty redundant cache on GPU
        torch.cuda.empty_cache()

        # early stopping
        if early_stopper.early_stop(val_dl_distance):     
            print(f"Early stopping triggered at epoch {epoch + 1}")        
            break
    
    return results

def init_weights_uni(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param.data, -0.08, 0.08)

class EarlyStopper:
    def __init__(self, patience):
        self.patience = patience 
        self.counter = 0 
        self.min_val_dl_distance = float('inf')

    def early_stop(self, val_dl_distance):

        if val_dl_distance < (self.min_val_dl_distance - 0.001):
            self.min_val_dl_distance = val_dl_distance
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience
