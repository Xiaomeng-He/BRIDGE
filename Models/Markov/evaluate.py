import numpy as np
import pandas as pd
import torch
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance as norm_dl_distance

def evaluate(model,
            dataloader,
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

    epoch_dl_distance, epoch_mea_len = 0.0, 0.0

    for batch in dataloader:

        prefixes, tgt_act = batch

        if decoding == 'bridge':
            pred_list, pred_len = model.bridge(prefixes=prefixes, 
                                               n_candidate=n_candidate, 
                                               n_sample=n_sample, 
                                               estimator=estimator, 
                                               diff=diff,
                                               sampling=bridge_sampling,
                                               k=bridge_sampling_k,
                                               p=bridge_sampling_p)
        elif decoding == 'argmax':
            pred_list, pred_len = model.argmax(prefixes)
        elif decoding == 'beam_search':
            pred_list, pred_len = model.beam_search(prefixes=prefixes, 
                                                    beam_width=beam_width)
        elif decoding == 'top_p':
            pred_list, pred_len = model.top_p_sampler(ctx=prefixes, 
                                                      p=p)
        elif decoding == 'd_action':
            pred_list, pred_len = model.d_action(prefixes)
        elif decoding == 'prob_rank':
            pred_list, pred_len = model.prob_rank(prefixes=prefixes, 
                                                  n_sample=n_sample)
        else:
            raise ValueError(f"Unknown decoding mode: {decoding}")
        
        # calculate performance metrics
        dl_distance, mae_len = performance_metrics(pred_list, pred_len, tgt_act)
        
        # sum up losses and metrics from all batches
        epoch_dl_distance += dl_distance
        epoch_mea_len += mae_len.item()

    # compute average losses and metrics over the batch
    avg_dl_distance = epoch_dl_distance / len(dataloader)
    avg_mae_len = epoch_mea_len / len(dataloader)
    
    return avg_dl_distance, avg_mae_len

def lens_till_eoc(x, eoc_index):
    
        B, T = x.shape
        is_eoc = (x == eoc_index) # (batch_size, suffix_len) bool
        has_eoc = is_eoc.any(dim=1) # (batch_size, ) bool

        # argmax gives first True only if there is any True; otherwise 0.
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
    
    tgt_len  = lens_till_eoc(tgt_act, eoc_index) 

    for i in range(batch_size):
        p = pred_list[i]
        t = tgt_act[i, :int(tgt_len[i])].tolist()
        total_dl_distance += norm_dl_distance(p, t)

    dl_distance = total_dl_distance / batch_size
    mae_len = torch.abs(pred_len - tgt_len).float().mean()
    
    return dl_distance, mae_len