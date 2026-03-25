"""
This module contains functions to generate trace prefix, and trace suffix.

Functions:
    create_trace_prefix
    create_trace_suffix   
"""

import pandas as pd
import numpy as np
import torch

def create_trace_prefix(df, 
                        trace_prefix_len, 
                        case_list,
                        start_idx,
                        end_idx,
                        trace_col_name,
                        categorical_features,
                        case_id,
                        event_name,
                        event_idx,
                        pad_position):

    df = df.sort_values(by=event_idx).reset_index(drop=True)   

    cate_tensors_list = []
    nume_tensors_list = []

    case_to_indices = {}  # where each case’s events are in df                                     
    for c in case_list:                                       
        idxs = np.flatnonzero(df[case_id].to_numpy() == c)
        if idxs.size > 0:
            case_to_indices[c] = idxs

    pos_in_case = -np.ones(len(df), dtype=np.int64)            # position within its case
    case_of_row = np.empty(len(df), dtype=object)              
    case_of_row[:] = None                                      
    for c, idxs in case_to_indices.items():                    
        pos_in_case[idxs] = np.arange(len(idxs))
        case_of_row[idxs] = c

    spans = []  # list of tuples: (case_id, start_pos_in_case, end_pos_in_case) 
    event_name_vals = df[event_name].to_numpy()                             
    for i in range(start_idx, end_idx):                                  
        c = case_of_row[i]
        if c is None:
            continue
        # skip SOC (event_name == 2) and EOC events (event_name == 3)
        if event_name_vals[i] in (2, 3):
            continue
        pos = pos_in_case[i]
        if pos < 0: # case is not in case list
            continue
        # take at most the last 'trace_prefix_len' events up to current position
        start_pos = max(0, pos - trace_prefix_len + 1)         
        end_pos = pos            
        spans.append((c, start_pos, end_pos)) 

    # No usable rows
    if not spans:                                                              
        raise ValueError("No valid prefix spans were found.")               

    for col in trace_col_name:

        # set padding values for categorical and continuous features
        padding_number = int(0) if col in categorical_features else float(-10000)

        # preallocate numpy array filled with padding
        if col in categorical_features:
            out = np.full((len(spans), trace_prefix_len), padding_number, dtype=np.int16)  
        else:
            out = np.full((len(spans), trace_prefix_len), padding_number, dtype=np.float32) 

        # prepare per-case column arrays once, to slice quickly by case positions
        col_values = df[col].to_numpy()                                                  
        case_to_colvals = {c: col_values[idxs] for c, idxs in case_to_indices.items()}  

        # fill each row using precomputed per-case spans
        for r, (c, start_pos, end_pos) in enumerate(spans):                           
            seq = case_to_colvals[c][start_pos:end_pos + 1]
            L = len(seq)
            if L == 0:
                continue
            if L >= trace_prefix_len:  # (no padding needed)
                # take the last trace_prefix_len elements
                seq = seq[-trace_prefix_len:]
                L = trace_prefix_len
                # both left/right fill the full row identically when no padding is needed
                out[r, :L] = seq
            else:
                if pad_position == 'right':
                    out[r, :L] = seq
                elif pad_position == 'left':
                    out[r, -L:] = seq

        # create tensor for each feature column
        trace_prefix_tensor = torch.as_tensor(out)  # from numpy without extra copy

        if col in categorical_features:
            trace_prefix_tensor = trace_prefix_tensor.to(torch.int16)  # shape: (num_obs, prefix_len)
            cate_tensors_list.append(trace_prefix_tensor)
            
        else:
            trace_prefix_tensor = trace_prefix_tensor.to(torch.float32) # shape: (num_obs, prefix_len)
            nume_tensors_list.append(trace_prefix_tensor)

    cate_tensors = torch.stack(cate_tensors_list, dim=-1) # shape: (num_obs, prefix_len, num_cate_features)
    nume_tensors = torch.stack(nume_tensors_list, dim=-1) # shape: (num_obs, prefix_len, num_nume_features)

    return cate_tensors, nume_tensors

def create_trace_suffix(df, 
                        trace_suffix_len,
                        case_list, 
                        start_idx,
                        end_idx,
                        trace_col_name,
                        categorical_features,
                        case_id, 
                        event_name,
                        event_idx,
                        pad_position='right'):
    
    df = df.sort_values(by=event_idx).reset_index(drop=True)

    suffix_tensors_list = []    

    case_to_indices = {}                                       
    for c in case_list:                                        
        idxs = np.flatnonzero(df[case_id].to_numpy() == c)
        if idxs.size > 0:
            case_to_indices[c] = idxs

    pos_in_case = -np.ones(len(df), dtype=np.int64)             
    case_of_row = np.empty(len(df), dtype=object)               
    case_of_row[:] = None                                       
    for c, idxs in case_to_indices.items():                     
        pos_in_case[idxs] = np.arange(len(idxs))
        case_of_row[idxs] = c

    spans = []  # list of tuples: (case_id, start_pos_in_case, end_pos_in_case)   
    event_name_vals = df[event_name].to_numpy()                                    
    for i in range(start_idx, end_idx):                                            
        c = case_of_row[i]
        if c is None:
            continue
        # skip SOC (event_name == 2) and EOC events (event_name == 3)
        if event_name_vals[i] in (2, 3):
            continue
        pos = pos_in_case[i]
        if pos < 0:
            continue
        last_pos_in_case = len(case_to_indices[c]) - 1
        start_pos = pos                           
        end_pos = last_pos_in_case                 
        spans.append((c, start_pos, end_pos))                                      

    # No usable rows
    if not spans:                                                                  
        raise ValueError("No valid suffix spans were found.")                      

    for col in trace_col_name:

        # set padding values for categorical and continuous features
        padding_number = int(0) if col in categorical_features else float(-10000)

        if col in categorical_features:
            out = np.full((len(spans), trace_suffix_len), padding_number, dtype=np.int16)    
        else:
            out = np.full((len(spans), trace_suffix_len), padding_number, dtype=np.float32)

        # prepare per-case column arrays once, to slice quickly by case positions
        col_values = df[col].to_numpy()                                                      
        case_to_colvals = {c: col_values[idxs] for c, idxs in case_to_indices.items()}      

        # fill each row using precomputed per-case spans
        for r, (c, start_pos, end_pos) in enumerate(spans):                                  
            seq = case_to_colvals[c][start_pos:end_pos + 1]  # inclusive of current event  
            L = len(seq)
            if L == 0:
                continue
            if L >= trace_suffix_len:  
                # For suffixes, keep the FIRST 'trace_suffix_len' elements (future direction)
                seq = seq[:trace_suffix_len]                                                
                L = trace_suffix_len
                # both left/right fill the full row identically when no padding is needed
                out[r, :L] = seq
            else:
                if pad_position == 'right':
                    out[r, :L] = seq
                elif pad_position == 'left':
                    out[r, -L:] = seq

        trace_suffix_tensor = torch.as_tensor(out)  
        if col in categorical_features:
            trace_suffix_tensor = trace_suffix_tensor.to(torch.int16) # shape: (num_obs, suffix_len)
        else: 
            trace_suffix_tensor = trace_suffix_tensor.to(torch.float32) # shape: (num_obs, suffix_len)

        suffix_tensors_list.append(trace_suffix_tensor)

    return suffix_tensors_list