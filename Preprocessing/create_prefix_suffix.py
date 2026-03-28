"""
This module contains functions to generate trace prefix and trace suffix.

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

    """
    Create trace prefix tensor.

    Parameters
    ----------
    df : pandas.DataFrame
        Event log. Use `training_df` for the training set,`val_df` for the 
        validation set, and `df` for the test set.
    trace_prefix_len : int
        The maximum length of the trace prefix.
    case_list : list
        List of cases (training/validation/test cases); only events in these 
        cases are used to generate the trace prefix.
    start_idx : int
        Index of the start of the range (inclusive).
    end_idx : int
        Index of the end of the range (exclusive). 
    trace_col_name : list
        Name(s) of column(s) containing features used in the trace prefix.
    categorical_features : list
        List of column names corresponding to categorical features. These columns
        are converted to integer tensors and padded with 0, while all other
        columns in `trace_col_name` are converted to float tensors and padded
        with -10000.
    case_id : str
        Name of the column containing case IDs.
    event_name : str
        Name of the column containing activity labels.
    event_idx : str
        Name of the column containing event ordering information.
    pad_position : str, {'left', 'right'}
        Indicates whether padding should be applied to the left or right.

    Returns
    -------
    cate_tensors : tensor or empty list
        Prefix tensors constructed from categorical features.
        If more than one categorical feature is included in `trace_col_name`,
        the tensor has shape `(num_obs, trace_prefix_len, num_cate_feature)`.
        If exactly one categorical feature is included, the tensor has shape
        `(num_obs, trace_prefix_len)`. If no categorical features are
        included, this is an empty list.
    nume_tensors : tensor or empty list
        Prefix tensors constructed from numerical features.
        If more than one numerical feature is included in `trace_col_name`,
        the tensor has shape `(num_obs, trace_prefix_len, num_nume_feature)`.
        If exactly one numerical feature is included, the tensor has shape
        `(num_obs, trace_prefix_len)`. If no numerical features are
        included, this is an empty list.
    """

    df = df.sort_values(by=event_idx).reset_index(drop=True)   

    cate_tensors_list = []
    nume_tensors_list = []

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

    spans = [] 
    event_name_vals = df[event_name].to_numpy()                             
    for i in range(start_idx, end_idx):                                  
        c = case_of_row[i]
        if c is None:
            continue
        # skip SOC (event_name == 2) and EOC (event_name == 3)
        if event_name_vals[i] in (2, 3):
            continue
        pos = pos_in_case[i]
        if pos < 0: # case is not in case list
            continue
        start_pos = max(0, pos - trace_prefix_len + 1)         
        end_pos = pos            
        spans.append((c, start_pos, end_pos)) 

    if not spans:                                                              
        raise ValueError("No valid prefix spans were found.")               

    for col in trace_col_name:

        # set padding values for categorical and numerical features
        padding_number = int(0) if col in categorical_features else float(-10000)

        if col in categorical_features:
            out = np.full((len(spans), trace_prefix_len), padding_number, dtype=np.int16)  
        else:
            out = np.full((len(spans), trace_prefix_len), padding_number, dtype=np.float32) 

        col_values = df[col].to_numpy()                                                  
        case_to_colvals = {c: col_values[idxs] for c, idxs in case_to_indices.items()}  

        for r, (c, start_pos, end_pos) in enumerate(spans):                           
            seq = case_to_colvals[c][start_pos:end_pos + 1]
            L = len(seq)
            if L == 0:
                continue
            if L >= trace_prefix_len:  # no padding needed
                seq = seq[-trace_prefix_len:]
                L = trace_prefix_len
                out[r, :L] = seq
            else:
                if pad_position == 'right':
                    out[r, :L] = seq
                elif pad_position == 'left':
                    out[r, -L:] = seq

        trace_prefix_tensor = torch.as_tensor(out) 

        if col in categorical_features:
            trace_prefix_tensor = trace_prefix_tensor.to(torch.int16)  # shape: (num_obs, prefix_len)
            cate_tensors_list.append(trace_prefix_tensor)
            
        else:
            trace_prefix_tensor = trace_prefix_tensor.to(torch.float32) # shape: (num_obs, prefix_len)
            nume_tensors_list.append(trace_prefix_tensor)
    if len(cate_tensors_list) == 0:
        cate_tensors = []
    elif len(cate_tensors_list) == 1:
        cate_tensors = cate_tensors_list[0]   # shape: (num_obs, prefix_len)
    else:
        cate_tensors = torch.stack(cate_tensors_list, dim=-1) # shape: (num_obs, prefix_len, num_cate_features)
    
    if len(nume_tensors_list) == 0:
        nume_tensors = []
    elif len(nume_tensors_list) == 1:
        nume_tensors = nume_tensors_list[0]   # shape: (num_obs, prefix_len)
    else:
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
    """
    Create trace prefix tensor.

    Parameters
    ----------
    df : pandas.DataFrame
        Event log. Use `training_df` for the training set,`val_df` for the 
        validation set, and `df` for the test set.
    trace_suffix_len : int
        The maximum length of the trace suffix.
    case_list : list
        List of cases (training/validation/test cases); only events in these 
        cases are used to generate the trace prefix.
    start_idx : int
        Index of the start of the range (inclusive).
    end_idx : int
        Index of the end of the range (exclusive). 
    trace_col_name : list
        Name(s) of column(s) containing attributes used in the trace suffix.
    categorical_features : list
        List of column names corresponding to categorical features. These columns
        are converted to integer tensors and padded with 0, while all other
        columns in `trace_col_name` are converted to float tensors and padded
        with -10000.
    case_id : str
        Name of the column containing case IDs.
    event_name : str
        Name of the column containing activity labels.
    event_idx : str
        Name of the column containing event ordering information.
    pad_position : str, {'left', 'right'}
        Indicates whether padding should be applied to the left or right.

    Returns
    -------
    suffix_tensors_list : tensor or list of tensors
        Suffix tensor(s). Returns a list of tensors (each of shape
        `(num_obs, trace_suffix_len)`) when more than one suffix is generated,
        or a single tensor of the same shape when only one suffix exists.
    """
    
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

    spans = []    
    event_name_vals = df[event_name].to_numpy()                                    
    for i in range(start_idx, end_idx):                                            
        c = case_of_row[i]
        if c is None:
            continue
        # skip SOC (event_name == 2) and EOC (event_name == 3)
        if event_name_vals[i] in (2, 3):
            continue
        pos = pos_in_case[i]
        if pos < 0:
            continue
        last_pos_in_case = len(case_to_indices[c]) - 1
        start_pos = pos                           
        end_pos = last_pos_in_case                 
        spans.append((c, start_pos, end_pos))                                      

    if not spans:                                                                  
        raise ValueError("No valid suffix spans were found.")                      

    for col in trace_col_name:

        # set padding values for categorical and numerical attributes
        padding_number = int(0) if col in categorical_features else float(-10000)

        if col in categorical_features:
            out = np.full((len(spans), trace_suffix_len), padding_number, dtype=np.int16)    
        else:
            out = np.full((len(spans), trace_suffix_len), padding_number, dtype=np.float32)

        col_values = df[col].to_numpy()                                                      
        case_to_colvals = {c: col_values[idxs] for c, idxs in case_to_indices.items()}      

        for r, (c, start_pos, end_pos) in enumerate(spans):                                  
            seq = case_to_colvals[c][start_pos:end_pos + 1] 
            L = len(seq)
            if L == 0:
                continue
            if L >= trace_suffix_len:  
                seq = seq[:trace_suffix_len]                                                
                L = trace_suffix_len
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

    return suffix_tensors_list[0] if len(suffix_tensors_list) == 1 else suffix_tensors_list
