from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import torch
from preprocessing import clean_log, sort_log, debiasing, map_case_id, \
    add_soc_eoc, create_time_features, fill_missing, standardize,\
    map_event_name, map_cat_feature
from train_test_split import get_train_test_split_point, \
    create_table_without_discard_case, get_case_list_by_ratio
from create_prefix_suffix import create_trace_prefix, create_trace_suffix
from utils import read_config, read_data_file

def create_tensors(dataset_name,
                   test_ratio=0.2,
                   val_ratio=0.2):

    BASE_DIR = Path(__file__).resolve().parent # Preprocessing

    output_dir = BASE_DIR.parent / f"{dataset_name}_data"
    tensor_dir = output_dir / "Tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    config = read_config(BASE_DIR / "dataset_configs.yml")
    dataset_cfg = config[dataset_name]

    # define basic variabls
    case_id = dataset_cfg["case_id"]
    event_name = dataset_cfg["event_name"]
    timestamp = dataset_cfg["timestamp"]
    event_idx = 'event_idx'

    cat_event_feature = dataset_cfg["cat_event_feature"]
    cat_case_feature = dataset_cfg["cat_case_feature"]
    num_event_feature = dataset_cfg["num_event_feature"]
    num_case_feature = dataset_cfg["num_case_feature"]
    time_feature = ['trace_ts_pre', 'trace_ts_start']
    categorical_features = [event_name] + cat_event_feature + cat_case_feature # used for prefix creation
    cat_features = cat_event_feature + cat_case_feature # used for categorical feature mapping
    numerical_features = time_feature + num_event_feature + num_case_feature

    case_len_quantile = dataset_cfg["max_case_len_quantile"]
    split_mode = dataset_cfg["split_mode"]

    # read event log
    file_name = dataset_cfg["file_name"]
    orn_df = read_data_file(BASE_DIR.parent / "Data" / file_name)
    df = orn_df.copy

    log_info = {}

    # preprocess event log
    print('Start preprocessing...')
    if event_name == 'combined_activity':
        if 'concept:name' in df.columns and 'lifecycle:transition' in df.columns:
            df['combined_activity'] = df['concept:name'] + '-' + df['lifecycle:transition']
        else:
            raise KeyError("Required columns are missing")
        
    df = clean_log(df, categorical_features, case_id, event_name, timestamp)
    df = sort_log(df, timestamp)
    df, max_case_len = debiasing(df,case_len_quantile, case_id, event_name, timestamp)
    df, case_id_dict = map_case_id(df, case_id)
    df = add_soc_eoc(df, case_id, event_name, timestamp)
    df = create_time_features(df, case_id, timestamp, event_idx)
    df, flag_feature = fill_missing(df, numerical_features)
    prefix_col_name = categorical_features + numerical_features + flag_feature
    num_nume_feature = len(set(numerical_features + flag_feature))

    # dataset split
    if split_mode == 'strict':
        train_test_split_time, train_test_split_idx = get_train_test_split_point(df, 
                                                                         test_ratio,
                                                                         case_id, 
                                                                         timestamp)
        
        df_no_discard = create_table_without_discard_case(df, test_ratio, case_id, timestamp)
        full_training_df = df_no_discard[df_no_discard[timestamp] < train_test_split_time]
        train_case_list, val_case_list = get_case_list_by_ratio(full_training_df,
                                                         val_ratio,
                                                         case_id,
                                                         timestamp)
        test_df = df[df[timestamp] >= train_test_split_time]
        test_case_list = test_df[case_id].unique()
        
    elif split_mode == 'workaround':
        train_test_split_time, train_test_split_idx = get_train_test_split_point(df, 
                                                                         test_ratio,
                                                                         case_id, 
                                                                         timestamp)
        full_train_case_list, test_case_list = get_case_list_by_ratio(df,
                                                         test_ratio,
                                                         case_id,
                                                         timestamp)
        full_training_df = df[df[case_id].isin(full_train_case_list)]
        train_case_list, val_case_list = get_case_list_by_ratio(full_training_df,
                                                         val_ratio,
                                                         case_id,
                                                         timestamp)
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

    training_df = df[df[case_id].isin(train_case_list)]

    # transform categorical features to intergers
    df, mapping_dicts = map_cat_feature(df, training_df, cat_features)
    cat_dims = [max(mapping.values()) + 1 for mapping in mapping_dicts.values()]

    df, event_name_dict = map_event_name(df, training_df, event_name)
    num_act = max(event_name_dict.values()) + 1

    # standardize numerical features
    df, mean_dict, std_dict = standardize(df, training_df, numerical_features)

    log_info = {
    "max_case_len": max_case_len,
    "num_act": num_act,
    "cat_dims": cat_dims,
    "num_nume_feature": num_nume_feature,
    "prefix_col_name": prefix_col_name}

    with open(output_dir / "log_info.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(log_info, f)

    torch.save(mean_dict, output_dir / "mean_dict.pt")
    torch.save(std_dict, output_dir / "std_dict.pt")
    
    # create tensors
    print('Start creating tensors...')
    trace_prefix_len = max_case_len + 1
    trace_suffix_len = max_case_len + 2
    suffix_col_name = [event_name]

    training_df = df[df[case_id].isin(train_case_list)].reset_index(drop=True)
    train_cate_prefix_r, train_nume_prefix_r  = create_trace_prefix(df=training_df, 
                        trace_prefix_len=trace_prefix_len, 
                        case_list=train_case_list,
                        start_idx=0,
                        end_idx=len(training_df),
                        prefix_col_name=prefix_col_name,
                        categorical_features=categorical_features,
                        case_id=case_id,
                        event_name=event_name,
                        event_idx=event_idx,
                        pad_position='right')
    train_cate_prefix_l, train_nume_prefix_l  = create_trace_prefix(df=training_df, 
                        trace_prefix_len=trace_prefix_len, 
                        case_list=train_case_list,
                        start_idx=0,
                        end_idx=len(training_df),
                        prefix_col_name=prefix_col_name,
                        categorical_features=categorical_features,
                        case_id=case_id,
                        event_name=event_name,
                        event_idx=event_idx,
                        pad_position='left')
    train_act_suffix = create_trace_suffix(df=training_df, 
                        trace_suffix_len=trace_suffix_len, 
                        case_list=train_case_list,
                        start_idx=0,
                        end_idx=len(training_df),
                        suffix_col_name=suffix_col_name,
                        categorical_features=categorical_features,
                        case_id=case_id,
                        event_name=event_name,
                        event_idx=event_idx,
                        pad_position='right')
    
    torch.save(train_cate_prefix_r, tensor_dir / "train_cate_prefix_r.pt")
    torch.save(train_nume_prefix_r, tensor_dir / "train_nume_prefix_r.pt")
    torch.save(train_cate_prefix_l, tensor_dir / "train_cate_prefix_l.pt")
    torch.save(train_nume_prefix_l, tensor_dir / "train_nume_prefix_l.pt")
    torch.save(train_act_suffix, tensor_dir / "train_act_suffix.pt")

    val_df = df[df[case_id].isin(val_case_list)].reset_index(drop=True)
    val_cate_prefix_r, val_nume_prefix_r = create_trace_prefix(df=val_df, 
                        trace_prefix_len=trace_prefix_len, 
                        case_list=val_case_list,
                        start_idx=0,
                        end_idx=len(val_df),
                        prefix_col_name=prefix_col_name,
                        categorical_features=categorical_features,
                        case_id=case_id,
                        event_name=event_name,
                        event_idx=event_idx,
                        pad_position='right')
    val_cate_prefix_l, val_nume_prefix_l = create_trace_prefix(df=val_df, 
                        trace_prefix_len=trace_prefix_len, 
                        case_list=val_case_list,
                        start_idx=0,
                        end_idx=len(val_df),
                        prefix_col_name=prefix_col_name,
                        categorical_features=categorical_features,
                        case_id=case_id,
                        event_name=event_name,
                        event_idx=event_idx,
                        pad_position='left')
    val_act_suffix = create_trace_suffix(df=val_df, 
                        trace_suffix_len=trace_suffix_len, 
                        case_list=val_case_list,
                        start_idx=0,
                        end_idx=len(val_df),
                        suffix_col_name=suffix_col_name,
                        categorical_features=categorical_features,
                        case_id=case_id,
                        event_name=event_name,
                        event_idx=event_idx,
                        pad_position='right')

    torch.save(val_cate_prefix_r, tensor_dir / "val_cate_prefix_r.pt")
    torch.save(val_nume_prefix_r, tensor_dir / "val_nume_prefix_r.pt")
    torch.save(val_cate_prefix_l, tensor_dir / "val_cate_prefix_l.pt")
    torch.save(val_nume_prefix_l, tensor_dir /"val_nume_prefix_l.pt")
    torch.save(val_act_suffix, tensor_dir / "val_act_suffix.pt")

    mask = df[timestamp] >= train_test_split_time
    train_test_split_idx = mask.idxmax()

    test_cate_prefix_r, test_nume_prefix_r = create_trace_prefix(df=df, 
                        trace_prefix_len=trace_prefix_len, 
                        case_list=test_case_list,
                        start_idx=train_test_split_idx,
                        end_idx=len(df),
                        prefix_col_name=prefix_col_name,
                        categorical_features=categorical_features,
                        case_id=case_id,
                        event_name=event_name,
                        event_idx=event_idx,
                        pad_position='right')
    test_cate_prefix_l, test_nume_prefix_l = create_trace_prefix(df=df, 
                        trace_prefix_len=trace_prefix_len, 
                        case_list=test_case_list,
                        start_idx=train_test_split_idx,
                        end_idx=len(df),
                        prefix_col_name=prefix_col_name,
                        categorical_features=categorical_features,
                        case_id=case_id,
                        event_name=event_name,
                        event_idx=event_idx,
                        pad_position='left')
    test_act_suffix = create_trace_suffix(df=df, 
                        trace_suffix_len=trace_suffix_len, 
                        case_list=test_case_list,
                        start_idx=train_test_split_idx,
                        end_idx=len(df),
                        suffix_col_name=suffix_col_name,
                        categorical_features=categorical_features,
                        case_id=case_id,
                        event_name=event_name,
                        event_idx=event_idx,
                        pad_position='right')

    torch.save(test_cate_prefix_r, tensor_dir / "test_cate_prefix_r.pt")
    torch.save(test_nume_prefix_r, tensor_dir / "test_nume_prefix_r.pt")
    torch.save(test_cate_prefix_l, tensor_dir / "test_cate_prefix_l.pt")
    torch.save(test_nume_prefix_l, tensor_dir / "test_nume_prefix_l.pt")
    torch.save(test_act_suffix, tensor_dir / "test_act_suffix.pt")
