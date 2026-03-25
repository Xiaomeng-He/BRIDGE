"""
This module contains functions to split dataset into training set, validation
set and test set

Functions:

    get_train_test_split_point
    get_discard_case_list
    create_table_without_discard_case
    get_train_val_case_list
    
"""

import pandas as pd
import numpy as np

def get_train_test_split_point(orn_df, 
                               test_ratio, 
                               case_id,
                               timestamp):
    """
    Get the splitting point to separate development set (containing training set 
    and validation set) and test set

    Parameters
    ----------
    df: pandas DataFrame
        Event log
    test_ratio: float
        The percentage of the test set
    case_id: str
        Name of column containing case ID
    timestamp: datetime
        Name of column containing timestamp

    Returns
    -------
    train_test_split_time: datetime
        The timestamp of the first event in test set
    train_test_split_idx: int
        The index of the first event in test set
    """

    df = orn_df.copy()

    # get the start time of each case and sort the resulting table by start time
    case_start_times = df.groupby(case_id)[timestamp].min().sort_values() 
   
    # get the index of the first case in test set
    first_test_case_idx = int(len(case_start_times) * (1 - test_ratio)) 
    
    # get the case_id of the first case in test set
    first_test_case_id = case_start_times.index[first_test_case_idx]
    
    # get the index of the first event in the first case of the test set
    train_test_split_idx = df[df[case_id] == first_test_case_id].index[0]
    
    # get the timestamp of the first event in test set, which is the splitting 
    # point to separate development set and test set
    train_test_split_time = df.loc[train_test_split_idx, timestamp]

    del (df)
    
    return train_test_split_time, train_test_split_idx

def get_discard_case_list(orn_df, 
                          test_ratio, 
                          case_id, 
                          timestamp):
    """
    Get the list of cases that should be removed from development set (for 
    BPIC2012, BPIC2017 and BAC) or test set (for BPIC2019) when generating 
    prefixes and suffixes. These cases start before and end after the splitting 
    point.

    Parameters
    ----------
    df: pandas DataFrame
        Event log
    test_ratio: float
        The percentage of the test set
    case_id: str
        Name of column containing case ID
    timestamp: datetime
        Name of column containing timestamp

    Returns
    -------
    discard_case_list: list
        List of case ID of cases to diacard
    """
    df = orn_df.copy()

    # get the start time and end time of each case
    case_start_times = df.groupby(case_id)[timestamp].min().rename('Start') 
    case_stop_times = df.groupby(case_id)[timestamp].max().rename('End')
    case_start_stop_times = pd.concat([case_start_times, case_stop_times], axis=1).reset_index()
    train_test_split_time, _ = get_train_test_split_point(df, test_ratio, case_id, timestamp)

    # get the list of cases that start before and end after the splitting point
    discard_cases = case_start_stop_times[
        (case_start_stop_times['Start'] < train_test_split_time) & 
        (case_start_stop_times['End'] >= train_test_split_time)]
    discard_case_list = discard_cases[case_id].tolist()

    del (df)
    
    return discard_case_list

def create_table_without_discard_case(orn_df, 
                                      test_ratio,
                                      case_id, 
                                      timestamp):
    """
    Create event log with discarded case removed

    Parameters
    ----------
    df: pandas DataFrame
        Event log
    test_ratio: float
        The percentage of the test set
    case_id: str
        Name of column containing case ID
    timestamp: datetime
        Name of column containing timestamp

    Returns
    -------
    df_without_discard_case: pandas DataFrame
        Event log

    """

    df = orn_df.copy()
    
    discard_case_list = get_discard_case_list(df, test_ratio, case_id, timestamp)
    df_without_discard_case = df[~df[case_id].isin(discard_case_list)]
    
    return df_without_discard_case

def get_train_val_case_list(orn_df, 
                          val_ratio, 
                          case_id, 
                          timestamp):
    """
    Get the list of training cases and validation cases

    Parameters
    ----------
    df: pandas DataFrame
        Event log pertaining to development set
    test_ratio: float
        The percentage of the test set
    case_id: str
        Name of column containing case ID
    timestamp: datetime
        Name of column containing timestamp

    Returns
    -------
    train_case_list: list
        The list of training cases
    val_case_list: list
        The list of validation cases
    """    
    df = orn_df.copy()

    # get the start time of each case and sort by start time
    case_start_times = df.groupby(case_id)[timestamp].min().sort_values() # The results is a series

    # get the number of training cases
    num_train_cases = int(len(case_start_times) * (1 - val_ratio))

    train_case_list = case_start_times[:num_train_cases+1].index.tolist()
    val_case_list = case_start_times[num_train_cases+1:].index.tolist()

    del (df)

    return train_case_list, val_case_list