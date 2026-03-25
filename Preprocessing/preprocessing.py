import pandas as pd
import numpy as np

def sort_log(df,
             categorical_features,
             case_id, 
             timestamp):
    """    
    Transform the format of timestamp and sort the event log by timestamp

    Parameters
    ----------
    df: pandas DataFrame
        Event log
    timestamp: str
        Name of column containing timestamp
                
    Returns
    -------
    sorted_df: pandas DataFrame
        Event log sorted by timestamp

    """
    # convert case id to strings
    df[case_id] = df[case_id].astype('str')
    
    # convert categorical features to strings
    for col in categorical_features:
        df[col] = df[col].astype('str')

    # convert timestamp to Pandas datetime
    df[timestamp] = pd.to_datetime(df[timestamp], format='mixed').dt.tz_convert('UTC')

    # sort the event log by timestamp
    sorted_df = df.sort_values(by=timestamp, kind='mergesort').reset_index(drop=True)

    return sorted_df

def debiasing(orn_df,
            case_id, 
            timestamp, 
            event_name):
    """
    Remove potentially incomplete cases.

    Parameters
    ----------
    df: pandas DataFrame
        Event log
    max_duration: float
        Maximum days a normal case lasts
    case_id: str
        Name of column containing case ID
    timestamp: datetime
        Name of column containing timestamp

    Returns
    -------
    df: pandas Dataframe
        Debiased event log  
    latest_start: datetime
        The last timestamp minus the max_duration, the ending point of test set

    """
    
    df = orn_df.copy()
    df = df.loc[:,[case_id, timestamp, event_name]]
    df_agg = df.groupby(case_id, sort=False).agg(
        tpt=(timestamp, lambda x: (x.max() - x.min()).total_seconds())
        ).reset_index()
    df_agg['tpt'] = df_agg['tpt']/(24 *3600)
    max_duration = df_agg["tpt"].quantile(0.75)
    print(f"Max duration is: {max_duration:.2f} days")
    del (df)
    
    df = orn_df.copy()
    # drop cases starting after the latest_start (potentially incomplete cases)
    # latest_start: the last timestamp minus the max_duration
    latest_start = df[timestamp].max() - pd.Timedelta(max_duration, unit='D')
    print(f"Latest start is: {latest_start}")
    # create array containing case_id of cases starting before latest_start
    case_starts_df = df.groupby(case_id)[timestamp].min().reset_index()
    cases_retained = case_starts_df[case_starts_df[timestamp] <= latest_start][case_id].values
    # filter the dataframe
    df = df[df[case_id].isin(cases_retained)].reset_index(drop=True)

    # ensure event log is sorted by timestamp
    df = df.sort_values(by=timestamp, kind='mergesort').reset_index(drop=True)

    return df

def mapping_case_id(df, 
                    case_id):
    """
    
    Create a dictionary that stores the one-to-one mapping between case_id and 
    index, then transform the case_id into index.  
    
    Parameters
    ----------
    df: pandas DataFrame
        Event log 
    case_id: str
        Name of column containing case ID

    Returns
    -------
    df: pandas DataFrame
        Dataframe with case ID transformed into index.
    case_id_dict: dictionary
        Store the one-to-one mapping between case_id and index
    
    """
    # create the mapping dictionary
    case_id_dict = {}
    n = 1
    for id in df[case_id].unique():
        case_id_dict[id] = n
        n += 1
    
    # map case ID to index
    df[case_id] = df[case_id].map(case_id_dict)

    return df, case_id_dict

def add_soc_eoc(df,
                case_id, 
                timestamp, 
                event_name):
    
    """
    
    Create rows containing SOC (Start of Case) token and EOC (End of Case) token. 
    
    Parameters
    ----------
    df: pandas.DataFrame
        Event log 
    case_id: str
        Name of column containing case ID
    timestamp: str
        Name of column containing timestamp
    event_name: str
        Name of column containing activity label

    Returns
    -------
    df: pandas.DataFrame
        Two rows (SOC, EOC) are added for each case, and one column (event_idx) 
        is added
    
    """
    # sort dataframe by case_id and timestamp
    df = df.sort_values(by=[case_id, timestamp], kind='mergesort').reset_index(drop=True)

    # add SOC and EOC rows
    new_rows = []
    for i in range(len(df)):

        # if this is the first event in a case, append a soc_row before 
        # appending all events of the case
        if i == 0 or df[case_id].iloc[i] != df[case_id].iloc[i - 1]:

            # Create a new row where event name is 'SOC', and values in other 
            # columns are the same as the first event
            soc_row = df.iloc[i].copy() # This returns a series
            soc_row[event_name] = 'SOC'
            
            # Append the 'SOC' row 
            new_rows.append(soc_row)

        # append other rows in this case
        new_rows.append(df.iloc[i])

        # if this is the last event in a case, append a eoc_row after all events of the case
        if i == len(df) - 1 or df[case_id].iloc[i] != df[case_id].iloc[i + 1]:
            # create a new row where event name is 'EOC', and values in other 
            # columns are the same as the last event
            eoc_row = df.iloc[i].copy() # This returns a series
            eoc_row[event_name] = 'EOC'

            # Append the 'EOC' row 
            new_rows.append(eoc_row)

    new_df = pd.DataFrame(new_rows)

    # Create a helper column 'Order' to ensure that SOC is always immediately 
    # above the first event in a case, EOC is always immediately below the last 
    # event in a case
    new_df['Order'] = new_df.apply(lambda row: 
                                   row[case_id] * 10 + (1 if row[event_name] == 'SOC' 
                                                    else (3 if row[event_name] == 'EOC' 
                                                    else 2)), 
                                    axis=1)
    new_df = new_df.sort_values(by=[timestamp, 'Order'], kind='mergesort').reset_index(drop=True)
    
    # Drop the helper column
    new_df = new_df.drop(columns=['Order'])

    # Create event index based on the chronological order 
    new_df['event_idx'] = range(1, len(new_df) + 1)

    return new_df

def create_time_features(df, 
                        case_id, 
                        timestamp,
                        event_idx):
    """
    
    Create two temporal fetures: 
    - trace_ts_pre: time since the previous event in the case
    - trace_ts_statr: time since the start (i.e. the first event) of the case
    
    Parameters
    ----------
    df: pandas DataFrame
        Event log 
    case_id: str
        Name of column containing case ID
    timestamp: datetime
        Name of column containing timestamp
    event_idx: str
        Index of events in event log

    Returns
    -------
    df: pandas DataFrame
        Event log with three temporal features
    
    """

    # calculate time since the previous event in case
    df = df.sort_values(by=[case_id, timestamp], kind='mergesort').reset_index(drop=True)
    df['trace_ts_pre'] = 0.0
    for i in range(1, len(df)): # start from 1 to avoid calculating i-1 when i=0
        if df[case_id].iloc[i] == df[case_id].iloc[i - 1]: # if it is not the first event in each case 
            df.loc[i, 'trace_ts_pre'] = (df[timestamp].iloc[i] - df[timestamp].iloc[i - 1]).total_seconds()

    # a helper column containing the start time of each case
    case_start_times_df = df.groupby(case_id)[timestamp].min().\
        to_frame(name='case_start_time').reset_index(names=case_id) 
    df = pd.merge(df, case_start_times_df, on=case_id, how='inner')

    # calculte time since the start of the case
    df['trace_ts_start'] = (df[timestamp] - df['case_start_time']).dt.total_seconds()
    
    # a helper column containing the end time of each case
    case_end_times_df = df.groupby(case_id)[timestamp].max().\
        to_frame(name='case_end_time').reset_index(names=case_id) 
    df = pd.merge(df, case_end_times_df, on=case_id, how='inner')

    # calculate remaining time
    df['rm'] = (df['case_end_time'] - df[timestamp]).dt.total_seconds()

    # drop the helper column
    df = df.drop(columns=['case_start_time', 'case_end_time'])
    df = df.sort_values(by=event_idx).reset_index(drop=True)

    return df

def train_standardize(df, continuous_features):
    mean_dict = {}
    std_dict = {}

    for col in continuous_features:
        col_mean = df[col].mean()
        col_std = df[col].std()

        mean_dict[col] = col_mean
        std_dict[col] = col_std

        df[col] = (df[col] - col_mean) / col_std

    return df, mean_dict, std_dict

def test_standardize(df, 
                     mean_dict, 
                     std_dict, 
                     continuous_features):
    
    for col in continuous_features:
        df[col] = (df[col] - mean_dict[col]) / std_dict[col]

    return df

def train_log_normalize(df, 
                     continuous_features):
    
    # initialize dictionaries to store min and max for each feature
    max_dict = {}
    min_dict = {}

    # loop through all features
    for col in continuous_features:
        # transform x into ln(1+x), since x could be 0
        df[col] = np.log1p(df[col])

        col_max = df[col].max()
        col_min = df[col].min()
        
        # store the max and min in the dictionaries
        max_dict[col] = col_max
        min_dict[col] = col_min
        
        # log-normalize the column
        df[col] = (df[col] - col_min) / (col_max - col_min)
    
    return df, max_dict, min_dict

def test_log_normalize(df, 
                       max_dict, 
                       min_dict,
                     continuous_features):
    
    # loop through all features
    for col in continuous_features:
        col_max = max_dict[col]
        col_min = min_dict[col]
        
        # log-normalize the column
        df[col] = np.log1p(df[col])
        df[col] = (df[col] - col_min) / (col_max - col_min)
    
    return df

def train_mapping_event_name(df, 
                             event_name):
    """
    
    Mapping activity labels in the training set to index.
    
    Parameters
    ----------
    df: pandas DataFrame
        Training set
    event_name: str
        Name of column containing activity label

    Returns
    -------
    df: pandas DataFrame
        Event log
    event_name_dict: dictionary
        A dictionary where keys are activity labels and values are their indices.
    
    """
    # initialize the dictionary
    event_name_dict = {"SOC":2, 
                       "EOC":3}
    
    # create the mapping dictionary
    n = int(4)
    for name in df[event_name].unique():
        if name not in event_name_dict.keys():
            event_name_dict[name] = n
            n += 1
    
    # map activity label to index
    df[event_name] = df[event_name].map(event_name_dict)

    return df, event_name_dict

def test_mapping_event_name(df, 
                            event_name_dict,
                            event_name):
    """
    
    Mapping activity labels in test set to index.
    
    Parameters
    ----------
    df: pandas DataFrame
        Event log 
    event_name_dict: dictionary
        A dictionary where keys are activity labels and values are their indices.
    event_name: str
        Name of column containing activity label

    Returns
    -------
    df: pandas DataFrame
        Event log
    event_name_dict: dictionary
        The keys are activity labels in training set and the values are the corresponding index.
    
    """
    # initialize the dictionary
    test_event_name_dict = event_name_dict.copy()
    
    # create the mapping dictionary
    for name in df[event_name].unique():
        # activity labels not appearing in training set will be assigned index 1
        if name not in test_event_name_dict.keys():
            test_event_name_dict[name] = int(1)
    
    # map activity label to index
    df[event_name] = df[event_name].map(test_event_name_dict)

    return df, test_event_name_dict

def fill_missing(df, 
                 continuous_features):
    
    flag_feature = []
    
    for col in continuous_features:
        if col not in df.columns:
            print(f" Column '{col}' not found in DataFrame.")
            continue
        
        missing_count = df[col].isna().sum()
        
        if missing_count == 0:
            print(f"No missing values in '{col}'.")
        else:
            print(f" '{col}' has {missing_count} missing values.")
            
            # Create a flag column
            flag_col = f"{col}_fl"
            df[flag_col] = df[col].isna().astype(int)
            
            # Fill missing values
            df[col] = df[col].fillna(0)

            flag_feature.append(flag_col)
            
    return df, flag_feature

def train_map_cat_feature(df,
                    col_name):
    
    if col_name not in df.columns:
        raise ValueError(f"'{col_name}' not found in DataFrame.")
    
    # initialize mapping dict with missing value rule
    mapping_dict = {np.nan: 2, None: 2}
    
    # collect unique non-missing categories
    categories = [cat for cat in df[col_name].unique() if pd.notna(cat)]
    
    n = 3
    for cat in categories:
        mapping_dict[cat] = n
        n += 1
    
    df[col_name] = df[col_name].map(lambda x: 2 if pd.isna(x) else mapping_dict[x])   
    
    return df, mapping_dict

def test_map_cat_feature(df,
                         mapping_dict,
                         col_name):
    
    # initialize the dictionary
    test_mapping_dict = mapping_dict.copy()
    
    # create the mapping dictionary
    for cat in df[col_name].unique():
        if pd.notna(cat) and cat not in test_mapping_dict:
            test_mapping_dict[cat] = int(1)
    
    # map activity label to index
    df[col_name] = df[col_name].map(lambda x: 2 if pd.isna(x) else test_mapping_dict[x]) 

    return df, test_mapping_dict

def map_cat_feature(df,
                    training_df,
                    cat_features):
    
    all_mappings = {}

    for col in cat_features:
        _, mapping_dict = train_map_cat_feature(training_df, col)
        df, test_mapping_dict = test_map_cat_feature(df, mapping_dict, col)
        all_mappings[col] = test_mapping_dict

    return df, all_mappings