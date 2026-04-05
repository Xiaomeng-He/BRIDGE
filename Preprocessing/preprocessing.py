import pandas as pd
import numpy as np

def clean_log(df,
              categorical_features,
              case_id,
              event_name,
              timestamp):
    """
    Remove missing rows, drop duplicates, and cast case ID and categorical 
    columns to string.
    """

    df = df.dropna(subset=[case_id, event_name, timestamp])
    df = df.drop_duplicates(keep="first")
    df[case_id] = df[case_id].astype('str')
    for col in categorical_features:
        df[col] = df[col].astype('str')

    return df

def sort_log(df,
             timestamp):
    """
    Transform the format of timestamp and sort the event log by timestamp.
    """  

    df[timestamp] = pd.to_datetime(df[timestamp], format='mixed').dt.tz_convert('UTC')
    sorted_df = df.sort_values(by=timestamp, kind='mergesort').reset_index(drop=True)

    return sorted_df

def debiasing(orn_df,
            case_len_quantile,
            case_id, 
            event_name,
            timestamp):
    """
    Remove potentially incomplete cases and cases whose length exceeds a 
    threshold.
    """
    
    df = orn_df.copy()
    df = df.loc[:,[case_id, timestamp, event_name]]
    df_agg = df.groupby(case_id).agg(
        tpt=(timestamp, lambda x: (x.max() - x.min()).total_seconds()),
        case_len=(event_name, "count")
    ).reset_index()
    df_agg['tpt'] = df_agg['tpt']/(24 *3600)
    max_duration = df_agg["tpt"].quantile(0.75)
    max_len = int(df_agg["case_len"].quantile(case_len_quantile))
    del (df)
    
    df = orn_df.copy()

    # drop cases starting after the latest_start (potentially incomplete cases)
    # latest_start: the last timestamp minus the max_duration
    latest_start = df[timestamp].max() - pd.Timedelta(max_duration, unit='D')
    case_starts_df = df.groupby(case_id)[timestamp].min().reset_index()
    cases_by_start = set(case_starts_df[case_starts_df[timestamp] <= latest_start][case_id])

    # drop cases longer than max_len
    cases_by_len = set(df_agg[df_agg["case_len"] <= max_len][case_id])

    cases_retained = cases_by_start & cases_by_len
    df = df[df[case_id].isin(cases_retained)].reset_index(drop=True)

    # ensure event log is sorted by timestamp
    df = df.sort_values(by=timestamp, kind='mergesort').reset_index(drop=True)

    return df, max_len

def map_case_id(df, 
                case_id):
    """
    Create a dictionary that stores the one-to-one mapping between case_id and 
    index, then transform the case_id into index.  
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
                event_name,
                timestamp):
    
    """ 
    Create rows containing SOC (Start of Case) and EOC (End of Case). 
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

    # create a helper column 'Order' to ensure that SOC is always immediately 
    # above the first event in a case, EOC is always immediately below the last 
    # event in a case
    new_df['Order'] = new_df.apply(lambda row: 
                                   row[case_id] * 10 + (1 if row[event_name] == 'SOC' 
                                                    else (3 if row[event_name] == 'EOC' 
                                                    else 2)), 
                                    axis=1)
    new_df = new_df.sort_values(by=[timestamp, 'Order'], kind='mergesort').reset_index(drop=True)
    
    # drop the helper column
    new_df = new_df.drop(columns=['Order'])

    # create event index based on the chronological order 
    new_df['event_idx'] = range(1, len(new_df) + 1)

    return new_df

def create_time_features(df, 
                        case_id, 
                        timestamp,
                        event_idx):
    """
    Create two temporal fetures: 
    - trace_ts_pre: time since the previous event in the case
    - trace_ts_start: time since the start (i.e. the first event) of the case
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

    # drop the helper column
    df = df.drop(columns=['case_start_time'])
    df = df.sort_values(by=event_idx).reset_index(drop=True)

    return df

def fill_missing(df, 
                 numerical_features):
    """
    Fill missing values in numerical features with zero and create indicator 
    columns for features with missing values.
    """

    flag_feature = []
    
    for col in numerical_features:
        if col not in df.columns:
            print(f" Column '{col}' not found in DataFrame.")
            continue
        
        missing_count = df[col].isna().sum()
        
        if missing_count != 0:
            
            # Create a flag column
            flag_col = f"{col}_fl"
            df[flag_col] = df[col].isna().astype(int)
            
            # Fill missing values
            df[col] = df[col].fillna(0)

            flag_feature.append(flag_col)
            
    return df, flag_feature

def train_standardize(df, 
                      numerical_features):
    """
    Compute mean and standard deviation of numerical features using training set.
    """

    mean_dict = {}
    std_dict = {}

    for col in numerical_features:
        col_mean = df[col].mean()
        col_std = df[col].std()

        mean_dict[col] = col_mean
        std_dict[col] = col_std

        df[col] = (df[col] - col_mean) / col_std

    return mean_dict, std_dict

def test_standardize(df, 
                     mean_dict, 
                     std_dict, 
                     numerical_features):
    """
    Standardize numerical features using precomputed mean and standard
    deviation values.
    """

    for col in numerical_features:
        df[col] = (df[col] - mean_dict[col]) / std_dict[col]

    return df

def standardize(df,
                training_df, 
                numerical_features):
    """
    Standardize numerical features using statistics derived from the
    training set.
    """

    mean_dict, std_dict = train_standardize(training_df, numerical_features)
    df = test_standardize(df, 
                     mean_dict, 
                     std_dict, 
                     numerical_features)
    
    return df, mean_dict, std_dict
    
def train_map_event_name(df, 
                         event_name):
    """
    Map event names in the training set to integer indices.
    """
    event_name_dict = {"SOC":2, 
                       "EOC":3}
    
    n = int(4)
    for name in df[event_name].unique():
        if name not in event_name_dict.keys():
            event_name_dict[name] = n
            n += 1
    
    df[event_name] = df[event_name].map(event_name_dict)

    return df, event_name_dict

def test_map_event_name(df, 
                        event_name_dict,
                        event_name):
    """
    Map event names in the test set to integer indices.
    """
    test_event_name_dict = event_name_dict.copy()
    
    for name in df[event_name].unique():
        # activity labels not appearing in training set will be assigned index 1
        if name not in test_event_name_dict.keys():
            test_event_name_dict[name] = int(1)
    
    df[event_name] = df[event_name].map(test_event_name_dict)

    return df, test_event_name_dict

def map_event_name(df,
                   training_df,
                   event_name):
    """
    Map event names using indices derived from the training set.
    """
    _, event_name_dict = train_map_event_name(training_df, event_name)
    df, test_event_name_dict = test_map_event_name(df, 
                             event_name_dict,
                             event_name)
    return df, test_event_name_dict

def train_map_cat_feature(df,
                    col_name):
    """
    Map categorical feature values in the training set to integer indices.
    """
    
    if col_name not in df.columns:
        raise ValueError(f"'{col_name}' not found in DataFrame.")
    
    mapping_dict = {np.nan: 2, None: 2}
    
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
    """
    Map categorical feature values in the test set to integer indices.
    """
    
    test_mapping_dict = mapping_dict.copy()
    
    for cat in df[col_name].unique():
        if pd.notna(cat) and cat not in test_mapping_dict:
            test_mapping_dict[cat] = int(1)
    
    df[col_name] = df[col_name].map(lambda x: 2 if pd.isna(x) else test_mapping_dict[x]) 

    return df, test_mapping_dict

def map_cat_feature(df,
                    training_df,
                    cat_features):
    """
    Map categorical features using indices derived from the training set.
    """
    
    all_mappings = {}

    for col in cat_features:
        _, mapping_dict = train_map_cat_feature(training_df, col)
        df, test_mapping_dict = test_map_cat_feature(df, mapping_dict, col)
        all_mappings[col] = test_mapping_dict

    return df, all_mappings
