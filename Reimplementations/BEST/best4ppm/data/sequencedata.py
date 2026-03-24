from __future__ import annotations

import numpy as np
import pandas as pd
import copy
import os
import math
import itertools
from enum import Enum

from ..util.sequence_utils import _filter_start_end

from sklearn.preprocessing import LabelEncoder

import logging
logger = logging.getLogger(__name__)

class SequenceData:
    """Data class for preprocessing of .csv files into sequences that can be processed by the BESTPredictor
    """
        
    def __init__(self, data, case_identifier, activity_identifier, timestamp_identifier, **kwargs):
        
        logger.info('Initializing sequence data...')

        self.data = data if data is not None else {}
        self.case_identifier = case_identifier
        self.activity_identifier = activity_identifier
        self.timestamp_identifier = timestamp_identifier

        self.traces = None
        self.all_prefixes = None
        self.relevant_prefixes = None

        logger.info('Recoding sequence data types...')
        self.recode_types()

        logger.info('Data initialization completed!')

    @classmethod
    def from_csv(cls, load_path: os.PathLike, **kwargs) -> SequenceData:
        """Initialize SequenceData from a .csv file

        Args:
            load_path (os.PathLike): the path to the .csv file

        Returns:
            SequenceData: the new SequenceData object
        """
        data = pd.read_csv(load_path)
        return cls(data=data, **kwargs)

    @classmethod
    def from_obj(cls, obj: SequenceData, **kwargs) -> SequenceData:
        """Initialize SequenceData from an existing SequenceData object and update parameters with given kwargs

        Args:
            obj (SequenceData): the SequenceData we want to initialize from

        Raises:
            TypeError: if an object from a different class is given as obj

        Returns:
            SequenceData: the new SequenceData object
        """

        if not isinstance(obj, cls):
            raise TypeError("object must be an instance of SequenceData")
        new_init_params = copy.deepcopy(vars(obj))
        for param, value in kwargs.items():
            new_init_params[param] = value

        return cls(**new_init_params)
    
    def recode_types(self) -> pd.DataFrame:
        """Recodes the types of the timestamp (-> datetime) and activity (str) columns

        Returns:
            pd.DataFrame: the pd.DataFrame with recoded column dtypes
        """
        recoded_times = pd.to_datetime(self.data[self.timestamp_identifier], format='mixed')

        self.data.loc[:,self.timestamp_identifier] = recoded_times
        self.data.loc[:,self.activity_identifier] = self.data.loc[:,self.activity_identifier].astype(object)
        self.data.loc[:,self.activity_identifier] = self.data.loc[:,self.activity_identifier].astype(str)


    def pad_columns(self, cols_to_pad: list[str], n_pad: int = 1,  forward_pad: str | None = 'START',  backward_pad: str | None = 'END') -> pd.DataFrame:
        """Pads the existing sequence data with specified START and/or END tokens

        Args:
            cols_to_pad (list[str]): the columns to pad
            n_pad (int, optional): the length of the padding. Defaults to 1.
            forward_pad (str | None, optional): the pad token string for the START pad. 
            None results in no padding. Defaults to 'START'.
            backward_pad (str | None, optional): the pad token string for the END pad. 
            None results in no padding. Defaults to 'END'.

        Returns:
            pd.DataFrame: the padded pd.DataFrame
        """
        logger.info('Padding columns...')
        self.start_token = forward_pad
        self.end_token = backward_pad

        padded_data = self.data.copy()
        previous_dtypes = padded_data.dtypes

        padded_data = padded_data.groupby(self.case_identifier).apply(lambda x: _inner_pad(x,
                                                                    case_identifier=self.case_identifier,
                                                                    cols_to_pad=cols_to_pad,
                                                                    n_pad=n_pad,
                                                                    forward_pad=forward_pad,
                                                                    backward_pad=backward_pad))
            
        padded_data = padded_data.reset_index(drop=True)
        for col, dtype in enumerate(previous_dtypes):
            try:
                padded_data[padded_data.columns[col]] = padded_data.iloc(axis=1)[col].astype(dtype)
            except pd.errors.IntCastingNaNError:
                pass

        self.data = padded_data
        logger.info('Column padding completed!')
    
    def extract_traces(self, columns: list[str]) -> list[dict]:
        # TODO
        # include doc string
        logger.info('Generating trace data...')
        
        traces = list()

        grouped = self.data.groupby(self.case_identifier)

        for name, group in grouped:
            sequences = _extract_trace(df=group, columns=columns)
            trace = {self.case_identifier:[name]*len(group), **sequences}
            traces.append(trace)

        self.traces = traces
        logger.info('Traces generated!')
    
    def encode_activities(self, act_encoder: LabelEncoder = None):
        # TODO
        # include doc string
        
        if act_encoder is None:
            act_encoder = LabelEncoder()
            act_encoder.fit(self.data[self.activity_identifier])
            self.act_encoder = act_encoder

        self.act_mapping = dict(zip(act_encoder.classes_, act_encoder.transform(act_encoder.classes_)))
        encoded_activities = self.data[self.activity_identifier].map(self.act_mapping).fillna(-1).astype(int)
        self.data[self.activity_identifier] = encoded_activities

        self.start_activity = self.act_mapping[self.start_token]
        self.end_activity = self.act_mapping[self.end_token]
    
    def generate_prefixes(self):
        prefixes = list()

        for trace in self.traces:
            case_id = trace[self.case_identifier][0]
            activity_sequence = trace[self.activity_identifier]

            for prefix_size in range(1, len(activity_sequence)):
                current_prefix = {'case_id':case_id,
                                    'prefix':activity_sequence[:prefix_size], 
                                    'full_sequence':activity_sequence,
                                    'full_future_sequence':activity_sequence[prefix_size:]}
                prefixes.append(current_prefix)

        self.all_prefixes = prefixes
    
    def pick_relevant_prefixes(self):
        relevant_prefixes = list()
        
        for prefix_dict in self.all_prefixes:
            prefix = prefix_dict['prefix']
            full_sequence = prefix_dict['full_sequence']
            full_n_start_acts = sum([True if act==self.start_activity else False for act in full_sequence])

            prefix_n_start_acts = sum([True if act==self.start_activity else False for act in prefix])
            prefix_n_end_acts = sum([True if act==self.end_activity else False for act in prefix])

            if prefix_n_start_acts == full_n_start_acts and prefix_n_end_acts == 0:
                relevant_prefixes.append(prefix_dict)

        self.relevant_prefixes = relevant_prefixes
    
    def generate_full_sequences(self, filter_sequences: bool = True):
        if self.relevant_prefixes is None:
            raise ValueError('relevant prefixes were not generated')
        
        full_sequences = [prefix['full_sequence'] for prefix in self.relevant_prefixes]

        if filter_sequences:
            filtered_full_sequences = [_filter_start_end(full_seq, self.start_activity, self.end_activity) for full_seq in full_sequences]
            self.full_sequences = filtered_full_sequences
        else:
            self.full_sequences = full_sequences
    
    def generate_full_future_sequences(self, filter_sequences: bool = True):
        if self.relevant_prefixes is None:
            raise ValueError('relevant prefixes were not generated')
        
        full_future_sequences = [prefix['full_future_sequence'] for prefix in self.relevant_prefixes]

        if filter_sequences:
            filtered_full_future_sequences = [_filter_start_end(full_seq, self.start_activity, self.end_activity) for full_seq in full_future_sequences]
            self.full_future_sequences = filtered_full_future_sequences
        else:
            self.full_future_sequences = full_future_sequences

    def generate_next_activities(self):
        if self.relevant_prefixes is None:
            raise ValueError('relevant prefixes were not generated')
        
        next_activities = [prefix['full_sequence'][len(prefix['prefix'])] for prefix in self.relevant_prefixes]
        self.next_activities = next_activities

    def train_test_split(self, train_pct: float, cv: int = 1) -> tuple[SequenceData, SequenceData] | list[tuple[SequenceData, SequenceData]]:
        """Function for splitting an existing SequenceData object into two distinct SequenceData objects (train and test).

        Args:
            train_pct (float): Percentage share of the data that should be transferred into the training set object.
            The test set consists of the remaining sequences.
            cv (int, optional): If k-fold cross-validation should be performed where cv is the number of folds. 
            Defaults to False.

        Returns:
            tuple[SequenceData]: training and test instances of the corresponding sequences as SequenceData objects
        """
        if cv > 1: # generate k-fold cross validation datasets
            logger.info(f'Splitting train and test data ({1/cv*100:.0f}-{(1-(1/cv))*100:.0f} split with {cv} folds)')
            # split the data into cv folds
            all_ids = self.data[self.case_identifier].unique()
            np.random.shuffle(all_ids)
            id_folds = [id_fold for id_fold in _batch_samples(all_ids, cv)]
            folds = list()
            for idf_idx in range(0, len(id_folds)):
                test_ids = id_folds[idf_idx].tolist()
                train_ids = list(itertools.chain(*[id_folds[i] for i in range(0, len(id_folds)) if i != idf_idx]))
                train_data = self.data[self.data[self.case_identifier].isin(train_ids)]
                test_data = self.data[self.data[self.case_identifier].isin(test_ids)]

                train_data_obj = SequenceData.from_obj(self, data=train_data)
                test_data_obj = SequenceData.from_obj(self, data=test_data)
                folds.append(tuple([train_data_obj, test_data_obj]))

            return folds
        else:
            logger.info(f'Splitting train and test data ({train_pct*100:.0f}-{(1-train_pct)*100:.0f} split)')
            all_ids = self.data[self.case_identifier].unique()
            train_ids = np.random.choice(a=all_ids, size=int(len(all_ids)*train_pct), replace=False)
            train_set = set(train_ids)
            test_ids = [id for id in all_ids if id not in train_set]

            train_data = self.data[self.data[self.case_identifier].isin(train_ids)]
            test_data = self.data[self.data[self.case_identifier].isin(test_ids)]

            train_data_obj = SequenceData.from_obj(self, data=train_data)
            test_data_obj = SequenceData.from_obj(self, data=test_data)

            return train_data_obj, test_data_obj
    
    def get_characteristics(self) -> dict:
        
        log_characteristics = dict()

        log_characteristics['n_cases'] = self._get_n_cases()
        log_characteristics['n_events'] = self._get_n_events()
        log_characteristics['n_activities'] = self._get_n_activities()
        log_characteristics['mean_trace_len'] = float(self._get_mean_trace_len())
        log_characteristics['median_trace_len'] = float(self._get_median_trace_len())
        log_characteristics['max_trace_len'] = self._get_max_trace_len()
        log_characteristics['n_variants'] = self._get_n_variants()
        
        return log_characteristics
    
    def _get_n_cases(self) -> int:
        n_cases = self.data[self.case_identifier].nunique()
        return n_cases

    def _get_n_events(self) -> int:
        n_events = len(self.data)
        return n_events

    def _get_n_activities(self) -> int:
        n_activities = self.data[self.activity_identifier].nunique()
        return n_activities

    def _get_median_trace_len(self) -> float:
        trace_lens = self._get_trace_lens()
        trace_lens.sort()
        center_idx = int(len(trace_lens)/2)
        if len(trace_lens) % 2 == 0:
            median_trace_len = sum(trace_lens[center_idx-1:center_idx+1])/2
        else:
            median_trace_len = trace_lens[center_idx]
        return median_trace_len

    def _get_mean_trace_len(self) -> float:
        mean_trace_len = self._get_trace_lens().mean()
        return mean_trace_len

    def _get_max_trace_len(self) -> float:
        max_trace_len = int(self._get_trace_lens().max())
        return max_trace_len

    def _get_trace_lens(self) -> np.array[int]:
        trace_lens = np.array(self.data[self.case_identifier].value_counts())
        return trace_lens
    
    def _get_n_variants(self) -> int:
        variant_strings = self.data.groupby(self.case_identifier)[self.activity_identifier].apply(lambda x: ','.join(x))
        unique_variants = variant_strings.unique()
        return len(unique_variants)

def _extract_trace(df: pd.DataFrame, columns: list[str]) -> dict:
    trace = dict()
    for col in columns:
        trace[col] = df[col].tolist()

    return trace

def _inner_pad(df: pd.DataFrame,
               case_identifier: str,
               cols_to_pad: list[str],
               n_pad: int = 1, 
               forward_pad: str | None = 'START', 
               backward_pad: str | None = 'END') -> pd.DataFrame:

    if forward_pad:
        df = df.reset_index(drop=True).reindex(range(-n_pad, len(df))).reset_index(drop=True)
        df.loc[0:n_pad-1, cols_to_pad] = forward_pad
        df[case_identifier] = df[case_identifier].bfill()
    if backward_pad:
        df = df.reset_index(drop=True).reindex(range(0, len(df)+n_pad)).reset_index(drop=True)
        df.loc[len(df)-n_pad:, cols_to_pad] = backward_pad
        df[case_identifier] = df[case_identifier].ffill()
    
    return df

def _batch_samples(samples, nbatches: int):
    nsamples = len(samples)
    batchsize = math.ceil(nsamples/nbatches)
    for idx in range(0, nsamples, batchsize):
        yield samples[idx:min(idx + batchsize, nsamples)]
