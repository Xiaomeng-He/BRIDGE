import numpy as np
import pandas as pd
import math
import itertools
import multiprocessing
import time
from collections import Counter
from tqdm import tqdm
from enum import Enum
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor
from ..data.sequencedata import SequenceData
from ..util.sequence_utils import _get_pattern_center, _child_matches_with_sequence, _filter_start_end

import logging
logger = logging.getLogger(__name__)

class Task(Enum):
    NAP = 'nap'
    RTP = 'rtp'
    
class BESTPredictor():
    """Prediction model using the Hierarchical Central Activity Pattern prediction algorithm.
    BEST is capable of predicting next activities as well as remaining traces for sequences of activities
    """

    def __init__(self, max_pattern_size, process_stage_width_percentage, min_freq, prune_func):

        params = {'max_pattern_size':max_pattern_size,
                        'process_stage_width_percentage':process_stage_width_percentage,
                        'min_freq':min_freq,
                        'prune_func':prune_func}
        logger.info(f'Initializing prediction model - { {k:v for k,v in params.items()} }')
        if max_pattern_size % 2 == 0 or max_pattern_size <= 1:
            raise ValueError('max_pattern_size must be an odd integer > 1')

        self.max_pattern_size = max_pattern_size
        self.process_stage_width_percentage = process_stage_width_percentage
        self.min_freq = min_freq
        self.prune_func = prune_func
        
        self._pattern_sizes = [_ for _ in range(1, self.max_pattern_size+1, 2)]
        self._padding_size = int(max_pattern_size/2)+1
        
        self.data_train = None
        self.data_test = None
        self.hca_patterns = None

        # length of chosen pattern in prediction tracker
        self.choice_tracker_nap = {'prob':[], 'len':[], 'dist':[]}
        self.choice_tracker_rtp = {'prob':[], 'len':[], 'dist':[]}
        
    def fit(self) -> None:
        """Fitting the model to X (training data). This involves the pattern generation as well as matching
        of the patterns with their respective children/parents to be able to construct a hierarchical tree of
        central activity patterns
        """
        self.generate_patterns(self.data_train)
        self.find_child_patterns()

        unpruned_nodes = dict()
        pruned_nodes = dict()
        stage_trees = dict()

        logger.info(f'Building pattern tree for {len(self._stages)} stages...')

        for stage in tqdm(self._stages):
            logger.debug(f'matching in stage {stage+1}/{len(self._stages)}')
            stage_matches = self._matches_per_stage[stage]
            stage_dict_matches = _get_matches_dict(pattern=(), all_matches=stage_matches, max_pattern_size=self.max_pattern_size, min_freq=self.min_freq)
            stage_trees[stage] = stage_dict_matches

            tree_nodes = extract_tree_with_pruning(stage_dict_matches, prune_func=self.prune_func)

            current_unpruned_nodes = {key:node for key, node in tree_nodes.items() if node['pruned'] is False}
            current_pruned_nodes = {key:node for key, node in tree_nodes.items() if node['pruned'] is True}
            pruned_nodes[stage] = current_pruned_nodes
            unpruned_nodes[stage] = current_unpruned_nodes
        
        logger.info('Pattern tree built!')

        self._pruned_nodes = pruned_nodes
        self._unpruned_nodes = unpruned_nodes
        self._stage_trees = stage_trees

    def predict(self, eval_pattern_size: int, task: str, break_buffer: float, filter_tokens: bool, ncores: int,
                prefixes: int) -> list[list[int]]:
        """Generates predictions for the chosen task (next activity prediction or remaining trace prediction)
        for a given set of sequences X.

        Args:
            task (str): The prediction task. Specify either 'nap' for next activity prediction or 'rtp' for
            remaining trace prediction
            break_buffer (float): factor by which a predicted sequence can overflow the length of the longest prefix
            seen in the training data

        Raises:
            NotImplementedError: If a different task other that 'nap' or 'rtp' is chosen

        Returns:
            np.array: The full predicted sequences (containing the given sequences with the appended predictions)
        """
        logger.info(f'Starting prediction - {task.upper()}')
        try:
            task = Task(task)
        except ValueError:
            raise ValueError(f'invalid task: {task} - only next activity prediction (nap) and \
                                      remaining trace prediction (rtp) are valid tasks')

        if task==Task.RTP:
            max_prefix_len = max([len(prefix) for prefix in prefixes])

            predicted_traces = list()
            if ncores == 1:
                for prefix in tqdm(prefixes):
                    prefix_sequence = prefix
                    pred_sequence = self._predict_sequence(prefix=prefix_sequence, eval_pattern_size=eval_pattern_size,
                                                        break_after_seq_len=break_buffer*max_prefix_len)
                    
                    predicted_traces.append(pred_sequence)
            else:
                prefix_batches = [pb for pb in self._batch_prefixes(ncores)]
                batch_lens = [len(b) for b in self._batch_prefixes(ncores)]
                with multiprocessing.Manager() as manager:
                    progress_dict = manager.dict({i: 0 for i in range(ncores)})

                    with ProcessPoolExecutor(max_workers=ncores) as executor:
                        batch_predictions = [executor.submit(self._batch_predict_sequence, 
                                                             eval_pattern_size, 
                                                             prefix_batches[i], 
                                                             break_buffer*max_prefix_len, 
                                                             i,
                                                             progress_dict) for i in range(ncores)]
                        
                        _progress_monitor(progress_dict, ncores, batch_lens)

                predicted_traces = list(itertools.chain(*[f.result() for f in batch_predictions]))

            predictions = predicted_traces

            if filter_tokens:
                filtered_predictions = list()
                for pred in predictions:

                    try:
                        filtered_predictions.append(_filter_start_end(pred, self.start_activity, self.end_activity))
                    except TypeError:
                        filtered_predictions.append(None)

                predictions = filtered_predictions

        return predictions

    def _predict_sequence(self, eval_pattern_size: int, prefix: list[int], break_after_seq_len: int = 10e5, verbose: bool = False) -> list[int]:
        """Predicts the remaining activities for a given prefix

        Args:
            prefix (list[int]): A prefix to predict remaining activities for
        
        Returns:
           list[int]: the predicted sequence containing the prefix and the predicted remaining activities
        """    
        predicted_sequence = prefix.copy()
        try:
            last_start = len(predicted_sequence) - predicted_sequence[::-1].index(self.start_activity) - 1
        except ValueError as v_error:
            v_error.args = ('test_sequence contains no start_activities - predict only for non-left-truncated sequences' ,)
            raise

        max_process_stage = max(self._unpruned_nodes.keys())
        min_process_stage = min(self._unpruned_nodes.keys())

        # start prediction loop
        while predicted_sequence[-1] != self.end_activity and len(predicted_sequence) < break_after_seq_len:
            current_process_stage = min(int((len(predicted_sequence[last_start:]) - 1) / self._abs_process_stage_width), max_process_stage)
            
            current_prediction, pattern_attributes = self._pred_for_process_stage(stage=current_process_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, verbose=verbose)
            increase_stage, decrease_stage = current_process_stage + 1, current_process_stage - 1

            while len(current_prediction)==0 and not (increase_stage>max_process_stage and decrease_stage<min_process_stage):
                # look for next higher stage
                increase_stage = min(increase_stage, max_process_stage)
                current_increase_prediction, increase_pattern_attributes = self._pred_for_process_stage(stage=increase_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, verbose=verbose)
                increase_stage += 1

                # look for next lower stage
                decrease_stage = max(decrease_stage, min_process_stage)
                current_decrease_prediction, decrease_pattern_attributes = self._pred_for_process_stage(stage=decrease_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, verbose=verbose)
                decrease_stage -= 1

                increase_prob = increase_pattern_attributes.get('prob')
                decrease_prob = decrease_pattern_attributes.get('prob')

                # if both find something take the one with the higher probability
                if decrease_prob is not None and increase_prob is not None:
                    if increase_prob > decrease_prob:
                        current_prediction = current_increase_prediction
                        pattern_attributes = increase_pattern_attributes
                    else:
                        current_prediction = current_decrease_prediction
                        pattern_attributes = decrease_pattern_attributes
                elif increase_prob is not None:
                    current_prediction = current_increase_prediction
                    pattern_attributes = increase_pattern_attributes
                elif decrease_prob is not None:
                    current_prediction = current_decrease_prediction
                    pattern_attributes = decrease_pattern_attributes
                else:
                    # nothing has been found in any of the other stages - repeat with different stages
                    pass
        
            if len(current_prediction) == 0:
                if len(predicted_sequence) == len(prefix):
                    logger.debug(f'We did not find any predicted activities for prefix {prefix}')
                else:
                    logger.debug(f'We did not find any further predicted activities for running prediction {predicted_sequence}')
                return predicted_sequence[len(prefix):]
            
            for choice_metric in pattern_attributes.keys():
                self.choice_tracker_rtp[choice_metric].append(pattern_attributes[choice_metric])

            predicted_sequence.extend(current_prediction[1:]) # whole pattern is appended to the prediction

        return predicted_sequence[len(prefix):]
    
    def _batch_predict_sequence(self, eval_pattern_size, prefixes, break_after_seq_len, proc_id, progress_dict):
        predicted_traces = list()
        for prefix_idx, prefix in enumerate(prefixes):
            prefix_sequence = prefix['prefix']
            pred_sequence = self._predict_sequence(prefix=prefix_sequence,
                                                   eval_pattern_size=eval_pattern_size,
                                                   break_after_seq_len=break_after_seq_len)
            progress_dict[proc_id] = prefix_idx + 1
            
            predicted_traces.append(pred_sequence)
        return predicted_traces


    def _predict_activity(self, eval_pattern_size: int, prefix: list[int], verbose: bool = False) -> int:
        """Predicts the next activity for a given sequence

        Args:
            prefix (list[int]): A prefix to predict remaining activities for
        
        Returns:
            int: the predicted activity
        """

        predicted_sequence = prefix.copy()
        try:
            last_start = len(predicted_sequence) - predicted_sequence[::-1].index(self.start_activity) - 1
        except ValueError as v_error:
            v_error.args = ('test_sequence contains no start_activities - predict only for non-left-truncated sequences' ,)
            raise

        max_process_stage = max(self._unpruned_nodes.keys())
        min_process_stage = min(self._unpruned_nodes.keys())

        # start prediction loop
        if predicted_sequence[-1] != self.end_activity:
            current_process_stage = min(int((len(predicted_sequence[last_start:]) - 1) / self._abs_process_stage_width), max_process_stage)
            
            current_prediction, pattern_attributes = self._pred_for_process_stage(stage=current_process_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, verbose=verbose)
            increase_stage, decrease_stage = current_process_stage + 1, current_process_stage - 1

            while len(current_prediction)==0 and not (increase_stage>max_process_stage and decrease_stage<min_process_stage):
                # look for next higher stage
                increase_stage = min(increase_stage, max_process_stage)
                current_increase_prediction, increase_pattern_attributes = self._pred_for_process_stage(stage=increase_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, verbose=verbose)
                increase_stage += 1

                # look for next lower stage
                decrease_stage = max(decrease_stage, min_process_stage)
                current_decrease_prediction, decrease_pattern_attributes = self._pred_for_process_stage(stage=decrease_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, verbose=verbose)
                decrease_stage -= 1

                increase_prob = increase_pattern_attributes.get('prob')
                decrease_prob = decrease_pattern_attributes.get('prob')

                # if both find something take the one with the higher probability
                if decrease_prob is not None and increase_prob is not None:
                    if increase_prob > decrease_prob:
                        current_prediction = current_increase_prediction
                        pattern_attributes = increase_pattern_attributes
                    else:
                        current_prediction = current_decrease_prediction
                        pattern_attributes = decrease_pattern_attributes
                elif increase_prob is not None:
                    current_prediction = current_increase_prediction
                    pattern_attributes = increase_pattern_attributes
                elif decrease_prob is not None:
                    current_prediction = current_decrease_prediction
                    pattern_attributes = decrease_pattern_attributes
                else:
                    # nothing has been found in any of the other stages - repeat with different stages
                    pass
        
            if len(current_prediction) == 0:
                if verbose:
                    print("WE DID NOT FIND ANY SUITABLE PREDICTION FOR THE CURRENT SEQUENCE ANYWHERE")
                return
            
            for choice_metric in pattern_attributes.keys():
                self.choice_tracker_nap[choice_metric].append(pattern_attributes[choice_metric])
        return current_prediction[1]

    def _batch_predict_activity(self, eval_pattern_size, prefixes, proc_id, progress_dict, **kwargs):
        predicted_activities = list()
        for prefix_idx, prefix in enumerate(prefixes):
            prefix_sequence = prefix['prefix']
            pred_activity = self._predict_activity(prefix=prefix_sequence, 
                                                   eval_pattern_size=eval_pattern_size, 
                                                   **kwargs)
            
            progress_dict[proc_id] = prefix_idx + 1
            
            predicted_activities.append(pred_activity)
        return predicted_activities

    # --- MOD ---    
    def load_data(self, train: SequenceData):
        self.data_train = train
        # self.data_test = test
    # --- MOD ---

    def prepare_train(self):
        # TODO
        # include doc string
        logger.info('Preparing training data...')
        if self.data_train is None:
            raise ValueError('data not found - make sure to load train and test data with load_data()')
        
        self.train_max_trace_len = self.data_train._get_max_trace_len()
        self.data_train.pad_columns(cols_to_pad=[self.data_train.activity_identifier], n_pad=self._padding_size)
        act_idx = self.data_train.data.groupby(self.data_train.case_identifier).apply(lambda x: pd.Series(range(-self._padding_size, 
                                                                                          len(x)-self._padding_size)))
        self.data_train.data['activity_idx'] = act_idx.reset_index(drop=True)

        # forward and backward fill timestamp column
        self.data_train.data[self.data_train.timestamp_identifier] = self.data_train.data.groupby(self.data_train.case_identifier)[self.data_train.timestamp_identifier].ffill()
        self.data_train.data[self.data_train.timestamp_identifier] = self.data_train.data.groupby(self.data_train.case_identifier)[self.data_train.timestamp_identifier].bfill()

        self.data_train.encode_activities()
        self.data_train.extract_traces(columns=[self.data_train.activity_identifier])
        self.start_activity = self.data_train.start_activity
        self.end_activity = self.data_train.end_activity

        self.data_train.generate_prefixes()
        self.data_train.pick_relevant_prefixes()

        self.max_prefix_len = max([len(prefix['prefix']) for prefix in self.data_train.relevant_prefixes])

        logger.info('Training data prepared!')

    def generate_patterns(self, X: SequenceData):
        """Generates sequential patterns of activities inside traces by process stage. The generated patterns are
        given as a dict with pattern sizes as keys. Each pattern is a tuple of two elements where the first element is
        the pattern as a list of ints and the second element is the respective process stage
        (with the absolute width being specified by self.process_stage_width_percentage)

        Args:
            X (SequenceData): The SequenceData object we generate the sequential patterns for. This has to be prepared
            such that it has generated traces as class attribute (SequenceData.prepare_train())
        """
        logger.info(f'Generating patterns from trace data...')
        patterns_by_size = dict()
        patterns = list()
        
        abs_process_stage_width = int((self.train_max_trace_len + 2)*self.process_stage_width_percentage) + 1
        self._abs_process_stage_width = abs_process_stage_width

        for pattern_size in self._pattern_sizes:
            current_size_patterns = list()
            for trace in X.traces:
                trace_to_process = trace[X.activity_identifier]
                trace_index_start = self._padding_size-int(pattern_size/2)-1
                trace_index_end = len(trace_to_process)-self._padding_size+int(pattern_size/2)+1
                trace_to_process = trace_to_process[trace_index_start:trace_index_end] # ensure no patterns of only END and START tokens
                
                # current_size_patterns.extend(make_pattern_stage_tuples(trace_to_process, pattern_size=pattern_size,
                #                                    pad_short_seqs=True, process_stage_width=self.process_stage_width))
                current_size_patterns.extend(make_pattern_stage_tuples(trace_to_process, pattern_size=pattern_size,
                                                   pad_short_seqs=True, process_stage_width=self._abs_process_stage_width))
                
            patterns_by_size[pattern_size] = current_size_patterns
            patterns.extend(current_size_patterns)
        
        self._hca_patterns_by_size = patterns_by_size
        self.hca_patterns = patterns

        pattern_strings = [(','.join([str(el) for el in pattern[0]]), pattern[1]) for pattern in patterns]
        pattern_freqs_by_stage = calc_stage_wise_freqs(pattern_strings)
        self._hca_pattern_freqs_by_stage = pattern_freqs_by_stage

        logger.info(f'Pattern generation completed!')

    def find_child_patterns(self):
        
        logger.info(f'Searching for child patterns...')
        all_stage_values = [pattern[1] for pattern in self.hca_patterns]
        stages = range(min(all_stage_values), max(all_stage_values)+1)
        self._stages = stages

        original_sizes = [0] + self._pattern_sizes[:-1]
        extended_sizes = self._pattern_sizes[:]

        matches_per_stage = dict()


        for stage in stages:
            unique_ext_patterns = list()
            all_matches = dict()
            for orig_size, ext_size in zip(original_sizes, extended_sizes):

                if len(unique_ext_patterns) == 0:
                    if orig_size == 0:
                        orig_patterns = [[]]
                    else:
                        orig_patterns = [pattern[0] for pattern in self._hca_patterns_by_size[orig_size] 
                                        if pattern[1]==stage]
                    
                    orig_pattern_counts = dict(Counter(map(tuple, orig_patterns)))
                    unique_orig_patterns = {pattern:{'prob':1,
                                                     'global_prob':freq/sum(orig_pattern_counts.values()),
                                                     'freq':freq}
                                            for pattern, freq in orig_pattern_counts.items()}
                else:
                    unique_orig_patterns = unique_ext_patterns.copy()

                ext_patterns = [pattern[0] for pattern in self._hca_patterns_by_size[ext_size] 
                                if pattern[1]==stage]
                ext_pattern_counts = dict(Counter(map(tuple, ext_patterns)))
                unique_ext_patterns = {pattern:{'prob':float(),
                                                'global_prob':freq/sum(ext_pattern_counts.values()),
                                                'freq':freq}
                                       for pattern, freq in ext_pattern_counts.items()}
                
                # get matching extended patterns for each original pattern
                size_matches = dict()
                for orig, orig_attrs in unique_orig_patterns.items():
                    matches = dict()
                    # match_counts = list()
                    if orig == ():
                    # if orig == []:
                        pattern_string = ''
                    else:
                        pattern_string = ','.join([str(el) for el in orig])
                    
                    if pattern_string not in all_matches.keys():
                        for ext, ext_attrs in unique_ext_patterns.items():
                            if orig == ext[1:-1]:
                                matches[ext] = ext_attrs
                        # scale probabilities to unit sum (for the matching extended patterns)
                        probs_to_rescale = [val['global_prob'] for val in matches.values()]
                        rescaled_probs = _rescale_probs(probs_to_rescale)
                        for idx, match_attrs in enumerate(matches.values()):
                            match_attrs['prob'] = rescaled_probs[idx]
                        
                        for ext, ext_attrs in matches.items():
                            unique_ext_patterns[ext] = ext_attrs
                        size_matches[pattern_string] = {'prob':orig_attrs['prob'], 
                                                        'global_prob':orig_attrs['global_prob'], 
                                                        'freq':orig_attrs['freq'], 
                                                        'matches':matches}
                all_matches[orig_size] = size_matches
            matches_per_stage[stage] = all_matches
        
        self._matches_per_stage = matches_per_stage
        logger.info(f'Child pattern search completed!')

    def _pred_for_process_stage(self, eval_pattern_size: int, stage: int, sequence: list[int], verbose: bool = False) -> list[int]:
        
        if self.prune_func is not None:
        
            current_important_patterns = self._unpruned_nodes[stage]

            empty_key = [key for key, val in current_important_patterns.items() if val['name']=='']
            if empty_key:
                if verbose:
                    print('i deleted the empty key, sir :)')
                current_important_patterns.pop(empty_key[0])

            center_activity = sequence[-1]
            # check for children of center activity
            children = {key:val 
                        for key, val in current_important_patterns.items() 
                        if _get_pattern_center(val) == center_activity}

            # we exclude single activity children here (because we check for matches with the center activity (not with the left part))
            non_atomic_patterns = [key for key, child in children.items() if len(child['name'].split(',')) > 1]
            children = {child_key: children[child_key] for child_key in non_atomic_patterns}

            if verbose:
                print(children)

            applying_children = {key:val 
                                    for key, val in children.items() 
                                    if _child_matches_with_sequence(val, sequence)}
            
            if len(applying_children)==0:
                return [], 0
            
            probs = [val['prob'] for val in applying_children.values()]
            dists = [val['total_log_rpif_dist'] for val in applying_children.values()]
            lens_children = [len(val['name'].split(',')) for val in applying_children.values()]

            max_prob = max(probs)
            argmax_prob_indices = [idx for idx, p in enumerate(probs) if p==max_prob]
            argmax_children = [[int(act) for act in [val['name'] for val in applying_children.values()][pick].split(',')] for pick in argmax_prob_indices]
            argmax_children_dists = [dists[idx] for idx in argmax_prob_indices]

            min_dist = min(argmax_children_dists)
            argmin_dist_indices = [idx for idx, d in enumerate([dists[i] for i in argmax_prob_indices]) if d==min_dist]
            min_dist_pick = np.argmin(argmax_children_dists)        
        else:
            
            method = 'PROB_LEN_DIST'
            # method = 'PROB_DIST'

            all_applying_children = self.extract_matching_patterns(stage, sequence)

            # kick single activity patterns from extracted matching patterns
            all_applying_children = all_applying_children[1:]
            
            lens_all_children = [len(p['name'].split(',')) for p in all_applying_children]
            applying_children = [c for c, c_len in zip(all_applying_children, lens_all_children) if c_len <= eval_pattern_size]

            if len(applying_children)==0:
                return [], {}

            # get probs, dists and lens
            probs = [p['prob'] for p in applying_children]
            global_probs = [p['global_prob'] for p in applying_children]
            dists = [p['total_log_rpif_dist'] for p in applying_children]
            lens_children = [len(p['name'].split(',')) for p in applying_children]

            # get max prob children (many conditional probs of 1.0 for the longer patterns)
            max_prob = max(probs)
            argmax_prob_indices = [idx for idx, p in enumerate(probs) if p==max_prob]
            argmax_prob_children_lens = [lens_children[idx] for idx in argmax_prob_indices]
            argmax_prob_dists = [dists[idx] for idx in argmax_prob_indices]
            argmax_prob_global_probs = [global_probs[idx] for idx in argmax_prob_indices]
            max_len = max(argmax_prob_children_lens)
            max_global_prob = max(argmax_prob_global_probs)
            argmax_prob_argmax_len_indices = [idx for idx, l in enumerate(lens_children) if l==max_len and idx in argmax_prob_indices]
            argmax_prob_argmax_len_dists = [dists[idx] for idx in argmax_prob_argmax_len_indices]
            min_dist = min(argmax_prob_dists)
            min_dist_max_len = min(argmax_prob_argmax_len_dists)
            argmax_prob_argmax_len_argmin_dist_indices = [idx for idx, d in enumerate(dists) if d==min_dist_max_len and idx in argmax_prob_argmax_len_indices]

            argmax_prob_argmin_dist_indices = [idx for idx, d in enumerate(dists) if d==min_dist and idx in argmax_prob_indices]
            argmax_prob_argmax_global_prob_indices = [idx for idx, p in enumerate(global_probs) if p==max_global_prob and idx in argmax_prob_indices]

            if method == 'PROB_LEN_DIST':
                # candidate_children = [[int(act) for act in [p['name'] for p in applying_children][pick].split(',')] for pick in argmax_prob_argmax_len_argmin_dist_indices] # with max prob - max len - min dist
                candidate_children = [[p for p in applying_children][pick] for pick in argmax_prob_argmax_len_argmin_dist_indices] # with max prob - max len - min dist

                if len(argmax_prob_argmax_len_argmin_dist_indices) > 1:
                    # pattern_lens = [len(c) for c in candidate_children]
                    pattern_lens = [len([int(act) for act in c['name'].split(',')]) for c in candidate_children]
                    logger.debug(f'we have {len(argmax_prob_argmax_len_argmin_dist_indices)} patterns with same cond prob of {max_prob}, len of {max_len} and min distance of {min_dist_max_len:.4f} - choosing randomly')
                    picked_child = candidate_children[np.random.choice(range(0, len(candidate_children)))]
                else:
                    picked_child = candidate_children[0]

            elif method == 'PROB_DIST':        
                # candidate_children = [[int(act) for act in [p['name'] for p in applying_children][pick].split(',')] for pick in argmax_prob_argmin_dist_indices] # with max prob - min dist
                candidate_children = [[p for p in applying_children][pick] for pick in argmax_prob_argmin_dist_indices]

                if len(argmax_prob_argmin_dist_indices) > 1:
                    # pattern_lens = [len(c) for c in candidate_children]
                    pattern_lens = [len([int(act) for act in c['name'].split(',')]) for c in candidate_children]
                    logger.debug(f'we have {len(argmax_prob_argmin_dist_indices)} patterns with same cond prob of {max_prob} and min distance of {min_dist:.4f} - they have lens {pattern_lens} - choosing randomly')
                    picked_child = candidate_children[np.random.choice(range(0, len(candidate_children)))]
                else:
                    picked_child = candidate_children[0]
        
        for prob, length in zip(probs, lens_children):

            if verbose:
                print(f"child prob: {prob}, child len: {length}")

        picked_pattern = [int(act) for act in picked_child['name'].split(',')]

        pred = picked_pattern[math.floor(len(picked_pattern)/2):]

        return pred, {'prob':picked_child['prob'], 'len':len(picked_pattern), 'dist':picked_child['total_log_rpif_dist']}
    
    def _batch_prefixes(self, nbatches: int):
        nprefixes = len(self.data_test.relevant_prefixes)
        batchsize = math.ceil(nprefixes/nbatches)
        for ndx in range(0, nprefixes, batchsize):
            yield self.data_test.relevant_prefixes[ndx:min(ndx + batchsize, nprefixes)]
    
    def extract_matching_patterns(self, process_stage: int, sequence: list[int], current_size: int = 1, current_tree: dict = None, total_log_rpif_dist: float = 0) -> dict:
        
        # initial tree lookup
        if current_tree is None:
            current_tree = self._stage_trees[process_stage]
        
        # look in the tree for the first matching node
        # tree structure is a dictionary with pattern sizes - we only look on the level that equals our current size
        n_matching_elements = math.ceil(current_size/2)

        sequence_to_match = sequence[-n_matching_elements:]
        tree_match_string = ','.join([str(el) for el in sequence_to_match])
        matching_child_idxs = [idx for idx, child in enumerate(current_tree['children'])
                          if ','.join(child['name'].split(',')[:n_matching_elements])==tree_match_string]
        matching_nodes = [current_tree['children'][idx] for idx in matching_child_idxs]

        if any([len(node['name'].split(',')) != current_size for node in matching_nodes]):
            raise ValueError('sizes are not handled correctly here!')
        
        for mn in matching_nodes[:]: # loop over shallow copy of list of matching nodes
            mn['total_log_rpif_dist'] = total_log_rpif_dist + mn['log_rpif_dist']

            if current_size < self.max_pattern_size:
                matching_nodes.extend(self.extract_matching_patterns(process_stage, sequence, current_size+2, mn, mn['total_log_rpif_dist']))

        return matching_nodes

def _get_matches_dict(pattern: tuple[int], all_matches: dict, max_pattern_size: int, min_k: int = None, max_k: int = None, min_freq: float = None) -> dict:
    """Recursive search of pattern matches given a starting pattern. For each match, we calculate the conditional probability
    (occurrence probability given the parent pattern). In each call we identify the matching patterns and execute the search
    for each of the matches until we find no matches in the next bigger pattern size.

    Args:
        pattern (tuple[int]): the base pattern we mine the recursive match dictionary for.
        all_matches (dict): dictionary of patterns that holds patterns for different pattern sizes.
        The pattern sizes are the first level of the dictionary, actual patterns are on the second level, respectively, 
        with their corresponding matches as values (list of matching patterns that are in format list[int]).
        max_pattern_size (int): model parameter - maximum pattern size mined in the model
        min_k (int, optional): model parameter - minimum patterns to keep per match. Defaults to None.
        max_k (int, optional): model parameter - maximum patterns to keep per match. Cannot be combined with min_freq. Defaults to None.
        min_freq (float, optional): model parameter - minimum frequence for patterns to keep per match.
        Cannot be combined with max_k. Can be combined with min_k. Defaults to None.

    Returns:
        dict: recursive dict of matches starting at the given pattern. matches found are from next bigger pattern size.
    """
    assert min_freq is None or (min_freq > 0 and min_freq <= 1), 'pick min_freq between 0 and 1'
    assert not (min_freq is not None and max_k is not None), 'specify either max_k for top-k selection or min_freq for minimal relative frequency selection'

    matches = dict()

    current_pattern_size = len(pattern)

    if current_pattern_size > 1:
        parent_pattern = pattern[1:-1]
    else:
        parent_pattern = None
    
    # identify matches
    if current_pattern_size <= max_pattern_size - 2:
        pattern_string = ','.join([str(el) for el in pattern])
        current_matches = all_matches[current_pattern_size][pattern_string]['matches']
        cond_prob = all_matches[current_pattern_size][pattern_string]['prob']
        global_prob = all_matches[current_pattern_size][pattern_string]['global_prob']
        freq = all_matches[current_pattern_size][pattern_string]['freq']
    else:
        parent_pattern_string = ','.join([str(el) for el in parent_pattern])
        cond_prob = all_matches[current_pattern_size-2][parent_pattern_string]['matches'][pattern]['prob']
        global_prob = all_matches[current_pattern_size-2][parent_pattern_string]['matches'][pattern]['global_prob']
        freq = all_matches[current_pattern_size-2][parent_pattern_string]['matches'][pattern]['freq']
        current_matches = dict()

    if max_k:
        current_matches = {key:val for idx, (key, val) in enumerate(current_matches.items()) if idx < max_k}

    if min_freq:
        if min_k:
            current_matches = {key:val for idx, (key, val) in enumerate(current_matches.items()) if val['prob'] >= min_freq or idx < min_k}
        else:
            current_matches = {key:val for key, val in current_matches.items() if val['prob'] >= min_freq}
        
    matches['name'] = ','.join([str(item) for item in pattern])
    matches['prob'] = cond_prob
    matches['global_prob'] = global_prob
    matches['freq'] = freq
    if parent_pattern:
        len_diff = len(pattern) - len(parent_pattern)
        matches['len_diff'] = len_diff
        matches['rpif_dist'] = math.exp(0.5 * len_diff) / cond_prob
        matches['log_rpif_dist'] = math.log(matches['rpif_dist'])
    else:
        matches['len_diff'] = None
        matches['rpif_dist'] = 1 / cond_prob
        matches['log_rpif_dist'] = math.log(matches['rpif_dist'])
    matches['children'] = list()

    for match in current_matches.keys():
        children = _get_matches_dict(match, all_matches, max_pattern_size, min_k, max_k, min_freq)
        if len(children) > 0:
            matches['children'].append(children)

    return matches

def _get_matches_from_dict(match_dict: dict) -> int:
    matches = list()
    children = match_dict['children']
    if len(children) > 0:
        for child in children:
            matches.append(child['name'])
            matches.extend(_get_matches_from_dict(child))

    return matches

def calc_stage_wise_freqs(pattern_stage_tuples: tuple[str,int]) -> pd.DataFrame:
    all_stage_values = [nst[1] for nst in pattern_stage_tuples]
    stages = range(min(all_stage_values), max(all_stage_values)+1)

    stage_wise_freqs = dict()

    for stage in stages:
        stage_patterns = [pst for pst in pattern_stage_tuples if pst[1]==stage]
        stage_pattern_freqs = pd.DataFrame({'pattern':[sn[0] for sn in stage_patterns], 
                                          'counter':[1]*len(stage_patterns)}).groupby('pattern').apply(lambda x: x['counter'].sum()).reset_index(name='freq')
        stage_wise_freqs[stage] = stage_pattern_freqs

    return stage_wise_freqs

def make_pattern_stage_tuples(vector: list[int], pattern_size: int, pad_short_seqs: bool, process_stage_width: int) -> tuple[list[int], int]:
    # TODO
    # include doc string

    assert pattern_size % 2 != 0, 'pick an uneven pattern size'

    if pad_short_seqs and pattern_size > len(vector):
        # this transformation makes sense if we pad the sequences with END tokens
        # if no padding is being performed beforehand this introduces non-existent 
        # self-loops of the last element in the shorter sequences
        vector = vector + [vector[-1]] * (pattern_size - len(vector))
    
    assert pattern_size <= len(vector), 'pattern size cannot be bigger than the vector - for bypass set pad_short_seqs=True'
    patterns = [vector[idx:idx+pattern_size] for idx in range(0, len(vector)-(pattern_size-1))]
    process_stages = [int(act_idx/process_stage_width) for act_idx in range(0, len(vector)-(pattern_size-1))]

    pattern_stage_tuples = [(p, stage) for p, stage in zip(patterns, process_stages)]

    return pattern_stage_tuples

def extract_tree_with_pruning(match_dict: dict, root_coords: list[float, float] = None, level: int = 0, node_idx: int = 1, legend: dict = None, prune_func = None, node_pruned = False) -> dict:
    if legend is None:
        legend = dict()
    
    x_range = len(_get_matches_from_dict(match_dict))
    children = match_dict['children']
    node_pruned = False # this leads to pruning some patterns but possibly not pruning their descendants - if we not execute this line - all descendants of a pruned pattern are also automatically pruned
    
    if not root_coords:
        root_coords = [x_range/2, 0]
        x_coords = np.linspace(0, x_range, len(children))
    else:
        x_coords = np.linspace(-x_range/2, x_range/2, len(children)) + root_coords[0]

    # early return if minimum prob is not met
    if prune_func:
        prob = match_dict['prob']
        if level > 1:
            cutoff = prune_func(level-1)
            if prob < cutoff:
                node_pruned = True
    
    legend[node_idx] = {'name':match_dict['name'],
                        'total_log_rpif_dist':root_coords[1],
                        'local_log_rpif_dist':match_dict['log_rpif_dist'],
                        'node_level':level,
                        'n_children':len(children),
                        'prob':match_dict['prob'],
                        # 'freq':match_dict['freq'],
                        'pruned':node_pruned}
    
    for child, x_coord in zip(children, x_coords):              # for each child plot the next level
        extract_tree_with_pruning(child, [x_coord, root_coords[1] + child['log_rpif_dist']], level+1, node_idx+1, legend=legend, prune_func=prune_func, node_pruned = node_pruned)
        node_idx += len(_get_matches_from_dict(child))+1
    
    return legend

def _rescale_probs(probs):
    probs = np.array([_ for _ in probs])
    probs = probs/sum(probs)
    return probs

def _progress_monitor(progress_dict, num_workers, batch_lens, update_freq = 0.1):
    """Monitoring dictionary to manage and display multiple progress bars updating in update_freq (sec)."""
    bars = [tqdm(total=batch_len, position=i, desc=f"Batch {i+1}") for i, batch_len in zip(range(0, num_workers), batch_lens)]
    
    while any(p < batch_len for p, batch_len in zip(progress_dict.values(), batch_lens)):
        time.sleep(update_freq)
        for i in range(num_workers):
            if bars[i].n < batch_lens[i]:
                bars[i].n = progress_dict[i]
                bars[i].refresh()
    
    for bar in bars:
        bar.close()
