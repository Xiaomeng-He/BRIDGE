import numpy as np
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance as norm_dl_distance


def levenshtein_similarity(pred_trace, ground_truth_trace, code_end):

    try:
        l1 = pred_trace.index(code_end)
    except ValueError:
        l1 = len(pred_trace)

    try:
        l2 = ground_truth_trace.index(code_end)
    except ValueError:
        l2 = len(ground_truth_trace)

    if max(l1, l2) == 0: return 0, 0, 1.0

    a = pred_trace[:l1+1]
    b = ground_truth_trace[:l2+1]

    norm_dl = norm_dl_distance(a, b) 
    result = 1.0 - norm_dl

    return l1, l2, result

def one_hot_windowed_sequence(current_sequence, window_size, n_activities):
    # If the prefix is less than the window size, pad it, otherwise, get the most recent W events
    if len(current_sequence) < window_size:
        current_sequence = [0] * (window_size - len(current_sequence)) + current_sequence
    else:
        current_sequence = current_sequence[-window_size:]

    # MlpPolicy expects a one hot vector as input.
    # Change the observation to onehots
    one_hot_sequence = np.zeros((window_size, n_activities))
    for i, activity in enumerate(current_sequence):
        # 0 will be a padding vector
        if activity != 0:
            one_hot_sequence[i, activity] = 1
    current_sequence = one_hot_sequence
    return current_sequence


