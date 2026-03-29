from Preprocessing.create_tensor_pipeline import create_tensors
from Models.Markov.Markov_pipeline import fit_evaluate_Markov
from Models.LSTM.LSTM_pipeline import train_evaluate_LSTM
from Models.Transformer.Transformer_pipeline import train_evaluate_Transformer

dataset_name = 'NASA'
model = 'Markov'
seed = 42

decoding='bridge'
estimator='MC'
n_candidate=50
n_sample=50
diff=False
bridge_sampling='random'

create_tensors(dataset_name)

if model == 'Markov':
    test_dl_distance, test_mae_len = fit_evaluate_Markov(dataset_name, 
                                                         seed,
                                                         decoding,
                                                         estimator,
                                                         n_candidate,
                                                         n_sample,
                                                         diff,
                                                         bridge_sampling)
elif model == 'LSTM':
    test_dl_distance, test_mae_len = train_evaluate_LSTM(dataset_name, 
                                                         seed,
                                                         decoding,
                                                         estimator,
                                                         n_candidate,
                                                         n_sample,
                                                         diff,
                                                         bridge_sampling)

elif model == 'Transformer':
    test_dl_distance, test_mae_len = train_evaluate_Transformer(dataset_name, 
                                                         seed,
                                                         decoding,
                                                         estimator,
                                                         n_candidate,
                                                         n_sample,
                                                         diff,
                                                         bridge_sampling)
