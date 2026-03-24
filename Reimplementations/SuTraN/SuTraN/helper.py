import torch
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance as norm_dl_distance

def load_checkpoint(model, path_to_checkpoint, train_or_eval, lr):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(device)

    if (train_or_eval!= 'train') and (train_or_eval!= 'eval'):
        print("train_or_eval argument should be either 'train' or 'eval'.")
        return -1, -1, -1, -1

    checkpoint = torch.load(path_to_checkpoint)
    # Loading saved weights of the model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # Loading saved state of the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    # optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Loading number of last epoch that the saved model was trained for. 
    final_epoch_trained = checkpoint['epoch:']
    # Last loss of the trained model. 
    final_loss = checkpoint['loss']

    if train_or_eval == 'train':
        model.train()
    else: 
        model.eval()
        
    return model, optimizer, final_epoch_trained, final_loss

def evaluate(model,
            window_size,
            mean_std_ttne,
            mean_std_tsp,
            mean_std_tss,
            dataloader,
            device):

    model.eval()

    # initialize losses and performance metrics
    epoch_dl_distance, epoch_mea_len = 0.0, 0.0

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):

        for batch in dataloader:

            # load data
            cate_prefix, nume_prefix,\
                dec_input_act, dec_input_time,\
                            tgt_act = batch
            
            # nn.Embedding requires IntTensor or LongTensor
            cate_prefix = cate_prefix.long().to(device)
            dec_input_act = dec_input_act.long().to(device)
            # In nn.CrossEntropyLoss, the target data type is required to be long 
            # when using class indices.
            tgt_act = tgt_act.long().to(device)
            dec_input_time = dec_input_time.float().to(device)
            nume_prefix = nume_prefix.float().to(device)

            inputs = [cate_prefix, nume_prefix, dec_input_act, dec_input_time]

            act_predictions, ttne_pred, rrt_pred  = model(inputs,
                                                    window_size,
                                                    mean_std_ttne,
                                                    mean_std_tsp,
                                                    mean_std_tss)
            
            # calculate performance metrics
            dl_distance, mae_len = performance_metrics(act_predictions, tgt_act)
            
            # sum up losses and metrics from all batches
            epoch_dl_distance += dl_distance
            epoch_mea_len += mae_len.item()

    # compute average losses and metrics over the batch
    avg_dl_distance = epoch_dl_distance / len(dataloader)
    avg_mae_len = epoch_mea_len / len(dataloader)
    
    return avg_dl_distance, avg_mae_len

def performance_metrics(act_predictions, 
                        tgt_act,
                        eoc_index = int(3)):
    """
    Compute three performance metrics for suffix prediction:  
    - dl_distance: Normalized Damerau-Levenshtein distance between predicted and 
      ground truth activity label suffixes.
    - mae: Mean Absolute Error between predicted suffix length and ground truth suffix length.

    """
    # act_predictions = act_probs.argmax(2)
    # get batch_size
    batch_size = act_predictions.shape[0]

    # initialize performance metrics
    total_dl_distance = 0.0
    
    pred_len = lens_till_eoc(act_predictions, eoc_index)  # (batch_size,)
    tgt_len  = lens_till_eoc(tgt_act, eoc_index)       # (batch_size,)
    assert (pred_len > 0).all().item(), "Error: predicted suffix lengths contain non-positive values"
    assert (tgt_len > 0).all().item(), "Error: target suffix lengths contain non-positive values"

    for i in range(batch_size):
        p = act_predictions[i, :int(pred_len[i])].tolist()
        t = tgt_act[i, :int(tgt_len[i])].tolist()
        # max_len = max(len(p), len(t))
        total_dl_distance += norm_dl_distance(p, t)

    # compute average metrics over the batch
    dl_distance = total_dl_distance / batch_size
    mae_len = torch.abs(pred_len - tgt_len).float().mean()
    
    return dl_distance, mae_len

def lens_till_eoc(x: torch.Tensor, eoc_index: int) -> torch.Tensor:
    
        B, T = x.shape
        is_eoc = (x == eoc_index) # (batch_size, suffix_len) bool
        has_eoc = is_eoc.any(dim=1) # (batch_size, ) bool

        # argmax gives first True only if there is any True; otherwise 0.
        first_eoc = is_eoc.int().argmax(dim=1) # (batch_size, ) long

        lengths = torch.where(has_eoc, first_eoc + 1, torch.full_like(first_eoc, T))

        return lengths
