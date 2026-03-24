import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance as norm_dl_distance

def train(model, 
          dataloader,
          optimizer,
          device):
    
    model.train() 

    # initialize losses
    epoch_loss, epoch_act_loss, epoch_rrt_loss = 0.0, 0.0, 0.0

    for batch in dataloader:

        # load data
        cate_prefix, nume_prefix,\
            suffix_act, suffix_rrt = batch
        
        cate_prefix = cate_prefix.long().to(device)
        suffix_act = suffix_act.long().to(device)
        nume_prefix = nume_prefix.float().to(device)
        suffix_rrt = suffix_rrt.float().to(device)
        
        # set the gradient to zero
        optimizer.zero_grad()

        # run a forward pass and obtain predictions
        act_probs, rrt_pred = model(cate_prefix, nume_prefix)
        # act_probs shape: (batch_size, suffix_len, num_act)
        # rrt_pred shape: (batch_size, suffix_len, 1)

        # ensure predictions are stored in device
        act_probs, rrt_pred = act_probs.to(device), rrt_pred.to(device)
        
        # the first element of suffix_act and suffix_rrt is the last element 
        # in the prefix, so tgt_act should exclude the first position and tgt_rrt
        # should exlude the last position
        tgt_act = suffix_act[:, 1:].clone()
        tgt_rrt = suffix_rrt[:, :-1].clone()
        
        # calculate act_loss, rrt_loss and loss
        loss, act_loss, rrt_loss = loss_function(act_probs,
                                                 rrt_pred,
                                                 tgt_act,
                                                 tgt_rrt,
                                                 device)

        # backpropagation
        loss.backward()
        optimizer.step()

        # sum up losses from all batches
        epoch_loss += loss.item()
        epoch_act_loss += act_loss.item()
        epoch_rrt_loss += rrt_loss.item()
    
    # compute average losses over the batch
    avg_loss = epoch_loss / len(dataloader)
    avg_act_loss = epoch_act_loss / len(dataloader)
    avg_rrt_loss = epoch_rrt_loss / len(dataloader)
    
    return avg_loss, avg_act_loss, avg_rrt_loss

def validate(model,
            dataloader,
            device):

    model.eval()

    # initialize losses and performance metrics
    epoch_loss, epoch_act_loss, epoch_rrt_loss = 0.0, 0.0, 0.0
    epoch_dl_distance, epoch_mea_len = 0.0, 0.0

    with torch.no_grad():

        for batch in dataloader:

            # load data
            cate_prefix, nume_prefix,\
                suffix_act, suffix_rrt = batch
            
            cate_prefix = cate_prefix.long().to(device)
            suffix_act = suffix_act.long().to(device)
            nume_prefix = nume_prefix.float().to(device)
            suffix_rrt = suffix_rrt.float().to(device)

            # run a forward pass and obtain predictions
            act_probs, rrt_pred = model(cate_prefix, nume_prefix)
            # act_probs shape: (batch_size, suffix_len, num_act)
            # rrt_pred shape: (batch_size, suffix_len, 1)

            # ensure predictions are stored in device
            act_probs, rrt_pred = act_probs.to(device), rrt_pred.to(device)
            # convert predicted activity label probabilities to lebel indices
            act_predictions = act_probs.argmax(2) # shape: (batch_size, suffix_len)

            # the first element of suffix_act and suffix_rrt is the last element 
            # in the prefix, so the target should exclude the first position
            tgt_act = suffix_act[:, 1:].clone()
            tgt_rrt = suffix_rrt[:, :-1].clone()

            # calculate act_loss, rrt_loss and loss
            loss, act_loss, rrt_loss = loss_function(act_probs,
                                                 rrt_pred,
                                                 tgt_act,
                                                 tgt_rrt,
                                                 device)
            
            # calculate performance metrics
            dl_distance, mae_len = performance_metrics(act_predictions, tgt_act)
            
            # sum up losses and metrics from all batches
            epoch_loss += loss.item()
            epoch_act_loss += act_loss.item()
            epoch_rrt_loss += rrt_loss.item()

            epoch_dl_distance += dl_distance
            epoch_mea_len += mae_len.item()

    # compute average losses and metrics over the batch
    avg_loss = epoch_loss / len(dataloader)
    avg_act_loss = epoch_act_loss / len(dataloader)
    avg_rrt_loss = epoch_rrt_loss / len(dataloader)

    avg_dl_distance = epoch_dl_distance / len(dataloader)
    avg_mae_len = epoch_mea_len / len(dataloader)
    
    return avg_loss, avg_act_loss, avg_rrt_loss, avg_dl_distance, avg_mae_len

def evaluate(model,
            dataloader,
            device):

    model.eval()

    # initialize performance metrics
    epoch_dl_distance, epoch_mea_len = 0.0, 0.0

    with torch.no_grad():

        for batch in dataloader:

            # load data
            cate_prefix, nume_prefix,\
                suffix_act, suffix_rrt = batch
            
            cate_prefix = cate_prefix.long().to(device)
            suffix_act = suffix_act.long().to(device)
            nume_prefix = nume_prefix.float().to(device)
            suffix_rrt = suffix_rrt.float().to(device)

            # run a forward pass and obtain predictions
            act_probs, rrt_pred = model(cate_prefix, nume_prefix)
            # act_probs shape: (batch_size, suffix_len, num_act)
            # rrt_pred shape: (batch_size, suffix_len, 1)

            # ensure predictions are stored in device
            act_probs, rrt_pred = act_probs.to(device), rrt_pred.to(device)
            # convert predicted activity label probabilities to lebel indices
            act_predictions = act_probs.argmax(2) # shape: (batch_size, suffix_len)

            # the first element of suffix_act and suffix_rrt is the last element 
            # in the prefix, so the target should exclude the first position
            tgt_act = suffix_act[:, 1:].clone()
            
            # calculate performance metrics
            dl_distance, mae_len = performance_metrics(act_predictions, tgt_act)
            
            # sum up metrics from all batches
            epoch_dl_distance += dl_distance
            epoch_mea_len += mae_len.item()

    # compute average metrics over the batch
    avg_dl_distance = epoch_dl_distance / len(dataloader)
    avg_mae_len = epoch_mea_len / len(dataloader)
    
    return avg_dl_distance, avg_mae_len

def loss_function(act_probs, 
                  rrt_pred, 
                  tgt_act,
                  tgt_rrt,
                  device,
                  time_masking = float(-10000)):

    act_criterion = nn.CrossEntropyLoss(ignore_index=0)
    rrt_criterion = nn.L1Loss()

    # calculate act loss
    act_probs = act_probs.view(-1, act_probs.size(-1)) # shape: (batch_size * suffix_len, num_act)
    tgt_act = tgt_act.view(-1) # shape: (batch_size * suffix_len,)
    act_loss = act_criterion(act_probs, tgt_act)

    # calculate time loss
    tgt_rrt = tgt_rrt.unsqueeze(-1) # shape: (batch_size, suffix_len, 1) 
    # mask padded entries (-10000) in the rrt suffix so that they do not 
    # contribute to the gradient
    mask = (tgt_rrt != time_masking).to(device)
    masked_tgt_rrt = torch.masked_select(tgt_rrt, mask) 
    masked_rrt_pred = torch.masked_select(rrt_pred, mask)
    rrt_loss = rrt_criterion(masked_rrt_pred, masked_tgt_rrt)

    # calculate overall loss
    loss = 0.5 * act_loss + 0.5 * rrt_loss

    return loss, act_loss, rrt_loss

def lens_till_eoc(x: torch.Tensor, eoc_index: int) -> torch.Tensor:
    
        B, T = x.shape
        is_eoc = (x == eoc_index) # (batch_size, suffix_len) bool
        has_eoc = is_eoc.any(dim=1) # (batch_size, ) bool

        # argmax gives first True only if there is any True; otherwise 0.
        first_eoc = is_eoc.int().argmax(dim=1) # (batch_size, ) long

        lengths = torch.where(has_eoc, first_eoc + 1, torch.full_like(first_eoc, T))

        return lengths

def performance_metrics(act_predictions, 
                        tgt_act,
                        eoc_index = int(3)):

    batch_size = act_predictions.shape[0]

    total_dl_distance = 0.0
    
    pred_len = lens_till_eoc(act_predictions, eoc_index)  # (batch_size,)
    tgt_len  = lens_till_eoc(tgt_act, eoc_index)       # (batch_size,)
    assert (pred_len > 0).all().item(), "Error: predicted suffix lengths contain non-positive values"
    assert (tgt_len > 0).all().item(), "Error: target suffix lengths contain non-positive values"

    for i in range(batch_size):
        pL = int(pred_len[i].item())
        tL = int(tgt_len[i].item())
        p = act_predictions[i, :pL].tolist()
        t = tgt_act[i, :tL].tolist()
        total_dl_distance += norm_dl_distance(p, t)

    # compute average metrics over the batch
    dl_distance = total_dl_distance / batch_size
    mae_len = torch.abs(pred_len - tgt_len).float().mean()
    
    return dl_distance, mae_len

def training_trial(model,
          train_dataloader,
          device,
          val_dataloader,
          model_state_path, 
          num_epochs = 500, 
          best_val_dl_distance = float("inf")):

    results = []

    optimizer = optim.NAdam(model.parameters(), lr=0.002)

    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode='min', 
                                  factor=0.5, 
                                  patience=16, 
                                  threshold=0.0001, 
                                  cooldown=0, 
                                  min_lr=0)
    
    early_stopper = EarlyStopper(patience=59)

    for epoch in tqdm.tqdm(range(num_epochs)):

        train_loss, train_act_loss, train_rrt_loss = train(model, 
                                                     train_dataloader,
                                                     optimizer,
                                                     device)
        
        val_loss, val_act_loss, val_rrt_loss, \
            val_dl_distance, val_mae_len= validate(model,
                                                   val_dataloader,
                                                   device)
        
        print(f"\tTrain Loss: {train_loss:7.3f} | Train Act Loss: {train_act_loss:7.3f} | Train RRT Loss: {train_rrt_loss:7.3f}")
        print(f"\tVal Loss: {val_loss:7.3f}| Val Act Loss: {val_act_loss:7.3f} | Val RRT Loss: {val_rrt_loss:7.3f}")
        print(f"\tVal DL Distance: {val_dl_distance:7.3f} | Val MAE length: {val_mae_len:7.3f}")

        if val_dl_distance < best_val_dl_distance:
            best_val_dl_distance = val_dl_distance
            torch.save(model.state_dict(), model_state_path)

        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_act_loss': train_act_loss,
            'train_rrt_loss': train_rrt_loss,
            'val_loss': val_loss,
            'val_act_loss': val_act_loss,
            'val_rrt_loss': val_rrt_loss,
            'val_dl_distance': val_dl_distance,
            'val_mae_len': val_mae_len
            })
        
        # learning rate scheduler
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_dl_distance)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"LR reduced: {old_lr:.6g} → {new_lr:.6g} (epoch {epoch+1})")
        
        # Empty redundant cache on GPU
        torch.cuda.empty_cache()

        # early stopping
        if early_stopper.early_stop(val_dl_distance):     
            print(f"Early stopping triggered at epoch {epoch + 1}")        
            break
    
    return results

class EarlyStopper:
    def __init__(self, patience):
        self.patience = patience 
        self.counter = 0 
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss):

        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience
