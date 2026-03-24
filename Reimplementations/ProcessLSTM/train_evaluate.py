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
    epoch_loss = 0.0

    for batch in dataloader:

        # load data
        act_prefix, time_prefix, next_act, next_time = batch
        # next_act shape: (batch_size,)
        # next_time shape: (batch_size,)
        
        act_prefix = act_prefix.long().to(device)
        next_act = next_act.long().to(device)

        time_prefix = time_prefix.float().to(device)
        next_time = next_time.float().to(device)
        
        # set the gradient to zero
        optimizer.zero_grad()

        # run a forward pass and obtain predictions
        act_logits, time_pred = model(act_prefix, time_prefix)
        # act_logits shape: (batch_size, num_activities)
        # time_pred shape: (batch_size, 1)

        # calculate loss
        loss = loss_function(act_logits, time_pred,
                             next_act, next_time)

        # backpropagation
        loss.backward()
        optimizer.step()

        # sum up losses from all batches
        epoch_loss += loss.item()
    
    # compute average losses over the batch
    avg_loss = epoch_loss / len(dataloader)
    
    return avg_loss


def validate(model,
            dataloader,
            device):

    model.eval()

    # initialize losses
    epoch_loss = 0.0

    with torch.no_grad():

        for batch in dataloader:

            # load data
            act_prefix, time_prefix, next_act, next_time = batch
            # next_act shape: (batch_size,)
            # next_time shape: (batch_size,)
            
            act_prefix = act_prefix.long().to(device)
            next_act = next_act.long().to(device)

            time_prefix = time_prefix.float().to(device)
            next_time = next_time.float().to(device)

            # run a forward pass and obtain predictions
            act_logits, time_pred = model(act_prefix, time_prefix)
            # act_logits shape: (batch_size, num_activities)
            # time_pred shape: (batch_size, 1)
            
            # calculate loss
            loss = loss_function(act_logits, time_pred,
                                next_act, next_time)

            # sum up losses from all batches
            epoch_loss += loss.item()

    # compute average losses over the batch
    avg_loss = epoch_loss / len(dataloader)
    
    return avg_loss

def evaluate(model,
            dataloader,
            mean_time,
            std_time,
            device):

    model.eval()

    # initialize performance metrics
    epoch_dl_distance, epoch_mea_len = 0.0, 0.0

    with torch.no_grad():

        for batch in dataloader:

            # load data
            act_prefix, time_prefix, tgt_act = batch
            # prefix shape: (batch_size, prefix_len)
            
            act_prefix = act_prefix.long().to(device)
            time_prefix = time_prefix.float().to(device)
            tgt_act = tgt_act.long().to(device)

            suffix_len = tgt_act.shape[-1]

            cur_act_pref = act_prefix.clone()
            cur_time_pref = time_prefix.clone()

            pred_acts = []

            for i in range(suffix_len):
                act_logits, time_pred = model(cur_act_pref, cur_time_pref)
                # act_logits shape: (batch_size, num_activities)
                # time_pred shape: (batch_size, 1)

                # avoid sampling from PAD (0) and SOC (2)
                act_logits[:, [0, 2]] = float('-inf')

                next_act = act_logits.argmax(-1) # (batch_size)

                pred_acts.append(next_act)

                cur_act_pref = torch.cat(
                    [cur_act_pref[:, 1:], next_act.unsqueeze(1)],
                    dim=1)
                
                # -- time --
                next_dt_std = time_pred.squeeze(-1)    # (batch_size,)

                # De-standardize to seconds
                next_dt_sec = next_dt_std * std_time + mean_time
                next_dt_sec = torch.clamp(next_dt_sec, min=0.0)

                # Last event's calendar features
                last_time_since_midnight_norm = cur_time_pref[:, -1, 1]   
                last_weekday_norm = cur_time_pref[:, -1, 2]       

                last_time_since_midnight_sec = last_time_since_midnight_norm * 86400.0
                last_weekday = torch.round(last_weekday_norm * 7.0).long()

                # Add predicted delta
                total_sec = last_time_since_midnight_sec + next_dt_sec   

                # Derive next daytime and weekday
                days_passed = torch.floor_divide(total_sec.long(), 86400) 
                next_time_since_midnight_sec = torch.remainder(total_sec, 86400.0)
                next_weekday = (last_weekday + days_passed) % 7

                # Normalize again
                next_time_since_midnight_norm = next_time_since_midnight_sec / 86400.0
                next_weekday_norm = next_weekday.float() / 7.0

                next_time_features = torch.stack(
                    [
                        next_dt_std,                    # standardized time since previous event
                        next_time_since_midnight_norm, # normalized daytime
                        next_weekday_norm              # normalized weekday
                    ],
                    dim=-1
                ).unsqueeze(1)                  

                cur_time_pref = torch.cat(
                    [cur_time_pref[:, 1:, :], next_time_features],
                    dim=1
                )
                
            act_prediction = torch.stack(pred_acts, dim=1) # (batch_size, suffix_len)

            # calculate performance metrics
            dl_distance, mae_len = performance_metrics(act_prediction, tgt_act)
            
            # sum up metrics from all batches
            epoch_dl_distance += dl_distance
            epoch_mea_len += mae_len.item()

    # compute average metrics over the batch
    avg_dl_distance = epoch_dl_distance / len(dataloader)
    avg_mae_len = epoch_mea_len / len(dataloader)
    
    return avg_dl_distance, avg_mae_len

def loss_function(act_logits, time_pred,
                  next_act, next_time):

    act_criterion = nn.CrossEntropyLoss(ignore_index=0)
    time_criterion = nn.L1Loss()

    act_loss = act_criterion(act_logits, next_act)
    next_time = next_time.unsqueeze(-1) 
    time_loss = time_criterion(time_pred, next_time)

    loss = (act_loss  + time_loss) / 2

    return loss

def lens_till_eoc(x: torch.Tensor, eoc_index: int) -> torch.Tensor:
    
        B, T = x.shape
        is_eoc = (x == eoc_index) # (batch_size, suffix_len) bool
        has_eoc = is_eoc.any(dim=1) # (batch_size, ) bool

        # argmax gives first True only if there is any True; otherwise 0.
        first_eoc = is_eoc.int().argmax(dim=1) # (batch_size, ) long

        lengths = torch.where(has_eoc, first_eoc + 1, torch.full_like(first_eoc, T))

        return lengths

def performance_metrics(act_prediction, tgt_act,
                        eoc_index = int(3)):
    
    tgt_act = tgt_act.detach().cpu()
    act_prediction = act_prediction.detach().cpu()
    batch_size = tgt_act.shape[0]

    total_dl_distance = 0.0
    
    pred_len = lens_till_eoc(act_prediction, eoc_index)  # (batch_size,)
    tgt_len  = lens_till_eoc(tgt_act, eoc_index)       # (batch_size,)

    for i in range(batch_size):
        p = act_prediction[i, :int(pred_len[i])].tolist()
        t = tgt_act[i, :int(tgt_len[i])].tolist()
        total_dl_distance += norm_dl_distance(p, t)

    # compute average metrics over the batch
    dl_distance = total_dl_distance / batch_size
    mae_len = torch.abs(pred_len - tgt_len).float().mean()
    
    return dl_distance, mae_len

def training_trial(model,
                   num_epochs,
                   train_dataloader,
                   device,
                   val_dataloader,
                   model_state_path,
                   best_val_loss = float("inf")):

    results = []

    optimizer = optim.NAdam(model.parameters(),lr=0.002,betas=(0.9, 0.999),eps=1e-8)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',      
        factor=0.5,
        patience=10,
        threshold=1e-4,   
        threshold_mode='rel', 
        cooldown=0,
        min_lr=0,
        verbose=False)

    early_stopper = EarlyStopper(patience=42)

    for epoch in tqdm.tqdm(range(num_epochs)):

        train_loss = train(model, 
                           train_dataloader,
                           optimizer,
                           device)
        
        val_loss = validate(model,
                            val_dataloader,
                            device)
        print(f"\tTrain Loss: {train_loss:7.3f} | Val Loss: {val_loss:7.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_state_path)

        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
            })
        
        # learning rate scheduler
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"LR reduced: {old_lr:.6g} → {new_lr:.6g} (epoch {epoch+1})")
        
        # early stopping
        if early_stopper.early_stop(val_loss):     
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
