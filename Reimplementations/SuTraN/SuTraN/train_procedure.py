import torch
import torch.nn as nn

from SuTraN.train_utils import MultiOutputLoss
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.data import DataLoader
from SuTraN.inference_procedure import inference_loop

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, 
                training_loader, 
                remaining_runtime_head, 
                outcome_bool,
                optimizer,
                loss_fn, 
                batch_interval,
                epoch_number, 
                max_norm):

    # Tracking global loss over all prediction heads:
    running_loss_glb = []
    # Tracking loss of each prediction head separately: 
    running_loss_act = [] # Cross-Entropy
    running_loss_ttne = [] # MAE

    # Creating auxiliary bools 
    only_rrt = (not outcome_bool) & remaining_runtime_head
    only_out = outcome_bool & (not remaining_runtime_head)
    both_not = (not outcome_bool) & (not remaining_runtime_head)
    both = outcome_bool & remaining_runtime_head
    # If the only additional prediction head (on top of activity and ttne 
    # suffix) is remaining time prediction
    if only_rrt:
        running_loss_rrt = [] # MAE
        num_target_tens = 3
    # If the only additional prediction head (on top of activity and ttne 
    # suffix) is (binary) outcome prediction
    elif only_out:
        running_loss_out = []
        num_target_tens = 3
    # If the additional prediction heads (on top of activity and ttne 
    # suffix) comprise both rrt and outcome prediction
    elif both:
        running_loss_rrt = []
        running_loss_out = []
        num_target_tens = 4
    # If their are no additional prediction heads (on top of activity and 
    # ttne suffix)
    elif both_not: 
        num_target_tens = 2

    original_norm_glb = []
    clipped_norm_glb = []



    for batch_num, data in enumerate(training_loader):

        inputs = data[:-num_target_tens]
        labels = data[-num_target_tens:]
        # Sending inputs and labels to GPU
        inputs = [input_tensor.to(device) for input_tensor in inputs]
        labels = [label_tensor.to(device) for label_tensor in labels]
        labels[-1] = labels[-1].long()

        # Restoring gradients to 0 for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss 
        loss_results = loss_fn(outputs, labels)
        loss = loss_results[0]

        # Compute gradients 
        loss.backward()

        # Keep track of original gradient norm 
        original_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        original_norm_glb.append(original_norm.item())

        # Clip gradient norm
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        clipped_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        clipped_norm_glb.append(clipped_norm.item())

        # Adjust learning weights
        optimizer.step()

        # Tracking losses and metrics
        running_loss_glb.append(loss.item())

        running_loss_act.append(loss_results[1])
        running_loss_ttne.append(loss_results[2])

        if only_rrt:
            running_loss_rrt.append(loss_results[-1])

        elif only_out: 
            running_loss_out.append(loss_results[-1])
        
        elif both:
            running_loss_rrt.append(loss_results[-2])
            running_loss_out.append(loss_results[-1])

    last_running_avg_glob = sum(running_loss_glb[-batch_interval:])/batch_interval
    last_running_avg_act = sum(running_loss_act[-batch_interval:])/batch_interval
    last_running_avg_ttne = sum(running_loss_ttne[-batch_interval:])/batch_interval

    if only_rrt:
        last_running_avg_rrt = sum(running_loss_rrt[-batch_interval:])/batch_interval
        last_running_avgs = last_running_avg_glob, last_running_avg_act, last_running_avg_ttne, last_running_avg_rrt, loss
        return model, optimizer, last_running_avgs
    elif only_out:
        last_running_avg_out = sum(running_loss_out[-batch_interval:])/batch_interval
        last_running_avgs = last_running_avg_glob, last_running_avg_act, last_running_avg_ttne, last_running_avg_out, loss
        return model, optimizer, last_running_avgs
    elif both:
        last_running_avg_rrt = sum(running_loss_rrt[-batch_interval:])/batch_interval
        last_running_avg_out = sum(running_loss_out[-batch_interval:])/batch_interval
        last_running_avgs = last_running_avg_glob, last_running_avg_act, last_running_avg_ttne, last_running_avg_rrt, last_running_avg_out, loss
        return model, optimizer, last_running_avgs
    elif both_not:
        last_running_avgs = last_running_avg_glob, last_running_avg_act, last_running_avg_ttne, loss
        return model, optimizer, last_running_avgs # model remains the same
            

def train_model(model, 
                optimizer, 
                train_dataloader, 
                val_dataset, 
                start_epoch, 
                num_epochs, 
                remaining_runtime_head, 
                outcome_bool, 
                num_classes, 
                batch_interval, 
                path_name, 
                mean_std_ttne, 
                mean_std_tsp, 
                mean_std_tss, 
                mean_std_rrt, 
                patience = 24, 
                lr_scheduler_present=False, 
                lr_scheduler=None, 
                best_MAE_ttne = 1e9, 
                best_DL_sim = -1, 
                best_MAE_rrt = 1e9, 
                best_BCE = 1e9, 
                best_auc_pr = -1,
                max_norm = 2.):
    
    if lr_scheduler_present:
        if lr_scheduler==None:
            print("No lr_scheduler provided.")
            return -1, -1, -1, -1

    # Checking whether GPU is being used
    print("Device: {}".format(device))

    # Assigning CRTP Transformer to GPU. 
    model.to(device)

    # Creating train and validation dataloaders 
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    # Tracking running averages over last ``batch_interval`` batches of each epoch
    # & tracking average validation losses
    train_losses_global = []
    train_losses_act = []
    train_losses_ttne = []

    # Track evolution of validation metrics over the epoch loop by initializing empty lists. 
    avg_MAE_ttne_stand_glob, avg_MAE_ttne_minutes_glob = ([] for _ in range(2))

    avg_dam_lev_glob, perc_too_early_glob, perc_too_late_glob, perc_correct_glob = ([] for _ in range(4))
    mean_absolute_length_diff_glob, mean_too_early_glob, mean_too_late_glob = ([] for _ in range(3))

    if remaining_runtime_head:
        avg_MAE_stand_RRT_glob, avg_MAE_minutes_RRT_glob = [], []
    if outcome_bool: 
        avg_BCE_out_glob, avg_auc_roc_glob, avg_auc_pr_glob = [], [], []

    # Creating auxiliary bools 
    only_rrt = (not outcome_bool) & remaining_runtime_head
    only_out = outcome_bool & (not remaining_runtime_head)
    both_not = (not outcome_bool) & (not remaining_runtime_head)
    both = outcome_bool & remaining_runtime_head

    # Initialize lists for keeping track of training losses of optional 
    # prediction heads if included. 
    if remaining_runtime_head:
        train_losses_rrt = []
    if outcome_bool:
        train_losses_out = []

    
    loss_fn = MultiOutputLoss(num_classes, remaining_runtime_head, outcome_bool)
    num_epochs_not_improved = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Setting seed for shuffling of training dataloader's shuffling 
        # of the instances, such that each epoch is shuffled differently, 
        # while still maintaining reproducability. 
        torch.manual_seed(epoch) #+100 without scheduler
        # print(" ")
        # print("------------------------------------")
        print('EPOCH {}:'.format(epoch))
        # print("____________________________________")

        # Activate gradient tracking
        model.train(True)
        model, optimizer, last_running_avgs = train_epoch(model, 
                                                          train_dataloader, 
                                                          remaining_runtime_head,
                                                          outcome_bool,  
                                                          optimizer, 
                                                          loss_fn, 
                                                          batch_interval, 
                                                          epoch, 
                                                          max_norm)
        train_losses_global.append(last_running_avgs[0])
        train_losses_act.append(last_running_avgs[1])
        train_losses_ttne.append(last_running_avgs[2])
        last_loss = last_running_avgs[-1]
        if only_rrt:
            train_losses_rrt.append(last_running_avgs[3])
        elif only_out:
            train_losses_out.append(last_running_avgs[3])
        elif both:
            train_losses_rrt.append(last_running_avgs[3])
            train_losses_out.append(last_running_avgs[4])

        # Set the model to evaluation mode and disabling dropout
        model.eval()

        inf_results = inference_loop(model=model, 
                                     inference_dataset=val_dataset,
                                     remaining_runtime_head=remaining_runtime_head, 
                                     outcome_bool=outcome_bool, 
                                     # num_categoricals_pref=num_categoricals_pref, 
                                     mean_std_ttne=mean_std_ttne, 
                                     mean_std_tsp=mean_std_tsp, 
                                     mean_std_tss=mean_std_tss, 
                                     mean_std_rrt=mean_std_rrt, 
                                     results_path=None, 
                                     val_batch_size=4096)
        # TTNE MAE metrics
        avg_MAE_ttne_stand, avg_MAE_ttne_minutes = inf_results[:2]
        # Average Normalized Damerau-Levenshtein similarity Activity Suffix 
        # prediction
        avg_dam_lev = inf_results[2]


        # Percentage of validation instances for which the END token was 
        # predicted too early. 
        perc_too_early = inf_results[3]
        # Percentage of validation instances for which the END token was 
        # predicted too late. 
        perc_too_late = inf_results[4]
        # Percentage of validation instances for which the END token was 
        # predicted at the right moment. 
        perc_correct = inf_results[5]
        # Mean absolute lenght difference between predicted and actual 
        # suffix. 
        mean_absolute_length_diff = inf_results[6]
        # Avg num events that END token was predicted too early, averaged 
        # over all instances for which END was predicted too early. 
        mean_too_early = inf_results[7]
        # Avg num events that END token was predicted too late, averaged 
        # over all instances for which END was predicted too late. 
        mean_too_late = inf_results[8]

        
        if only_rrt:
            # MAE standardized RRT predictions 
            avg_MAE_stand_RRT = inf_results[9]
            # MAE RRT converted to minutes
            avg_MAE_minutes_RRT = inf_results[10]
        
        elif only_out:
            # Binary Cross Entropy outcome prediction
            avg_BCE_out = inf_results[9]
            # AUC-ROC outcome prediction
            auc_roc = inf_results[10]
            # AUC-PR outcome prediction
            auc_pr = inf_results[11]
        
        elif both:
            # MAE standardized RRT predictions 
            avg_MAE_stand_RRT = inf_results[9]
            # MAE RRT converted to minutes
            avg_MAE_minutes_RRT = inf_results[10]
            # Binary Cross Entropy outcome prediction
            avg_BCE_out = inf_results[11]
            # AUC-ROC outcome prediction
            auc_roc = inf_results[12]
            # AUC-PR outcome prediction
            auc_pr = inf_results[13]
        

        better = False
        if avg_MAE_ttne_stand < best_MAE_ttne: 
            better = True
            best_MAE_ttne = avg_MAE_ttne_stand
        if avg_dam_lev > best_DL_sim:
            better = True 
            best_DL_sim = avg_dam_lev

        print("Avg MAE TTNE prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_ttne_stand, avg_MAE_ttne_minutes))
        print("Avg 1-(normalized) DL distance acitivty suffix prediction validation set: {}".format(avg_dam_lev))
        if remaining_runtime_head: 
            if avg_MAE_stand_RRT < best_MAE_rrt: 
                better = True
                best_MAE_rrt = avg_MAE_stand_RRT
            print("Avg MAE RRT prediction validation set: {} (standardized) ; {} (minutes)'".format(avg_MAE_stand_RRT, avg_MAE_minutes_RRT))
            # Store evolution validation measures RRT: 
            avg_MAE_stand_RRT_glob.append(avg_MAE_stand_RRT)
            avg_MAE_minutes_RRT_glob.append(avg_MAE_minutes_RRT)
        if outcome_bool:
            if avg_BCE_out < best_BCE:
                better = True
                best_BCE = avg_BCE_out
            if auc_pr > best_auc_pr:
                better = True
                best_auc_pr = auc_pr
            print("Avg BCE outcome prediction validation set: {}".format(avg_BCE_out))
            print("AUC-ROC outcome prediction validation set: {}".format(auc_roc))
            print("AUC-PR outcome prediction validation set: {}".format(auc_pr))
            avg_BCE_out_glob.append(avg_BCE_out)
            avg_auc_roc_glob.append(auc_roc)
            avg_auc_pr_glob.append(auc_pr)
        if better == False: 
            num_epochs_not_improved += 1
        else:
            num_epochs_not_improved = 0
        # Store other validation measures
        #   TTNE measures
        avg_MAE_ttne_stand_glob.append(avg_MAE_ttne_stand)
        avg_MAE_ttne_minutes_glob.append(avg_MAE_ttne_minutes)
        avg_dam_lev_glob.append(avg_dam_lev)
        perc_too_early_glob.append(perc_too_early)
        perc_too_late_glob.append(perc_too_late)
        perc_correct_glob.append(perc_correct)
        mean_absolute_length_diff_glob.append(mean_absolute_length_diff)
        mean_too_early_glob.append(mean_too_early)
        mean_too_late_glob.append(mean_too_late)
        # Saving checkpoint every epoch

        model_path = os.path.join(path_name, 'model_epoch_{}.pt'.format(epoch))
        checkpoint = {'epoch:' : epoch, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(), 
                        'loss': last_loss}
        torch.save(checkpoint, model_path)
            
        if lr_scheduler_present:
            # Update the learning rate
            lr_scheduler.step()
            
        torch.cuda.empty_cache()


        if num_epochs_not_improved >= patience:
            print("No improvements in validation loss for {} consecutive epochs. Final epoch: {}".format(patience, epoch))
            break
    # Writing training progress to csv at the end of the current training loop
    results_path = os.path.join(path_name, 'backup_results.csv')
    epoch_list = [i for i in range(len(train_losses_global))]
    if both:
        results = pd.DataFrame(data = {'epoch' : epoch_list, 
                            'composite training loss' : train_losses_global, 
                            'activity training loss (cross entropy)': train_losses_act, 
                            'time till next event training loss (MAE)': train_losses_ttne, 
                            '(complete) remaining runtime training loss (MAE)': train_losses_rrt, 
                            'outcome prediction training loss (BCE)': train_losses_out,
                            'TTNE - standardized MAE validation': avg_MAE_ttne_stand_glob, 
                            'TTNE - minutes MAE validation': avg_MAE_ttne_minutes_glob, 
                            'Activity suffix: 1-DL (validation)': avg_dam_lev_glob,  
                            'Percentage too early (validation)': perc_too_early_glob,    
                            'Percentage correct END prediction (validation)': perc_correct_glob,   
                            'Percentage too late (validation)': perc_too_late_glob,   
                            'Avg absolute amount of events predicted too early / too late (validation)': mean_absolute_length_diff_glob, 
                            'Avg too early (validation)': mean_too_early_glob, 
                            'Avg too late (validation)': mean_too_late_glob, 
                            'RRT - standardized MAE validation': avg_MAE_stand_RRT_glob, 
                            'RRT - mintues MAE validation': avg_MAE_minutes_RRT_glob, 
                            'Binary Outcome - BCE validation': avg_BCE_out_glob, 
                            'Binary Outcome - AUC-ROC validation': avg_auc_roc_glob, 
                            'Binary Outcome - AUC-PR validation': avg_auc_pr_glob})
        results.to_csv(results_path, index=False)
    elif only_rrt:
        results = pd.DataFrame(data = {'epoch' : epoch_list, 
                            'composite training loss' : train_losses_global, 
                            'activity training loss (cross entropy)': train_losses_act, 
                            'time till next event training loss (MAE)': train_losses_ttne, 
                            '(complete) remaining runtime training loss (MAE)': train_losses_rrt, 
                            'TTNE - standardized MAE validation': avg_MAE_ttne_stand_glob, 
                            'TTNE - minutes MAE validation': avg_MAE_ttne_minutes_glob, 
                            'Activity suffix: 1-DL (validation)': avg_dam_lev_glob,  
                            'Percentage too early (validation)': perc_too_early_glob,    
                            'Percentage correct END prediction (validation)': perc_correct_glob,   
                            'Percentage too late (validation)': perc_too_late_glob,   
                            'Avg absolute amount of events predicted too early / too late (validation)': mean_absolute_length_diff_glob, 
                            'Avg too early (validation)': mean_too_early_glob, 
                            'Avg too late (validation)': mean_too_late_glob, 
                            'RRT - standardized MAE validation': avg_MAE_stand_RRT_glob, 
                            'RRT - mintues MAE validation': avg_MAE_minutes_RRT_glob})
        results.to_csv(results_path, index=False)
    
    elif only_out:
        results = pd.DataFrame(data = {'epoch' : epoch_list, 
                            'composite training loss' : train_losses_global, 
                            'activity training loss (cross entropy)': train_losses_act, 
                            'time till next event training loss (MAE)': train_losses_ttne,
                            'outcome prediction training loss (BCE)': train_losses_out,
                            'TTNE - standardized MAE validation': avg_MAE_ttne_stand_glob, 
                            'TTNE - minutes MAE validation': avg_MAE_ttne_minutes_glob, 
                            'Activity suffix: 1-DL (validation)': avg_dam_lev_glob,  
                            'Percentage too early (validation)': perc_too_early_glob,    
                            'Percentage correct END prediction (validation)': perc_correct_glob,   
                            'Percentage too late (validation)': perc_too_late_glob,   
                            'Avg absolute amount of events predicted too early / too late (validation)': mean_absolute_length_diff_glob, 
                            'Avg too early (validation)': mean_too_early_glob, 
                            'Avg too late (validation)': mean_too_late_glob,
                            'Binary Outcome - BCE validation': avg_BCE_out_glob, 
                            'Binary Outcome - AUC-ROC validation': avg_auc_roc_glob, 
                            'Binary Outcome - AUC-PR validation': avg_auc_pr_glob})
        results.to_csv(results_path, index=False)
    else:
        results = pd.DataFrame(data = {'epoch' : epoch_list, 
                            'composite training loss' : train_losses_global, 
                            'activity training loss (cross entropy)': train_losses_act, 
                            'time till next event training loss (MAE)': train_losses_ttne,
                            'TTNE - standardized MAE validation': avg_MAE_ttne_stand_glob, 
                            'TTNE - minutes MAE validation': avg_MAE_ttne_minutes_glob, 
                            'Activity suffix: 1-DL (validation)': avg_dam_lev_glob,  
                            'Percentage too early (validation)': perc_too_early_glob,    
                            'Percentage correct END prediction (validation)': perc_correct_glob,   
                            'Percentage too late (validation)': perc_too_late_glob,   
                            'Avg absolute amount of events predicted too early / too late (validation)': mean_absolute_length_diff_glob, 
                            'Avg too early (validation)': mean_too_early_glob, 
                            'Avg too late (validation)': mean_too_late_glob})
        results.to_csv(results_path, index=False)