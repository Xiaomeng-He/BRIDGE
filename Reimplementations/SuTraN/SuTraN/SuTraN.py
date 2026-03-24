import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from SuTraN.transformer_prefix_encoder import EncoderLayer
from SuTraN.transformer_suffix_decoder import DecoderLayer

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()

        # Check if d_model is an integer and is even
        assert isinstance(d_model, int), "d_model must be an integer"
        assert d_model % 2 == 0, "d_model must be an even number"
        
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        self.register_buffer('pe', pe) # shape (max_len, d_model)

    def forward(self, x):
      
        x = x + self.pe[:x.size(1), :] # (batch_size, window_size, d_model)
        return self.dropout(x)

class SuTraN(nn.Module):
    def __init__(self, 
                 num_activities, 
                 act_embed, 
                 input_dims, 
                 emb_dims,
                 num_nume_feature,
                 d_model = 32,  
                 num_prefix_encoder_layers = 4, 
                 num_decoder_layers = 4,
                 num_heads=8, 
                 d_ff = 128, 
                 dropout = 0.2, 
                 remaining_runtime_head = True, 
                 layernorm_embeds = True, 
                 outcome_bool = False,
                 ):
    
        super(SuTraN, self).__init__()

        self.num_activities = num_activities

        self.d_model = d_model

        self.num_prefix_encoder_layers = num_prefix_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.remaining_runtime_head = remaining_runtime_head
        self.layernorm_embeds = layernorm_embeds
        self.outcome_bool = outcome_bool

        # Initialize positional encoding layer 
        self.positional_encoding = PositionalEncoding(d_model)

        self.act_embedding = nn.Embedding(num_activities, act_embed, 0)
        self.embeddings = nn.ModuleList([
            nn.Embedding(inp, dim, 0)
            for inp, dim in zip(input_dims, emb_dims)
        ])

        self.dim_init_prefix = act_embed + sum(emb_dims) + num_nume_feature

        self.input_embeddings_encoder = nn.Linear(self.dim_init_prefix, self.d_model)

        self.dim_init_suffix = act_embed + 2

        # Initial input embedding prefix events (encoder)
        self.input_embeddings_decoder = nn.Linear(self.dim_init_suffix, self.d_model)

        # Initializing the num_prefix_encoder_layers encoder layers 
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(self.num_prefix_encoder_layers)])
        # Initializing the num_decoder_layers decoder layers 
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(self.num_decoder_layers)])

        # Initializing the additional activity output layer
        self.fc_out_act = nn.Linear(self.d_model, self.num_activities) # (batch_size, window_size, num_activities)

        # Initializing the additional time till next event prediction layer
        self.fc_out_ttne = nn.Linear(self.d_model, 1)

        if self.remaining_runtime_head:
            # Additional remaining runtime layers
            self.fc_out_rrt = nn.Linear(self.d_model, 1)

        if self.outcome_bool:
            # Additional (binary) outcome head 
            self.fc_out_out = nn.Linear(self.d_model, 1)
            # Sigmoid activiation function
            self.sigmoid_out = nn.Sigmoid()
        
        
        if self.layernorm_embeds:
            self.norm_enc_embeds = nn.LayerNorm(self.d_model)
            self.norm_dec_embeds = nn.LayerNorm(self.d_model)

            
        self.dropout = nn.Dropout(self.dropout)

        # Creating forward call bools to know what to output 
        self.only_rrt = (not self.outcome_bool) & self.remaining_runtime_head
        self.only_out = self.outcome_bool & (not self.remaining_runtime_head)
        self.both_not = (not self.outcome_bool) & (not self.remaining_runtime_head)
        self.both = self.outcome_bool & self.remaining_runtime_head
    


    # window_size : number of decoding steps during inference (model.eval())
    def forward(self,
                inputs,  
                window_size=None, 
                mean_std_ttne=None, 
                mean_std_tsp=None, 
                mean_std_tss=None):

        cate_prefix, nume_prefix, dec_input_act, dec_input_time = inputs
        cate_prefix = cate_prefix.long()
        dec_input_act = dec_input_act.long()

        num_ftrs_suf = dec_input_time # (batch_size, window_size, 2)
        
        padding_mask_input = (cate_prefix[:, :, 0] == 0) # shape: (batch_size, prefix_len)
        act_embed = self.act_embedding(cate_prefix[:, :, 0])

        if cate_prefix.shape[-1] > 1: # if there are other categorical features rather than activity label
            embedded = [act_embed]
            for i, emb in enumerate(self.embeddings):
                feature_i = cate_prefix[:, :, i+1]   
                emb_i = emb(feature_i)  
                embedded.append(emb_i)
            embedded = torch.cat(embedded, dim=-1)
        else:
            embedded = act_embed

        x = torch.cat([embedded, nume_prefix], dim=-1)

        # Dropout over concatenated features: 
        x = self.dropout(x)

        # Initial embedding encoder (prefix events)
        x = self.positional_encoding(self.input_embeddings_encoder(x) * math.sqrt(self.d_model)) # (batch_size, window_size, d_model)
        if self.layernorm_embeds:
            x = self.norm_enc_embeds(x) # (batch_size, window_size, d_model)

        # Updating the prefix event embeddings with the encoder blocks 
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, padding_mask_input)

        # ---------------------------

        if self.training: # Teacher forcing (for now)

            # Using the activity embedding layer shared with the encoder 
            cat_emb_suf = self.act_embedding(dec_input_act)

            # Concatenate cat_emb with the numerical features to get initial vector representations suffix event tokens.
            target_in = torch.cat((cat_emb_suf, num_ftrs_suf), dim = -1) # (batch_size, window_size, self.dim_init_suffix)
            
            # Initial embeddings decoder suffix event tokens 
            # The positional encoding module applies dropout over the result 
            target_in = self.positional_encoding(self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)) # (batch_size, window_size, d_model)

            if self.layernorm_embeds:
                target_in = self.norm_dec_embeds(target_in) # (batch_size, window_size, d_model)

            # Activating the decoder
            dec_output = target_in
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, x, padding_mask_input) # (batch_size, window_size)

            # Next activity prediction head: 
            act_probs = self.fc_out_act(dec_output) # (batch_size, window_size, self.num_activities)

            # Time till next event prediction (ttne) head:
            ttne_pred = self.fc_out_ttne(dec_output) # (batch_size, window_size, 1)

            # if self.remaining_runtime_head:
            if self.only_rrt:
                # Complete remaining runtime prediction (rrt) head
                rrt_pred = self.fc_out_rrt(dec_output) # (batch_size, window_size, 1)

                return act_probs, ttne_pred, rrt_pred 
                # (batch_size, window_size, self.num_activities), (batch_size, window_size, 1), (batch_size, window_size, 1)
            elif self.only_out:
                out_pred = self.fc_out_out(dec_output) # (batch_size, window_size, 1)
                out_pred = self.sigmoid_out(out_pred) # (batch_size, window_size, 1)
                # Only first decoding step output needed 
                out_pred = out_pred[:, 0, :] # (batch_size, 1)
                return act_probs, ttne_pred, out_pred
            elif self.both:
                rrt_pred = self.fc_out_rrt(dec_output) # (batch_size, window_size, 1)

                out_pred = self.fc_out_out(dec_output) # (batch_size, window_size, 1)
                out_pred = self.sigmoid_out(out_pred) # (batch_size, window_size, 1)
                # Only first decoding step output needed 
                out_pred = out_pred[:, 0, :] # (batch_size, 1)
                return act_probs, ttne_pred, rrt_pred, out_pred
            else: 
                return act_probs, ttne_pred
                # (batch_size, window_size, self.num_activities), (batch_size, window_size, 1)

        else: # Inference mode greedy decoding activities 
            # NOTE: Considerations for future work, in which you adopt 
            # similar procedure during training for rescheduled sampling: 
            # Pay attention to gradient tracking, whether you should detach 
            # the next decoder suffix event token's derived features based 
            # on predictions current decoding step. Figure out whether 
            # the operations involved still maintain differentiability wrt 
            # predictions used for deriving new features. 

            # Retrieving suffix activity integer vector `act_inputs`.
            #   `act_inputs` still contains the ground truth activity 
            #   labels (shifted by 1) for the entire suffixes. However, 
            #   at each decoding step `dec_step`, we will only predict 
            #   based on the shifed suffix generated up till that point, 
            #   and use those predictions to update the activity labels  
            #   for the subsequent decoding step. Finally, the look-ahead  
            #   mask ensures that the decoder cannot incorporate any 
            #   information regarding ground-truth activity labels in the 
            #   suffix.
            #   NOTE: the same holds for the two time features of the 
            #   suffix event tokens (`num_ftrs_suf`).

            # act_inputs = inputs[idx] # (B, W)
            act_inputs = dec_input_act # (B, W)

            batch_size = act_inputs.size(0) # B

            # Initializing zero filled tensors for storing the activity 
            # and timestamp predictions during decoding 
            suffix_acts_decoded = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.int64).to(device) # (B, W)
            suffix_ttne_preds = torch.full(size=(batch_size, window_size), fill_value=0, dtype=torch.float32).to(device) # (B, W)

            for dec_step in range(0, window_size):
                # Leveraging learned embedding 
                cat_emb_suf = self.act_embedding(act_inputs) # (B, W, self.activity_emb_size)

                # Concatenating both
                target_in = torch.cat((cat_emb_suf, num_ftrs_suf), dim = -1) # (B, W, dim_init_suffix)

                # Initial embeddings decoder suffix event tokens 
                target_in = self.positional_encoding(self.input_embeddings_decoder(target_in) * math.sqrt(self.d_model)) # (B, W, d_model)

                # Applying layernorm if specified 
                if self.layernorm_embeds:
                    target_in = self.norm_dec_embeds(target_in) # (B, W, d_model)

                # Activating the decoder
                dec_output = target_in
                for dec_layer in self.decoder_layers:
                    dec_output = dec_layer(dec_output, x, padding_mask_input) # (batch_size, window_size)

                # Next activity prediction head: 
                act_logits = self.fc_out_act(dec_output) # (B, W, self.num_activities)

                # Time till next event prediction (ttne) head:
                ttne_pred = self.fc_out_ttne(dec_output) # (B, W, 1)

                #   Selecting predictions for current decoding step
                act_outputs = act_logits[:, dec_step, :] # (B, C)
                ttne_outputs = ttne_pred[:, dec_step, 0] # (B, )

                # Adding time pred as-is 
                suffix_ttne_preds[:, dec_step] = ttne_outputs # (B, W)


                # Remaining Runtime Predictions and optional outcome 
                # prediction only performed at the very first decoding 
                # step 
                if dec_step == 0:
                    if self.remaining_runtime_head:
                        rrt_pred = self.fc_out_rrt(dec_output) # (B, W, 1)
                        # Slicing out first decoding step prediction only
                        rrt_pred = rrt_pred[:, 0, 0] # (B,)

                    if self.outcome_bool:
                        out_pred = self.fc_out_out(dec_output) # (B, W, 1)
                        out_pred = self.sigmoid_out(out_pred) # (B, W, 1)
                        # Slicing out first decoding step prediction only
                        out_pred = out_pred[:, 0, 0] # (batch_size, )

                # Decoding activity preditions (greedily)
                #   "Masking padding token"
                act_outputs[:, 0] = -1e9

                #   Greedy selection 
                act_selected = torch.argmax(act_outputs, dim=-1) # (batch_size,), torch.int64

                #   Adding selected activity integers to suffix_acts_decoded
                suffix_acts_decoded[:, dec_step] = act_selected

                if dec_step < (window_size-1):

                    # Deriving activity indices pertaining to the 
                    # selected activities for the derived next suffix 
                    # event to be fed to the decoder in the next decoding 
                    # step. 
                    act_suf_updates = act_selected.clone() # (batch_size, )

                    #   There is no artificially added END token present in the 
                    #   suffix activity representations, and hence there is no 
                    #   end token index in the suffix activity representations 
                    #   on index num_activities-1. Therefore, we clamp 
                    #   it on num_activities-2. Predictions for already finished 
                    #   instances will not be taken into account at the end. 
                    act_suf_updates = torch.clamp(act_suf_updates, max=self.num_activities-2) # (batch_size,) aka (B,)

                    # Updating `act_inputs` for suffix decoder for next decoding step 

                    act_inputs[:, dec_step+1] = act_suf_updates # (B, W)

                    # Deriving TSS and TSP time features for next decoding 
                    # step based on the TTNE predictions 

                    #   Converting predictions standardized TTNE 
                    #   back to original scale (seconds)
                    time_preds_seconds = ttne_outputs*mean_std_ttne[1] + mean_std_ttne[0] # (batch_size,)

                    #   Truncating at zero (no negatives allowed)
                    time_preds_seconds = torch.clamp(time_preds_seconds, min=0)

                    #   Converting standardized TSS feature current decoding 
                    #   step's suffix event token to original scale (seconds) 
                    tss_stand = num_ftrs_suf[:, dec_step, 0].clone() # (batch_size,)
                    tss_seconds = tss_stand*mean_std_tss[1] + mean_std_tss[0] # (batch_size,)

                    #   Clamping at zero again 
                    tss_seconds = torch.clamp(tss_seconds, min=0)

                    #   Updating tss in seconds next decoding step based on 
                    #   converted TTNE predictions 
                    tss_seconds_new = tss_seconds + time_preds_seconds # (batch_size,)

                    #   Converting back to preprocessed scale based on 
                    #   training mean and std
                    tss_stand_new = (tss_seconds_new - mean_std_tss[0]) / mean_std_tss[1] # (batch_size,)

                    #   TSP: time since previous event next decoding step 
                    #   is equal to the ttne in seconds, standardized with 
                    #   the training mean and std of the Suffix TSP feature 
                    tsp_stand_new = (time_preds_seconds - mean_std_tsp[0]) / mean_std_tsp[1] # (batch_size,)


                    #   Concatenating both 
                    new_suffix_timefeats = torch.cat((tss_stand_new.unsqueeze(-1), tsp_stand_new.unsqueeze(-1)), dim=-1) # (B, 2)
                    #   Updating next decoding step's time feature
                    #   tensor for the suffix event tokens 
                    num_ftrs_suf[:, dec_step+1, :] = new_suffix_timefeats # (B, W, 2)
            
            if self.only_rrt:
                return suffix_acts_decoded, suffix_ttne_preds, rrt_pred
                # (B, W), (B, W) and (B,)
            elif self.only_out:
                return suffix_acts_decoded, suffix_ttne_preds, out_pred
                # (B, W), (B, W) and (B, )
            elif self.both:
                return suffix_acts_decoded, suffix_ttne_preds, rrt_pred, out_pred
                # (B, W), (B, W), (B,) and (B, )
            else:
                return suffix_acts_decoded, suffix_ttne_preds
                # (B, W), (B, W)
    