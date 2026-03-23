import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_seqs as dl_distance_seqs

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):

        super(Encoder, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout)
                
    def forward(self, x):
        
        enc_states, (hidden, cell) = self.lstm(x)
        # enc_states: (batch_size, prefix_len, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)

        return enc_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self,
                 input_size, hidden_size, num_layers, dropout,
                 output_size):
               
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size , 
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, dec_input, hidden, cell):
            
        outputs, (hidden, cell) = self.lstm(dec_input, (hidden, cell)) 
        
        act_logits = self.fc(outputs)

        return act_logits, hidden, cell

class ED_LSTM(nn.Module): 
             
    def __init__(self, 
                 act_embed, input_dims, emb_dims,
                 num_act,
                 enc_input_size, hidden_size, num_layers, dropout,
                 dec_input_size,
                 eoc_index=int(3)):
        super(ED_LSTM, self).__init__()
        self.num_act = num_act
        self.eoc_index = eoc_index
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.act_embedding = nn.Embedding(num_act, act_embed, 0)
        self.embeddings = nn.ModuleList([
            nn.Embedding(inp, dim, 0)
            for inp, dim in zip(input_dims, emb_dims)
        ])

        self.encoder = Encoder(enc_input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(dec_input_size, hidden_size, num_layers, dropout, num_act)

    def forward(self, 
                cate_prefix, 
                nume_prefix,
                dec_input_act):

        # -- encoder --

        x = self.process_enc_input(cate_prefix, nume_prefix) # (batch_size, prefix_len, enc_input_size)

        enc_states, hidden, cell = self.encoder(x)
        # enc_states: (batch_size, prefix_len, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)

        # -- decoder --

        dec_act_embed = self.act_embedding(dec_input_act)
        act_logits, _, _ = self.decoder(dec_act_embed, hidden, cell)

        return act_logits
    
    def process_enc_input(self, 
                          cate_prefix, 
                          nume_prefix): 
        
        act_embed = self.act_embedding(cate_prefix[:, :, 0]) # (batch_size, prefix_len, act_embed)

        if cate_prefix.shape[-1] > 1:
            embedded = [act_embed]
            for i, emb in enumerate(self.embeddings):
                feature_i = cate_prefix[:, :, i+1]   
                emb_i = emb(feature_i)  
                embedded.append(emb_i)
            embedded = torch.cat(embedded, dim=-1)
        else:
            embedded = act_embed
        
        x = torch.cat([embedded, nume_prefix], dim=-1) # (batch_size, prefix_len, enc_input_size)

        return x
    
    def argmax(self,
               cate_prefix, 
               nume_prefix,
               dec_input_act):
        
        batch_size = cate_prefix.shape[0]
        suffix_len = dec_input_act.shape[1]

        # -- encoder --
        x = self.process_enc_input(cate_prefix, nume_prefix) # (batch_size, prefix_len, enc_input_size)

        enc_states, hidden, cell = self.encoder(x)
        # enc_states (batch_size, prefix_len, hidden_size)
        # hidden (num_layers, batch_size, hidden_size)
        # cell (num_layers, batch_size, hidden_size)

        # -- decoder --

        all_act_logits = []

        dec_act = dec_input_act[:, :1] # (batch_size, 1)
        
        for _ in range(suffix_len):

            dec_act_embed = self.act_embedding(dec_act) # (batch_size, 1, act_embed)
            act_logits, hidden, cell = self.decoder(dec_act_embed, hidden, cell) # (batch_size, 1, num_act)
            act_logits = act_logits.squeeze(1) # (batch_size, num_act)
            
            # avoid sampling from PAD (0) and SOC (2)
            act_logits[:, [0, 2]] = float('-inf')

            all_act_logits.append(act_logits)

            dec_act = act_logits.argmax(1, keepdim=True) # (batch_size, 1)

        all_act_logits = torch.stack(all_act_logits, dim=1) # (batch_size, suffix_len, num_act)
        all_act_predictions = all_act_logits.argmax(2) # (batch_size, suffix_len)

        pred_len = lens_till_eoc(all_act_predictions, self.eoc_index)

        pred_list = [all_act_predictions[i, :int(pred_len[i])].tolist() 
                          for i in range(batch_size)]
        
        return all_act_logits, pred_list, pred_len

    def beam_search(self,
               cate_prefix, 
               nume_prefix,
               dec_input_act,
               beam_width,
               length_penalty=0.65):
        
        device = cate_prefix.device
        batch_size = cate_prefix.shape[0]
        suffix_len = dec_input_act.shape[1]
        K = beam_width

        def get_logprobs(dec_act,
                         hidden,
                         cell): 
            
            dec_act_embed = self.act_embedding(dec_act)
            act_logits, hidden, cell = self.decoder(dec_act_embed, 
                                                    hidden, 
                                                    cell)  

            act_logits = act_logits.squeeze(1)

            # avoid sampling from PAD (0) and SOC (2)
            act_logits[:, [0, 2]] = float('-inf')

            return F.log_softmax(act_logits, dim=-1), hidden, cell   

        # -- encoder --

        x = self.process_enc_input(cate_prefix, nume_prefix) # (batch_size, prefix_len, enc_input_size)

        enc_states, enc_hidden, enc_cell = self.encoder(x)
        # enc_states: (batch_size, prefix_len, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)

        # -- decoder --

        # first step
        dec_act = dec_input_act[:, :1] # (B, 1)
        logprobs, hidden, cell = get_logprobs(dec_act, enc_hidden, enc_cell)

        # first expansion: 1 -> K beams
        hidden = hidden.repeat_interleave(K, dim=1)  # (num_layers, batch_size*beam_width, hidden_size)
        cell = cell.repeat_interleave(K, dim=1)  # (num_layers, batch_size*beam_width, hidden_size)

        topk_logp, topk_indices = torch.topk(logprobs, k=K, dim=-1)  # (batch_size, beam_width), (batch_size, beam_width)
        scores = topk_logp  # (batch_size, beam_width)

        seqs = torch.stack([dec_act.repeat(1, K), topk_indices], dim=-1)  # (batch_size, beam_width, 2)
        gen = seqs[:, :, 1:] # (batch_size, beam_width, 1)
        flat_gen = gen.reshape(batch_size * K, 1) # (batch_size*beam_width, 1)
        flat_lengths = lens_till_eoc(flat_gen, eoc_index=self.eoc_index) # (batch_size*beam_width, )
        
        # track whether eoc is already generated
        alive = (topk_indices != self.eoc_index) # (batch_size, beam_width)

        while seqs.size(-1) < 1 + suffix_len:
            T = seqs.size(-1)

            # produce the input
            flat_seqs = seqs.reshape(batch_size * K, T)  # (batch_size*beam_width, T)
            dec_act = flat_seqs[:, -1:] # (batch_size*beam_width, 1)

            # produce the probs
            logprobs, hidden, cell = get_logprobs(dec_act, hidden, cell) # (batch_size*beam_width, num_act)

            # freeze finished beams: only allow EOC with logprob 0
            finished_logprobs = torch.full_like(logprobs, float('-inf'))  # (batch_size*beam_width, num_act)
            finished_logprobs[:, self.eoc_index] = 0.0
            flat_alive = alive.reshape(batch_size * K) # (batch_size*beam_width)
            logprobs = torch.where(flat_alive.unsqueeze(-1), logprobs, finished_logprobs)

            # get the lengths for length normalization
            increment = flat_alive.long()  # (batch_size*beam_width,)  1 if alive othwise 0
            flat_lengths = flat_lengths + increment
            lengths = flat_lengths.view(batch_size, K) # (batch_size, beam_width)

            # update the scores by adding probs at this step
            logprobs = logprobs.view(batch_size, K, -1)  # (batch_size, beam_width, num_act)
            
            cand_scores = scores.unsqueeze(-1) + logprobs  # (batch_size, beam_width, num_act) unnormalized scores

            # apply length normalization
            if length_penalty and length_penalty > 0:
                lp = ((5.0 + lengths.float()) / 6.0) ** length_penalty
                rank_scores = cand_scores / lp.unsqueeze(-1)
            else:
                rank_scores = cand_scores

            V = logprobs.size(-1)
            rank_scores_flat = rank_scores.view(batch_size, K * V) # (batch_size, beam_width * num_act)

            new_rank_scores, new_idx = torch.topk(rank_scores_flat, k=K, dim=-1)  # (batch_size, beam_width), (batch_size, beam_width)
            # get the raw scores
            cand_scores_flat = cand_scores.view(batch_size, K * V)
            new_scores = torch.gather(cand_scores_flat, dim=1, index=new_idx) # (batch_size, beam_width)

            # floor devision
            new_beam = new_idx // V  # (batch_size, beam_width)
            # integer remainder operator 
            new_indices = new_idx % V  # (batch_size, beam_width)

            gather_idx = new_beam.unsqueeze(-1).expand(batch_size, K, T)   # (batch_size, beam_width, T)
            seqs = torch.gather(seqs, dim=1, index=gather_idx)    # (batch_size, beam_width, T)
            seqs = torch.cat([seqs, new_indices.unsqueeze(-1)], dim=-1)   # (batch_size, beam_width, T+1)

            alive = torch.gather(alive, dim=1, index=new_beam) # (batch_size, beam_width)
            alive = alive & (new_indices != self.eoc_index)  # (batch_size, beam_width)

            lengths = flat_lengths.view(batch_size, K)
            lengths = torch.gather(lengths, dim=1, index=new_beam) # (batch_size, beam_width)
            flat_lengths = lengths.reshape(batch_size * K)  # (batch_size * beam_width

            scores = new_scores # (batch_size, beam_width

            hidden = hidden.view(self.num_layers, batch_size, K, self.hidden_size)
            cell = cell.view(self.num_layers, batch_size, K, self.hidden_size)
            gather_state_idx = new_beam.unsqueeze(0).unsqueeze(-1).expand(self.num_layers, batch_size, K, self.hidden_size)
            hidden = hidden.gather(dim=2, index=gather_state_idx).reshape(self.num_layers, batch_size*K, self.hidden_size)
            cell = cell.gather(dim=2, index=gather_state_idx).reshape(self.num_layers, batch_size*K, self.hidden_size)

        gen = seqs[:, :, 1:]  # (batch_size, beam_width,  suffix_len)
        lengths = lens_till_eoc(gen.reshape(batch_size*K, -1), self.eoc_index).view(batch_size, K)
        if length_penalty and length_penalty > 0:
            lp = ((5.0 + lengths.float()) / 6.0) ** length_penalty
            final_rank = scores / lp
        else:
            final_rank = scores

        best = final_rank.argmax(dim=1)

        all_act_predictions = gen[torch.arange(batch_size, device=device), best] # (batch_size suffix_len)   

        pred_len = lens_till_eoc(all_act_predictions, self.eoc_index)

        pred_list = [all_act_predictions[i, :int(pred_len[i])].tolist() 
                          for i in range(batch_size)]
            
        return pred_list, pred_len
    
    def top_p_sampler(self,
               cate_prefix, 
               nume_prefix,
               dec_input_act,
               p,
               temperature=1):
        
        batch_size = cate_prefix.shape[0]
        suffix_len = dec_input_act.shape[1]
        
        # -- encoder --

        x = self.process_enc_input(cate_prefix, nume_prefix) # (batch_size, prefix_len, enc_input_size)

        enc_states, hidden, cell = self.encoder(x)
        # enc_states: (batch_size, prefix_len, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)

        # -- decoder --

        all_act_predictions = []

        dec_act = dec_input_act[:, :1] # (batch_size, 1)

        temperature = float(temperature)

        for _ in range(suffix_len):

            dec_act_embed = self.act_embedding(dec_act) # (batch_size, 1, act_embed)
            act_logits, hidden, cell = self.decoder(dec_act_embed, hidden, cell) # (batch_size, 1, num_act)
            act_logits = act_logits.squeeze(1) # (batch_size, num_act)
            
            # avoid sampling from PAD (0) and SOC (2)
            act_logits[:, [0, 2]] = float('-inf')
            act_logits /= temperature

            probs = F.softmax(act_logits, dim=-1) # (batch_size, num_act)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_remove = cumulative_probs > p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            
            sorted_probs = sorted_probs.masked_fill(sorted_remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True) # Re-normalize probabilities

            next_act_position = torch.multinomial(sorted_probs, 1) # (batch_size, 1)
            next_act_pred = sorted_indices.gather(1, next_act_position) # (batch_size, 1)

            all_act_predictions.append(next_act_pred) 

            dec_act = next_act_pred
        
        all_act_predictions = torch.cat(all_act_predictions, dim=1) # (batch_size, suffix_len)

        pred_len = lens_till_eoc(all_act_predictions, self.eoc_index)

        pred_list = [all_act_predictions[i, :int(pred_len[i])].tolist() 
                          for i in range(batch_size)]

        return pred_list, pred_len


    def d_action(self,
               cate_prefix, 
               nume_prefix,
               dec_input_act,
               eps: float = 1e-12):
        
        batch_size = cate_prefix.shape[0]
        suffix_len = dec_input_act.shape[1]

        # -- encoder --
        x = self.process_enc_input(cate_prefix, nume_prefix) # (batch_size, prefix_len, enc_input_size)

        enc_states, hidden, cell = self.encoder(x)
        # enc_states (batch_size, prefix_len, hidden_size)
        # hidden (num_layers, batch_size, hidden_size)
        # cell (num_layers, batch_size, hidden_size)

        # -- decoder --

        all_act_predictions = []

        dec_act = dec_input_act[:, :1] # (batch_size, 1)

        act_prefix = cate_prefix[:, :, 0] # (batch_size, prefix_len)

        gen_acts = act_prefix.new_empty((batch_size, 0)) # (batch_size, 0)
        
        for _ in range(suffix_len):

            dec_act_embed = self.act_embedding(dec_act) # (batch_size, 1, act_embed)
            act_logits, hidden, cell = self.decoder(dec_act_embed, hidden, cell) # (batch_size, 1, num_act)
            act_logits = act_logits.squeeze(1) # (batch_size, num_act)
            
            # avoid sampling from PAD (0) and SOC (2)
            act_logits[:, [0, 2]] = float('-inf')

            # --- Daemon Action ---
            probs = torch.softmax(act_logits, dim=-1)  
            num_act = probs.size(-1)
            hist = torch.cat([act_prefix, gen_acts], dim=1)  # (batch_size, prefix_len + t)

            valid = (hist != 0) & (hist != 2)

            counts = torch.zeros((batch_size, num_act), device=probs.device, dtype=torch.float32)
            for i in range(batch_size):
                h_i = hist[i][valid[i]]
                if h_i.numel() > 0:
                    counts[i] = torch.bincount(h_i, minlength=num_act).float()
            
            adjusted_probs = probs.clone()

            # penalize only seen activities (count > 0)
            seen_act = counts > 0
            adjusted_probs[seen_act] = probs[seen_act] / counts[seen_act]

            # avoid sampling from PAD (0) and SOC (2)
            adjusted_probs[:, [0, 2]] = 0.0
            dec_act = adjusted_probs.argmax(1, keepdim=True) # (batch_size, 1)

            all_act_predictions.append(dec_act)

            gen_acts = torch.cat([gen_acts, dec_act], dim=1)

        all_act_predictions = torch.cat(all_act_predictions, dim=1) # (batch_size, suffix_len)

        pred_len = lens_till_eoc(all_act_predictions, self.eoc_index)

        pred_list = [all_act_predictions[i, :int(pred_len[i])].tolist() 
                          for i in range(batch_size)]
        
        return pred_list, pred_len

    def mc_bridge(self,
            cate_prefix,
            nume_prefix,
            dec_input_act,
            n_candidate,
            n_sample,
            sampling='random',
            k=10,
            p=0.9,
            diff=False):
        """
        BRIDGE with Monte Carlo estimator
        """
        
        device = cate_prefix.device
        batch_size = cate_prefix.shape[0]
        eoc_tensor = torch.tensor(self.eoc_index, device=device)
        suffix_len = dec_input_act.shape[1]
        
        # -- encoder --

        x = self.process_enc_input(cate_prefix, nume_prefix) # (batch_size, prefix_len, enc_input_size)

        enc_states, enc_hidden, enc_cell = self.encoder(x)
        # enc_states: (batch_size, prefix_len, hidden_size)
        # enc_hidden: (num_layers, batch_size, hidden_size)
        # enc_cell: (num_layers, batch_size, hidden_size)

        def generate_samples(n, sampling_mode):

            predictions = torch.empty((batch_size * n, suffix_len), 
                              dtype=torch.long, 
                              device=device)
            
            # track whether eoc is already generated
            alive = torch.ones((batch_size * n, ), dtype=torch.bool, device=device) # filled with True

            # -- encoder --
            hidden = enc_hidden.repeat_interleave(n, dim=1) # (batch_size * n)
            cell = enc_cell.repeat_interleave(n, dim=1) # (batch_size * n)
            
            # -- decoder --
            dec_act = dec_input_act[:, :1] # (batch_size, 1)
            dec_act = dec_act.repeat_interleave(n, dim=0) # (batch_size * n, 1)

            def filter_logits(logits):

                V = logits.size(-1)

                # avoid sampling PAD (0) and SOC (2)
                logits = logits.clone()
                logits[:, [0, 2]] = float('-inf')

                if sampling_mode == "random":
                    return logits

                if sampling_mode == "top_k":
                    k = min(k, V)
                    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
                    filtered = torch.full_like(logits, float('-inf'))
                    filtered.scatter_(1, top_k_indices, top_k_logits)
                    return filtered

                if sampling_mode == "top_p":
                    probs = F.softmax(logits, dim=-1)  
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)

                    remove = cum_probs > p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False

                    keep_sorted = ~remove
                    keep_vocab = torch.zeros_like(keep_sorted).scatter(1, sorted_idx, keep_sorted)

                    filtered = logits.masked_fill(~keep_vocab, float('-inf'))
                    return filtered

                raise ValueError(f"Unknown sampling mode: {sampling_mode}")
            
            for t in range(suffix_len):

                dec_act_embed = self.act_embedding(dec_act) # (batch_size * n, 1, act_embed)
                act_logits, hidden, cell = self.decoder(dec_act_embed, hidden, cell) # (batch_size * n, 1, num_act)
                act_logits = act_logits.squeeze(1) # (batch_size * n, num_act)

                filtered_logits = filter_logits(act_logits)  

                probs = F.softmax(filtered_logits, dim=-1) # (batch_size * n, num_act)
                next_act = torch.multinomial(probs, 1).squeeze(-1) # (batch_size * n)

                # If a row is no longer alive, keep appending eoc_idx
                next_act = torch.where(alive, next_act, eoc_tensor)

                predictions[:, t] = next_act

                dec_act = next_act.unsqueeze(-1) # (batch_size, 1)

                # update alive AFTER writing next_act
                alive = alive & (next_act != self.eoc_index)

                if not alive.any():
                    if t + 1 < suffix_len:
                        predictions[:, t+1:] = self.eoc_index
                    break
            
            return predictions
        
        if diff:
            candidates = generate_samples(n_candidate, sampling_mode=sampling)
            samples = generate_samples(n_sample, sampling_mode='random')
        else:
            assert n_candidate == n_sample, "Error: n_candidate is different from n_sample"
            samples = generate_samples(n_sample, sampling_mode='random')
            candidates = samples.clone()

        lens_sample = lens_till_eoc(samples, self.eoc_index)

        cand_3d = candidates.view(batch_size, n_candidate, suffix_len)
        samp_3d = samples.view(batch_size, n_sample, suffix_len)
        ls_2d = lens_sample.view(batch_size, n_sample)

        cand_3d = cand_3d.detach().cpu()
        samp_3d = samp_3d.detach().cpu()
        ls_2d = ls_2d.detach().cpu()

        pred_list = []
        pred_lens = []

        for b in range(batch_size):

            cand_block = cand_3d[b] 

            uniq_cands = torch.unique(cand_block, dim=0)

            uniq_lens = lens_till_eoc(uniq_cands, self.eoc_index)

            cand_lists = [uniq_cands[i, :int(uniq_lens[i])].tolist() 
                          for i in range(uniq_cands.size(0))]
            samp_lists = [samp_3d[b, j, :int(ls_2d[b, j])].tolist() 
                          for j in range(n_sample)]
            
            Nc_u = len(cand_lists)

            if Nc_u <= n_sample:
                rows = [np.asarray(dl_distance_seqs(c, samp_lists), dtype=np.float32) 
                        for c in cand_lists]
                D = np.stack(rows, axis=0) 
            
            else:
                cols = [np.asarray(dl_distance_seqs(s, cand_lists), dtype=np.float32) 
                        for s in samp_lists]
                D = np.stack(cols, axis=0).T 
            
            avg = D.mean(axis=1) 
            best_idx = int(avg.argmin())
            pred_list.append(cand_lists[best_idx])
            pred_lens.append(int(uniq_lens[best_idx]))
        
        pred_len = torch.tensor(pred_lens)

        return pred_list, pred_len
    
    def mb_bridge(self,
            cate_prefix,
            nume_prefix,
            dec_input_act,
            n_candidate,
            n_sample,
            sampling='random',
            k=10,
            p=0.9,
            length_norm=False,
            diff=False):
        """
        BRIDGE with model-based estimator
        """

        device = cate_prefix.device
        batch_size = cate_prefix.shape[0]
        eoc_tensor = torch.tensor(self.eoc_index, device=device)
        suffix_len = dec_input_act.shape[1]
        
        # -- encoder --

        x = self.process_enc_input(cate_prefix, nume_prefix) # (batch_size, prefix_len, enc_input_size)

        enc_states, enc_hidden, enc_cell = self.encoder(x)
        # enc_states: (batch_size, prefix_len, hidden_size)
        # enc_hidden: (num_layers, batch_size, hidden_size)
        # enc_cell: (num_layers, batch_size, hidden_size)

        def generate_samples(n, sampling_mode):

            predictions = torch.empty((batch_size * n, suffix_len), 
                              dtype=torch.long, 
                              device=device)
            
            # store per-step log probability
            logprobs = torch.zeros((batch_size * n, suffix_len),
                                dtype=torch.float,
                                device=device)
            
            # track whether eoc is already generated
            alive = torch.ones((batch_size * n, ), dtype=torch.bool, device=device) # filled with True

            # -- encoder --
            hidden = enc_hidden.repeat_interleave(n, dim=1) # (batch_size * n)
            cell = enc_cell.repeat_interleave(n, dim=1) # (batch_size * n)
            
            # -- decoder --
            dec_act = dec_input_act[:, :1] # (batch_size, 1)
            dec_act = dec_act.repeat_interleave(n, dim=0) # (batch_size * n, 1)
            
            def filter_logits(logits):

                V = logits.size(-1)

                # avoid sampling PAD (0) and SOC (2)
                logits = logits.clone()
                logits[:, [0, 2]] = float('-inf')

                if sampling_mode == "random":
                    return logits

                if sampling_mode == "top_k":
                    k = min(k, V)
                    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
                    filtered = torch.full_like(logits, float('-inf'))
                    filtered.scatter_(1, top_k_indices, top_k_logits)
                    return filtered

                if sampling_mode == "top_p":
                    probs = F.softmax(logits, dim=-1) 
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)

                    remove = cum_probs > p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False

                    keep_sorted = ~remove
                    keep_vocab = torch.zeros_like(keep_sorted).scatter(1, sorted_idx, keep_sorted)

                    filtered = logits.masked_fill(~keep_vocab, float('-inf'))
                    return filtered

                raise ValueError(f"Unknown sampling mode: {sampling_mode}")
    
            for t in range(suffix_len):

                dec_act_embed = self.act_embedding(dec_act) # (batch_size * n, 1, act_embed)
                act_logits, hidden, cell = self.decoder(dec_act_embed, hidden, cell) # (batch_size * n, 1, num_act)
                act_logits = act_logits.squeeze(1) # (batch_size * n, num_act)

                filtered_logits = filter_logits(act_logits)  

                probs = F.softmax(filtered_logits, dim=-1) # (batch_size * n, num_act)
                next_act = torch.multinomial(probs, 1).squeeze(-1) # (batch_size * n)

                # If a row is no longer alive, keep appending eoc_idx
                next_act = torch.where(alive, next_act, eoc_tensor)

                predictions[:, t] = next_act

                # store log probs
                filtered_logprobs = F.log_softmax(filtered_logits, dim=-1)  
                chosen_lp = filtered_logprobs.gather(1, next_act.unsqueeze(1)).squeeze(1)
                # if not alive, make the step logprob 0 so it doesn't affect products
                chosen_lp = torch.where(alive, chosen_lp, torch.zeros_like(chosen_lp))
                logprobs[:, t] = chosen_lp

                dec_act = next_act.unsqueeze(-1) # (batch_size, 1)

                # Update alive AFTER writing next_act
                alive = alive & (next_act != self.eoc_index)

                if not alive.any():
                    if t + 1 < suffix_len:
                        predictions[:, t+1:] = self.eoc_index
                        logprobs[:, t+1:] = 0.0
                    break
            
            return predictions, logprobs
        
        if diff:
            candidates, _ = generate_samples(n_candidate, sampling_mode=sampling)
            samples, logprobs = generate_samples(n_sample, sampling_mode='random')
        else:
            assert n_candidate == n_sample, "Error: n_candidate is different from n_sample"
            samples, logprobs = generate_samples(n_sample, sampling_mode='random')
            candidates = samples.clone()

        cand_3d = candidates.view(batch_size, n_candidate, suffix_len)
        samp_3d = samples.view(batch_size, n_sample, suffix_len)
        lp_3d = logprobs.view(batch_size, n_sample, suffix_len)

        cand_3d = cand_3d.detach().cpu()
        samp_3d = samp_3d.detach().cpu()
        lp_3d = lp_3d.detach().cpu()

        pred_list = []
        pred_lens = []

        for b in range(batch_size):

            cand_block = cand_3d[b] # (n_candidate, suffix_len)
            samp_block = samp_3d[b] # (n_sample, suffix_len)
            lp_block = lp_3d[b]

            uniq_cands = torch.unique(cand_block, dim=0)
            uniq_samps, inv = torch.unique(samp_block, dim=0, return_inverse=True)

            cand_lens = lens_till_eoc(uniq_cands, self.eoc_index)
            samp_lens = lens_till_eoc(uniq_samps, self.eoc_index)

            cand_lists = [uniq_cands[i, :int(cand_lens[i])].tolist() 
                          for i in range(uniq_cands.size(0))]
            samp_lists = [uniq_samps[j, :int(samp_lens[j])].tolist() 
                          for j in range(uniq_samps.size(0))]
            
            # get logprobs for unique samples (first occurrence)
            n_samp = len(samp_lists)
            N = inv.numel()
            pos = torch.arange(N, device=inv.device, dtype=torch.long)
            first_pos = torch.full((n_samp,), N, device=inv.device, dtype=torch.long)
            first_pos.scatter_reduce_(0, inv, pos, reduce="amin", include_self=True)
            uniq_lp = lp_block[first_pos]

            # compute score per unique sample
            N, L = uniq_lp.shape
            ar = torch.arange(L, device=uniq_lp.device).unsqueeze(0)
            mask = (ar < samp_lens.unsqueeze(1)).to(uniq_lp.dtype)
            seq_logp = (uniq_lp * mask).sum(dim=1)

            if length_norm:
                log_score = seq_logp + samp_lens.to(seq_logp.dtype)  
            else:
                log_score = seq_logp
            
            log_score_np = log_score.numpy()
            m = log_score_np.max()
            w = np.exp(log_score_np - m).astype(np.float32)    
            w = w / (w.sum() + 1e-8)

            rows = [np.asarray(dl_distance_seqs(c, samp_lists), dtype=np.float32) 
                    for c in cand_lists]
            D = np.stack(rows, axis=0) 

            D = D * w[None, :] 
            
            avg = D.sum(axis=1)
            best_idx = int(avg.argmin())
            pred_list.append(cand_lists[best_idx])
            pred_lens.append(int(cand_lens[best_idx]))
        
        pred_len = torch.tensor(pred_lens)

        return pred_list, pred_len
    
    def prob_rank(self,
            cate_prefix,
            nume_prefix,
            dec_input_act,
            n_sample):
        """
        Re-ranking method that randomly samples suffixes and ranks them by likelihood
        """
        
        device = cate_prefix.device
        batch_size = cate_prefix.shape[0]
        eoc_tensor = torch.tensor(self.eoc_index, device=device)
        suffix_len = dec_input_act.shape[1]
        
        # -- encoder --

        x = self.process_enc_input(cate_prefix, nume_prefix) # (batch_size, prefix_len, enc_input_size)

        enc_states, enc_hidden, enc_cell = self.encoder(x)
        # enc_states: (batch_size, prefix_len, hidden_size)
        # enc_hidden: (num_layers, batch_size, hidden_size)
        # enc_cell: (num_layers, batch_size, hidden_size)

        def generate_samples_with_logprob(n):

            predictions = torch.empty((batch_size * n, suffix_len), 
                              dtype=torch.long, 
                              device=device)
            
            alive = torch.ones((batch_size * n, ), dtype=torch.bool, device=device)

            seq_logprob = torch.zeros((batch_size * n,), dtype=torch.float32, device=device)

            # -- encoder --
            hidden = enc_hidden.repeat_interleave(n, dim=1) # (batch_size * n)
            cell = enc_cell.repeat_interleave(n, dim=1) # (batch_size * n)
            
            # -- decoder --
            dec_act = dec_input_act[:, :1] # (batch_size, 1)
            dec_act = dec_act.repeat_interleave(n, dim=0) # (batch_size * n, 1)
            
            for t in range(suffix_len):

                dec_act_embed = self.act_embedding(dec_act) # (batch_size * n, 1, act_embed)
                act_logits, hidden, cell = self.decoder(dec_act_embed, hidden, cell) # (batch_size * n, 1, num_act)
                act_logits = act_logits.squeeze(1) # (batch_size * n, num_act)

                # avoid sampling from PAD (0) and SOC (2)
                act_logits[:, [0, 2]] = float('-inf')

                log_probs = F.log_softmax(act_logits, dim=-1)
                probs = F.softmax(act_logits, dim=-1) # (batch_size * n, num_act)
                sampled_next_act = torch.multinomial(probs, 1).squeeze(-1) # (batch_size * n)

                # If a row is no longer alive, keep appending eoc_idx
                next_act = torch.where(alive, sampled_next_act, eoc_tensor)

                predictions[:, t] = next_act

                step_logprob = log_probs.gather(1, sampled_next_act.unsqueeze(1)).squeeze(1)
                seq_logprob = seq_logprob + torch.where(
                    alive,
                    step_logprob,
                    torch.zeros_like(step_logprob)
                    )

                dec_act = next_act.unsqueeze(-1) # (batch_size, 1)

                # Update alive AFTER writing next_act
                alive = alive & (next_act != self.eoc_index)

                if not alive.any():
                    if t + 1 < suffix_len:
                        predictions[:, t+1:] = self.eoc_index
                    break
            
            return predictions, seq_logprob
        
        samples, sample_logprob = generate_samples_with_logprob(n_sample)
        lens_sample = lens_till_eoc(samples, self.eoc_index)

        samp_3d = samples.view(batch_size, n_sample, suffix_len)
        lp_2d = sample_logprob.view(batch_size, n_sample)
        ls_2d = lens_sample.view(batch_size, n_sample)

        best_idx = lp_2d.argmax(dim=1)  # (batch_size,)
        batch_idx = torch.arange(batch_size)
        best_lens = ls_2d[batch_idx, best_idx]                    # (batch_size,)
        best_seqs = samp_3d[batch_idx, best_idx, :]
                      # (batch_size, suffix_len)
        pred_list = [best_seqs[b, :best_lens[b]].tolist() for b in range(batch_size)]

        pred_len = best_lens
        
        return pred_list, pred_len
    
def lens_till_eoc(x: torch.Tensor, eoc_index: int) -> torch.Tensor:
        
        """
        Return length ending at EOC (inclusive) if EOC is present
        """
    
        B, T = x.shape
        is_eoc = (x == eoc_index) # (batch_size, suffix_len) bool
        has_eoc = is_eoc.any(dim=1) # (batch_size, ) bool

        # argmax gives first True only if there is any True; otherwise 0.
        first_eoc = is_eoc.int().argmax(dim=1) # (batch_size, ) long

        lengths = torch.where(has_eoc, first_eoc + 1, torch.full_like(first_eoc, T))

        return lengths
