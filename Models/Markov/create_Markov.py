import numpy as np
import torch
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance_seqs as dl_distance_seqs

class MarkovModel:

    def __init__(self, suffix_len: int, vocab_size: int, alpha: float = 1.0):
        self.suffix_len = int(suffix_len)
        self.V = int(vocab_size)
        self.alpha = float(alpha)

        self.probs = None 

    @torch.no_grad()
    def fit(self, trigrams: torch.Tensor):

        V = self.V
        device = trigrams.device

        tri = trigrams.to(torch.long)
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]

        # first-order Markov model
        idx2 = b * V + c                                    
        bigram_counts = torch.bincount(idx2, minlength=V * V).reshape(V, V).to(torch.float32)

        bigram_row_sum = bigram_counts.sum(dim=1)          
        bigram_seen = bigram_row_sum > 0                 

        bigram_probs = torch.empty_like(bigram_counts)     
        bigram_probs[bigram_seen] = bigram_counts[bigram_seen] / bigram_row_sum[bigram_seen].unsqueeze(1)
        bigram_probs[~bigram_seen] = 1.0 / V   # uniform if unseen b

        # second-order Markov model
        idx3 = a * (V * V) + b * V + c                      
        trigram_counts = torch.bincount(idx3, minlength=V * V * V).reshape(V * V, V).to(torch.float32)

        tri_row_sum = trigram_counts.sum(dim=1)            
        tri_seen = tri_row_sum > 0                        

        probs = torch.empty_like(trigram_counts)         
        
        # seen contexts: second-order Markov model
        probs[tri_seen] = trigram_counts[tri_seen] / tri_row_sum[tri_seen].unsqueeze(1)

        # unseen contexts: back off to first-order Markov model
        row_ids = torch.arange(V * V, device=device)
        b_for_row = row_ids % V                          

        unseen_rows = ~tri_seen
        probs[unseen_rows] = bigram_probs[b_for_row[unseen_rows]]

        probs[:, [0, 2]] = 0.0
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

        self.probs = probs.to(device=device, dtype=torch.float32)     
        self.bigram_probs = bigram_probs.to(device=device, dtype=torch.float32)
        return self

    @torch.no_grad()
    def next_probs(self, contexts: torch.Tensor) -> torch.Tensor:
        V = self.V
        ctx = contexts.to(torch.long)
        rows = ctx[:, 0] * V + ctx[:, 1]  # (B,)
        return self.probs.index_select(0, rows)

    @torch.no_grad()
    def argmax(self, ctx: torch.Tensor, eoc_id: int = 3):
        device = ctx.device
        B = ctx.shape[0]

        ctx = ctx.clone()

        preds = torch.empty((B, self.suffix_len), device=device, dtype=torch.long)

        for t in range(self.suffix_len):
            p = self.next_probs(ctx)                    
            nxt = torch.argmax(p, dim=1)          
            preds[:, t] = nxt
            ctx[:, 0] = ctx[:, 1]
            ctx[:, 1] = nxt

        pred_len = lens_till_eoc(preds, int(eoc_id))

        preds_cpu = preds.cpu()
        pred_len_cpu = pred_len.cpu()

        pred_list = [
            preds_cpu[i, : int(pred_len_cpu[i])].tolist()
            for i in range(B)
        ]
        return pred_list, pred_len

    @torch.no_grad()
    def beam_search(self, prefixes: torch.Tensor, beam_width: int, eoc_id: int = 3, length_penalty: float = 0.65):
        
        device = prefixes.device
        V = self.V
        B = prefixes.size(0)
        K = int(beam_width)
        T_max = self.suffix_len
        eoc_id = int(eoc_id)

        prefixes = prefixes.clone()

        def get_logprobs_from_ctx(flat_ctx: torch.Tensor) -> torch.Tensor:
            probs = self.next_probs(flat_ctx)  
            return torch.log(probs)

        logprobs0 = get_logprobs_from_ctx(prefixes.to(torch.long)) 

        topk_logp, topk_tok = torch.topk(logprobs0, k=K, dim=-1)
        scores = topk_logp                            

        gen = topk_tok.unsqueeze(-1)          

        alive = (topk_tok != eoc_id)    

        lengths = torch.ones((B, K), device=device, dtype=torch.long)

        ctx = torch.empty((B, K, 2), device=device, dtype=torch.long)
        ctx[:, :, 0] = prefixes[:, 1].unsqueeze(1).expand(B, K)
        ctx[:, :, 1] = topk_tok

        for _ in range(1, T_max): 
            flat_ctx = ctx.reshape(B * K, 2)

            logprobs = get_logprobs_from_ctx(flat_ctx) 

            finished_logprobs = torch.full_like(logprobs, float("-inf"))
            finished_logprobs[:, eoc_id] = 0.0

            flat_alive = alive.reshape(B * K)
            logprobs = torch.where(flat_alive.unsqueeze(-1), logprobs, finished_logprobs)

            flat_lengths = lengths.reshape(B * K)
            flat_lengths = flat_lengths + flat_alive.long()
            lengths = flat_lengths.view(B, K) 

            logprobs = logprobs.view(B, K, V)     
            cand_scores = scores.unsqueeze(-1) + logprobs 

            if length_penalty and length_penalty > 0:
                lp = ((5.0 + lengths.float()) / 6.0) ** float(length_penalty)
                rank_scores = cand_scores / lp.unsqueeze(-1)   
            else:
                rank_scores = cand_scores

            rank_scores_flat = rank_scores.view(B, K * V) 
            new_rank, new_idx = torch.topk(rank_scores_flat, k=K, dim=-1)

            cand_scores_flat = cand_scores.view(B, K * V)
            new_scores = torch.gather(cand_scores_flat, dim=1, index=new_idx)

            new_beam = new_idx // V     
            new_tok = new_idx % V   

            t_cur = gen.size(-1)
            gather_gen_idx = new_beam.unsqueeze(-1).expand(B, K, t_cur)
            gen = torch.gather(gen, dim=1, index=gather_gen_idx)   
            gen = torch.cat([gen, new_tok.unsqueeze(-1)], dim=-1)  

            alive = torch.gather(alive, dim=1, index=new_beam)  
            alive = alive & (new_tok != eoc_id)

            lengths = torch.gather(lengths, dim=1, index=new_beam) 

            gather_ctx_idx = new_beam.unsqueeze(-1).expand(B, K, 2)
            prev_ctx = torch.gather(ctx, dim=1, index=gather_ctx_idx) 
            ctx = torch.empty_like(prev_ctx)
            ctx[:, :, 0] = prev_ctx[:, :, 1]
            ctx[:, :, 1] = new_tok

            scores = new_scores

        flat_gen = gen.reshape(B * K, T_max)
        flat_len = lens_till_eoc(flat_gen, eoc_id).to(torch.long)  
        lengths_final = flat_len.view(B, K)     

        if length_penalty and length_penalty > 0:
            lp = ((5.0 + lengths_final.float()) / 6.0) ** float(length_penalty)
            final_rank = scores / lp
        else:
            final_rank = scores

        best = final_rank.argmax(dim=1)

        best_seq = gen[torch.arange(B, device=device), best]  
        pred_len = lens_till_eoc(best_seq, eoc_id)  

        best_seq_cpu = best_seq.cpu()
        pred_len_cpu = pred_len.cpu()

        pred_list = [best_seq_cpu[i, : int(pred_len_cpu[i])].tolist() for i in range(B)]
        return pred_list, pred_len
    
    @torch.no_grad()
    def top_p_sampler(self, ctx: torch.Tensor, p: float, eoc_id: int = 3):
        
        B = ctx.shape[0]
        p = float(p)

        ctx = ctx.clone()
        preds = torch.empty((B, self.suffix_len), device=ctx.device, dtype=torch.long)

        for t in range(self.suffix_len):
            probs = self.next_probs(ctx)

            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1) 
            cum = torch.cumsum(sorted_probs, dim=-1)                         

            remove = cum > p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False

            filtered = sorted_probs.masked_fill(remove, 0.0)
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)

            next_act_position = torch.multinomial(filtered, 1)        
            next_act_pred = sorted_idx.gather(1, next_act_position).squeeze(1) 

            preds[:, t] = next_act_pred
            ctx[:, 0] = ctx[:, 1]
            ctx[:, 1] = next_act_pred

        pred_len_t = lens_till_eoc(preds, int(eoc_id))
        preds_cpu = preds.cpu()
        pred_len_cpu = pred_len_t.cpu()

        pred_list = [preds_cpu[i, : int(pred_len_cpu[i])].tolist() for i in range(B)]
        return pred_list, pred_len_t

    @torch.no_grad()
    def d_action(self, ctx: torch.Tensor, eoc_id: int = 3):
        device = ctx.device
        B = ctx.shape[0]

        ctx = ctx.clone()

        preds = torch.empty((B, self.suffix_len), device=device, dtype=torch.long)
        act_prefix = ctx.clone() # (batch_size, 2)
        gen_acts = act_prefix.new_empty((B, 0)) # (batch_size, 0)

        for t in range(self.suffix_len):
            probs = self.next_probs(ctx)                 
            
            # --- Daemon Action ---
            num_act = probs.size(-1)
            hist = torch.cat([act_prefix, gen_acts], dim=1)  # (batch_size, prefix_len + T)

            valid = (hist != 0) & (hist != 2)

            counts = torch.zeros((B, num_act), device=probs.device, dtype=torch.float32)
            for i in range(B):
                h_i = hist[i][valid[i]]
                if h_i.numel() > 0:
                    counts[i] = torch.bincount(h_i, minlength=num_act).float()
            
            adjusted_probs = probs.clone()

            # penalize only seen activities (count > 0)
            seen_act = counts > 0

            adjusted_probs[seen_act] = probs[seen_act] / counts[seen_act]

            # avoid sampling from PAD (0) and SOC (2)
            adjusted_probs[:, [0, 2]] = 0.0

            nxt = torch.argmax(adjusted_probs, dim=1)            
            preds[:, t] = nxt
            ctx[:, 0] = ctx[:, 1]
            ctx[:, 1] = nxt

            gen_acts = torch.cat([gen_acts, nxt.unsqueeze(-1)], dim=1)

        pred_len = lens_till_eoc(preds, int(eoc_id)) 

        preds_cpu = preds.cpu()
        pred_len_cpu = pred_len.cpu()

        pred_list = [
            preds_cpu[i, : int(pred_len_cpu[i])].tolist()
            for i in range(B)
        ]
        return pred_list, pred_len
    
    @torch.no_grad()
    def mc_bridge(
        self,
        prefixes,
        n_candidate,
        n_sample,
        sampling='random',
        k=10,
        p=0.9,
        diff=False,
        eoc_id=int(3),
        banned_ids=(0, 2),
    ):
        """
        BRIDGE with Monte Carlo estimator
        """

        device = prefixes.device
        B = prefixes.size(0)
        T = self.suffix_len
        V = self.V
        eoc_id = int(eoc_id)
        banned_ids = list(banned_ids)

        eoc_tensor = torch.tensor(eoc_id, device=device, dtype=torch.long)

        prefixes = prefixes.clone()

        def generate_samples(n, sampling_mode):
            n = int(n)

            predictions = torch.empty((B * n, T), dtype=torch.long, device=device)

            alive = torch.ones((B * n,), dtype=torch.bool, device=device)

            ctx = prefixes.to(torch.long).repeat_interleave(n, dim=0)

            def filter_probs(probs):
                p = probs.clone()

                if banned_ids:
                    p[:, banned_ids] = 0.0

                if sampling_mode == "random":
                    denom = p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    return p / denom

                if sampling_mode == "top_k":
                    k = min(int(k), V)
                    topv, topi = torch.topk(p, k, dim=-1)
                    out = torch.zeros_like(p)
                    out.scatter_(1, topi, topv)
                    out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    return out

                if sampling_mode == "top_p":
                    p_thr = float(p)
                    sorted_p, sorted_i = torch.sort(p, descending=True, dim=-1)
                    cum = torch.cumsum(sorted_p, dim=-1)

                    remove = cum > p_thr
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False

                    sorted_p = sorted_p.masked_fill(remove, 0.0)
                    sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                    out = torch.zeros_like(p)
                    out.scatter_(1, sorted_i, sorted_p)
                    return out

                raise ValueError(f"Unknown sampling mode: {sampling_mode}")

            for t in range(T):
                base_probs = self.next_probs(ctx)  
                filt_probs = filter_probs(base_probs) 

                next_act = torch.multinomial(filt_probs, 1).squeeze(1)

                next_act = torch.where(alive, next_act, eoc_tensor)

                predictions[:, t] = next_act

                alive = alive & (next_act != eoc_id)

                ctx = torch.stack([ctx[:, 1], next_act], dim=1)

                if not alive.any():
                    if t + 1 < T:
                        predictions[:, t + 1 :] = eoc_id
                    break

            return predictions

        if diff:
            candidates = generate_samples(n_candidate, sampling_mode=sampling) 
            samples = generate_samples(n_sample, sampling_mode='random') 
        else:
            assert n_candidate == n_sample, "Error: n_candidate is different from n_sample"
            samples = generate_samples(n_sample, sampling_mode='random')
            candidates = samples.clone()
        
        lens_sample = lens_till_eoc(samples, eoc_id)

        cand_3d = candidates.view(B, n_candidate, T)
        samp_3d = samples.view(B, n_sample, T)
        ls_2d = lens_sample.view(B, n_sample)

        cand_3d = cand_3d.detach().cpu()
        samp_3d = samp_3d.detach().cpu()
        ls_2d = ls_2d.detach().cpu()

        pred_list = []
        pred_lens = []

        for b in range(B):

            cand_block = cand_3d[b] 

            uniq_cands = torch.unique(cand_block, dim=0) 
            uniq_lens = lens_till_eoc(uniq_cands, eoc_id)

            cand_lists = [
                uniq_cands[i, : int(uniq_lens[i])].tolist()
                for i in range(uniq_cands.size(0))
            ]
            samp_lists = [
                samp_3d[b, j, : int(ls_2d[b, j])].tolist()
                for j in range(n_sample)
            ]

            Nc_u = len(cand_lists)

            if Nc_u <= n_sample:
                rows = [
                    np.asarray(dl_distance_seqs(c, samp_lists), dtype=np.float32)
                    for c in cand_lists
                ]
                D = np.stack(rows, axis=0)
            else:
                cols = [
                    np.asarray(dl_distance_seqs(s, cand_lists), dtype=np.float32)
                    for s in samp_lists
                ]
                D = np.stack(cols, axis=0).T

            avg = D.mean(axis=1)  
            best_idx = int(avg.argmin())

            pred_list.append(cand_lists[best_idx])
            pred_lens.append(int(uniq_lens[best_idx]))

        pred_len = torch.tensor(pred_lens, dtype=torch.long, device=device)

        return pred_list, pred_len

    @torch.no_grad()
    def mb_bridge(
        self,
        prefixes, 
        n_candidate,
        n_sample,
        sampling = "random",
        k = 10,
        p = 0.9,
        length_norm = False,
        diff = False,
        eoc_id = 3,
        banned_ids=(0, 2),    
    ):
        """
        BRIDGE with model-based estimator
        """

        device = prefixes.device
        B = prefixes.size(0)
        T = self.suffix_len
        V = self.V
        eoc_id = int(eoc_id)
        banned_ids = list(banned_ids)

        eoc_tensor = torch.tensor(eoc_id, device=device, dtype=torch.long)

        prefixes = prefixes.clone()

        def generate_samples(n, sampling_mode):
    
            n = int(n)
            BN = B * n

            predictions = torch.empty((BN, T), dtype=torch.long, device=device)
            logprobs = torch.zeros((BN, T), dtype=torch.float32, device=device)

            alive = torch.ones((BN,), dtype=torch.bool, device=device)
            ctx = prefixes.to(torch.long).repeat_interleave(n, dim=0)

            def filter_probs(probs: torch.Tensor) -> torch.Tensor:
            
                p = probs.clone()

                if banned_ids:
                    p[:, banned_ids] = 0.0

                if sampling_mode == "random":
                    denom = p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    return p / denom

                if sampling_mode == "top_k":
                    k = min(int(k), V)
                    topv, topi = torch.topk(p, k, dim=-1)  # (BN,k)
                    out = torch.zeros_like(p)
                    out.scatter_(1, topi, topv)
                    out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    return out

                if sampling_mode == "top_p":
                    p_thr = float(p)
                    sorted_p, sorted_i = torch.sort(p, descending=True, dim=-1)   # (BN,V)
                    cum = torch.cumsum(sorted_p, dim=-1)

                    remove = cum > p_thr
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False

                    sorted_p = sorted_p.masked_fill(remove, 0.0)
                    sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                    out = torch.zeros_like(p)
                    out.scatter_(1, sorted_i, sorted_p)
                    return out

                raise ValueError(f"Unknown sampling mode: {sampling_mode}")

            for t in range(T):
                base_probs = self.next_probs(ctx)  
                filt_probs = filter_probs(base_probs)  

                next_act = torch.multinomial(filt_probs, 1).squeeze(1)

                next_act = torch.where(alive, next_act, eoc_tensor)
                predictions[:, t] = next_act

                chosen_p = filt_probs.gather(1, next_act.unsqueeze(1)).squeeze(1) 
                chosen_lp = torch.log(chosen_p.clamp_min(1e-12))
                chosen_lp = torch.where(alive, chosen_lp, torch.zeros_like(chosen_lp))
                logprobs[:, t] = chosen_lp

                alive = alive & (next_act != eoc_id)

                ctx = torch.stack([ctx[:, 1], next_act], dim=1)

                if not alive.any():
                    if t + 1 < T:
                        predictions[:, t + 1 :] = eoc_id
                        logprobs[:, t + 1 :] = 0.0
                    break

            return predictions, logprobs

        if diff:
            candidates, _ = generate_samples(n_candidate, sampling_mode=sampling)
            samples, logprobs = generate_samples(n_sample, sampling_mode='random')
        else:
            assert n_candidate == n_sample, "Error: n_candidate is different from n_sample"
            samples, logprobs = generate_samples(n_sample, sampling_mode='random')
            candidates = samples.clone()

        cand_3d = candidates.view(B, n_candidate, T)
        samp_3d = samples.view(B, n_sample, T)
        lp_3d = logprobs.view(B, n_sample, T)

        cand_3d = cand_3d.detach().cpu()
        samp_3d = samp_3d.detach().cpu()
        lp_3d = lp_3d.detach().cpu()

        pred_list = []
        pred_lens = []

        for b in range(B):
            cand_block = cand_3d[b]
            samp_block = samp_3d[b]
            lp_block = lp_3d[b] 

            uniq_cands = torch.unique(cand_block, dim=0)
            uniq_samps, inv = torch.unique(samp_block, dim=0, return_inverse=True)

            cand_lens = lens_till_eoc(uniq_cands, eoc_id)
            samp_lens = lens_till_eoc(uniq_samps, eoc_id)

            cand_lists = [uniq_cands[i, : int(cand_lens[i])].tolist() for i in range(uniq_cands.size(0))]
            samp_lists = [uniq_samps[j, : int(samp_lens[j])].tolist() for j in range(uniq_samps.size(0))]

            n_samp_u = uniq_samps.size(0)
            N = inv.numel()
            pos = torch.arange(N, dtype=torch.long)

            first_pos = torch.full((n_samp_u,), N, dtype=torch.long)
            first_pos.scatter_reduce_(0, inv, pos, reduce="amin", include_self=True)

            uniq_lp = lp_block[first_pos]

            Ns_u, L = uniq_lp.shape
            ar = torch.arange(L).unsqueeze(0) 
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

            rows = [np.asarray(dl_distance_seqs(c, samp_lists), dtype=np.float32) for c in cand_lists]
            D = np.stack(rows, axis=0) 

            D = D * w[None, :]
            avg = D.sum(axis=1) 

            best_idx = int(avg.argmin())
            pred_list.append(cand_lists[best_idx])
            pred_lens.append(int(cand_lens[best_idx]))

        pred_len = torch.tensor(pred_lens, dtype=torch.long, device=device)

        return pred_list, pred_len

    @torch.no_grad()
    def prob_rank(
        self,
        prefixes, 
        n_sample,
        eoc_id = 3):
        """
        Re-ranking method that randomly samples suffixes and ranks them by likelihood
        """
        
        device = prefixes.device
        B = prefixes.size(0)
        T = self.suffix_len
        eoc_id = int(eoc_id)
        eoc_tensor = torch.tensor(eoc_id, device=device, dtype=torch.long)

        prefixes = prefixes.clone()

        def generate_samples_with_logprob(n: int):

            n = int(n)

            predictions = torch.empty((B * n, T), dtype=torch.long, device=device)

            alive = torch.ones((B * n,), dtype=torch.bool, device=device)

            seq_logprob = torch.zeros((B * n,), dtype=torch.float32, device=device)

            ctx = prefixes.to(torch.long).repeat_interleave(n, dim=0)

            for t in range(T):
                probs = self.next_probs(ctx)
                log_probs = torch.log(probs.clamp_min(1e-12))  

                sampled_next_act = torch.multinomial(probs, 1).squeeze(1) 

                next_act = torch.where(alive, sampled_next_act, eoc_tensor)

                predictions[:, t] = next_act

                step_logprob = log_probs.gather(1, sampled_next_act.unsqueeze(1)).squeeze(1)
                seq_logprob = seq_logprob + torch.where(
                    alive,
                    step_logprob,
                    torch.zeros_like(step_logprob)
                    )

                alive = alive & (next_act != eoc_id)

                ctx = torch.stack([ctx[:, 1], next_act], dim=1) 

                if not alive.any():
                    if t + 1 < T:
                        predictions[:, t + 1 :] = eoc_id
                    break

            return predictions, seq_logprob

        samples, sample_logprob = generate_samples_with_logprob(n_sample)
        
        lens_sample = lens_till_eoc(samples, eoc_id)

        samp_3d = samples.view(B, n_sample, T)
        lp_2d = sample_logprob.view(B, n_sample)
        ls_2d = lens_sample.view(B, n_sample)

        best_idx = lp_2d.argmax(dim=1)
        batch_idx = torch.arange(B)
        best_lens = ls_2d[batch_idx, best_idx]  
        best_seqs = samp_3d[batch_idx, best_idx, :]
        pred_list = [best_seqs[b, :best_lens[b]].tolist() for b in range(B)]

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
