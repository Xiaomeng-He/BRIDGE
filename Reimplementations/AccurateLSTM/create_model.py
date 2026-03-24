import random
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset

class AccurateLSTM(nn.Module):

    def __init__(
        self,
        num_activities,
        num_roles,
        activity_emb_dim,
        role_emb_dim,
        activity_embedding_weights,
        role_embedding_weights,
        hidden_size,
        num_layers, 
        dropout):
        super().__init__()

        # Activity embedding
        self.activity_embedding = nn.Embedding(num_activities, activity_emb_dim)

        # Role embedding
        self.role_embedding = nn.Embedding(num_roles, role_emb_dim)

        # Load pretrained weights
        if activity_embedding_weights.shape != (num_activities, activity_emb_dim):
            raise ValueError(
                "activity_embedding_weights has wrong shape. "
                f"Expected {(num_activities, activity_emb_dim)}, "
                f"got {tuple(activity_embedding_weights.shape)}"
            )
        with torch.no_grad():
            self.activity_embedding.weight.copy_(activity_embedding_weights)

        if role_embedding_weights.shape != (num_roles, role_emb_dim):
            raise ValueError(
                "role_embedding_weights has wrong shape. "
                f"Expected {(num_roles, role_emb_dim)}, "
                f"got {tuple(role_embedding_weights.shape)}"
            )
        with torch.no_grad():
            self.role_embedding.weight.copy_(role_embedding_weights)

        self.activity_embedding.weight.requires_grad = False
        self.role_embedding.weight.requires_grad = False

        # Categorical branch (activity + role)
        self.cat_lstm = nn.LSTM(
            input_size=activity_emb_dim + role_emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.cat_bn = nn.BatchNorm1d(hidden_size)

        self.act_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.role_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Time branch
        self.time_lstm_1 = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.time_bn = nn.BatchNorm1d(hidden_size)

        self.time_lstm_2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output layers
        self.act_head = nn.Linear(hidden_size, num_activities)
        self.role_head = nn.Linear(hidden_size, num_roles)
        self.time_head = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(
        self,
        activity_prefix,
        role_prefix,
        time_prefix):

        # --- Shared categorical LSTM

        # Embeddings
        ac_emb = self.activity_embedding(activity_prefix)  # shape: (batch_size, prefix_len, act_embed)
        rl_emb = self.role_embedding(role_prefix) # shape: (batch_size, prefix_len, role_embed)

        # Concatenate categorical embeddings
        merged_cat = torch.cat([ac_emb, rl_emb], dim=-1) # shape: (batch_size, prefix_len, act_embed + role_embed)

        # Shared categorical LSTM
        cat_outputs, _ = self.cat_lstm(merged_cat) # shape: (batch_size, prefix_len, hidden_size)

        # BatchNorm over feature dimension
        cat_outputs_bn = self.cat_bn(cat_outputs.transpose(1, 2)).transpose(1, 2)
        cat_outputs_bn = self.dropout(cat_outputs_bn)

        # Activity-specific LSTM
        act_outputs, _ = self.act_lstm(cat_outputs_bn) # shape: (batch_size, prefix_len, hidden_size)
        act_output = act_outputs[:, -1, :] # shape: (batch_size, hidden_size)
        act_output = self.dropout(act_output)
        act_logits = self.act_head(act_output) # shape: (batch_size, num_activities)

        # Role-specific LSTM
        role_outputs, _ = self.role_lstm(cat_outputs_bn)
        role_output = role_outputs[:, -1, :]
        role_output = self.dropout(role_output)
        role_logits = self.role_head(role_output) # shape: (batch_size, num_roles)

        # --- Time LSTM --- 

        if time_prefix.dim() == 2:
            time_prefix = time_prefix.unsqueeze(-1)  # shape: (batch_size, prefix_len, 1)

        # First time LSTM
        time_seq, _ = self.time_lstm_1(time_prefix)   
        time_seq_bn = self.time_bn(time_seq.transpose(1, 2)).transpose(1, 2)
        time_seq_bn = self.dropout(time_seq_bn)

        # Second time LSTM
        time_outputs, _ = self.time_lstm_2(time_seq_bn)    # [B, L, H]
        time_output = time_outputs[:, -1, :]       # [B, H]
        time_output = self.dropout(time_output)
        time_pred = self.time_head(time_output) # shape: (batch_size, 1)

        return act_logits, role_logits, time_pred

class ActivityRoleEmbeddingModel(nn.Module):

    def __init__(self, num_activities, num_roles, embedding_dim):
        super().__init__()
        self.activity_embedding = nn.Embedding(num_activities, embedding_dim, 0)
        self.role_embedding = nn.Embedding(num_roles, embedding_dim, 0)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.activity_embedding.weight)
        nn.init.xavier_uniform_(self.role_embedding.weight)

    def forward(self, activity_ids, role_ids: torch.Tensor) -> torch.Tensor:
        ac = self.activity_embedding(activity_ids.long())  
        rl = self.role_embedding(role_ids.long()) 

        ac = F.normalize(ac, p=2, dim=-1)
        rl = F.normalize(rl, p=2, dim=-1)

        sim = torch.sum(ac * rl, dim=-1) 
        return sim


class PairBatchDataset(IterableDataset):
    def __init__(
        self,
        pairs: List[Tuple[int, int]],
        valid_activities,
        valid_roles,
        num_activities: int,
        num_roles: int,
        n_positive: int = 50,
        negative_ratio: int = 1,
    ):
        super().__init__()
        self.pairs = pairs
        self.pairs_set = set(pairs)
        self.valid_activities = valid_activities
        self.valid_roles = valid_roles
        self.num_activities = num_activities
        self.num_roles = num_roles
        self.n_positive = n_positive
        self.negative_ratio = negative_ratio
        self.batch_size = int(n_positive * (1 + negative_ratio))

    def __iter__(self):

        while True:
            batch_ac = torch.zeros(self.batch_size, dtype=torch.long)
            batch_rl = torch.zeros(self.batch_size, dtype=torch.long)
            batch_y = torch.zeros(self.batch_size, dtype=torch.float32)

            # positive examples
            sampled = random.sample(self.pairs, self.n_positive)
            idx = 0
            for idx, (ac, rl) in enumerate(sampled):
                batch_ac[idx] = ac
                batch_rl[idx] = rl
                batch_y[idx] = 1.0
            idx += 1

            # negative examples
            while idx < self.batch_size:
                random_ac = random.choice(self.valid_activities)
                random_rl = random.choice(self.valid_roles)
                if (random_ac, random_rl) not in self.pairs_set:
                    batch_ac[idx] = random_ac
                    batch_rl[idx] = random_rl
                    batch_y[idx] = 0.0
                    idx += 1

            perm = torch.randperm(self.batch_size)
            yield batch_ac[perm], batch_rl[perm], batch_y[perm]
