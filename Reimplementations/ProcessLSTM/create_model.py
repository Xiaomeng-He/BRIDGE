import torch
import torch.nn as nn
import torch.nn.functional as F

class ProcessLSTM(nn.Module):

    def __init__(
        self,
        num_act,
        hidden_size,
        num_layers, 
        dropout):
        super().__init__()

        self.num_act = num_act

        self.shared_lstm = nn.LSTM(
            input_size=num_act+3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.act_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.time_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.shared_bn = nn.BatchNorm1d(hidden_size)
        self.act_bn = nn.BatchNorm1d(hidden_size)
        self.time_bn = nn.BatchNorm1d(hidden_size)

        self.act_head = nn.Linear(hidden_size, num_act)
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
        act_prefix, # shape: (batch_size, prefix_len)
        time_prefix): # shape: (batch_size, prefix_len, 3)

        # --- shared LSTM

        # One-hot encoding
        oh_act_prefix = F.one_hot(act_prefix, num_classes=self.num_act).float()

        # Concatenate inputs
        x = torch.cat([oh_act_prefix, time_prefix], dim=-1) # shape: (batch_size, prefix_len, act_embed + role_embed)

        # Shared LSTM
        shared_outputs, _ = self.shared_lstm(x) # shape: (batch_size, prefix_len, hidden_size)
        shared_outputs_bn = self.shared_bn(shared_outputs.transpose(1, 2)).transpose(1, 2)
        shared_outputs_bn = self.dropout(shared_outputs_bn)

        # Activity-specific LSTM
        act_outputs, _ = self.act_lstm(shared_outputs_bn) # shape: (batch_size, prefix_len, hidden_size)
        act_output = act_outputs[:, -1, :] # shape: (batch_size, hidden_size)
        act_output = self.act_bn(act_output)
        act_output = self.dropout(act_output)
        act_logits = self.act_head(act_output) # shape: (batch_size, num_act)

        # Time-specific LSTM
        time_outputs, _ = self.time_lstm(shared_outputs_bn)    # [B, L, H]
        time_output = time_outputs[:, -1, :]       # [B, H]
        time_output = self.time_bn(time_output)
        time_output = self.dropout(time_output)
        time_pred = self.time_head(time_output) # shape: (batch_size, 1)

        return act_logits, time_pred
