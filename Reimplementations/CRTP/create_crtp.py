import torch
import torch.nn as nn
import torch.nn.init as init

class CRTP_LSTM(nn.Module):
    def __init__(self, 
                 num_activities, 
                 act_embed, 
                 input_dims, 
                 emb_dims,
                 num_nume_feature,
                 d_model = 80,
                 dropout = 0.2, 
                 num_shared_LSTMlayers = 1,
                 num_dedicated_LSTMlayers = 1,
                 ):
        
        super(CRTP_LSTM, self).__init__()
        self.num_activities = num_activities

        self.d_model = d_model

        self.dropout = dropout

        self.num_shared_LSTMlayers = num_shared_LSTMlayers
        self.num_dedicated_LSTMlayers = num_dedicated_LSTMlayers

        self.act_embedding = nn.Embedding(num_activities, act_embed, 0)
        self.embeddings = nn.ModuleList([
            nn.Embedding(inp, dim, 0)
            for inp, dim in zip(input_dims, emb_dims)
        ])

        self.input_size = act_embed + sum(emb_dims) + num_nume_feature

        self.dropout = nn.Dropout(self.dropout)

        assert d_model % 2 == 0, "d_model must be an even number"

        self.hidden_size = self.d_model // 2 
        self.lstm_shared = nn.LSTM(input_size=self.input_size, 
                                   hidden_size=self.hidden_size, 
                                   num_layers=self.num_shared_LSTMlayers, 
                                   batch_first=True, 
                                   bidirectional=True)

        self.bn_shared = nn.BatchNorm1d(self.d_model)

        self.lstm_act = nn.LSTM(input_size=d_model, 
                                hidden_size=self.hidden_size, 
                                num_layers=self.num_dedicated_LSTMlayers, 
                                batch_first=True, 
                                bidirectional=True)

        self.bn_act = nn.BatchNorm1d(self.d_model)

        self.fc_out_act = nn.Linear(self.d_model, self.num_activities)

        self.lstm_rrt = nn.LSTM(input_size=d_model, 
                                hidden_size=self.hidden_size, 
                                num_layers=self.num_dedicated_LSTMlayers, 
                                batch_first=True, 
                                bidirectional=True)

        self.bn_rrt = nn.BatchNorm1d(self.d_model)

        self.fc_out_rrt = nn.Linear(self.d_model, 1)  

        self.reset_parameters_lstm()

    def reset_parameters_lstm(self):

        for name, param in self.lstm_shared.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)

        for name, param in self.lstm_act.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)

        for name, param in self.lstm_rrt.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)

    def forward(self, 
                cate_prefix, 
                nume_prefix):

        act_embed = self.act_embedding(cate_prefix[:, :, 0])

        if cate_prefix.shape[-1] > 1: 
            embedded = [act_embed]
            for i, emb in enumerate(self.embeddings):
                feature_i = cate_prefix[:, :, i+1]   
                emb_i = emb(feature_i)  
                embedded.append(emb_i)
            embedded = torch.cat(embedded, dim=-1)
        else:
            embedded = act_embed

        x = torch.cat([embedded, nume_prefix], dim=-1)

        x = self.dropout(x)

        # -- Shared LSTM -- 
        shared_out, _ = self.lstm_shared(x) 
        shared_out = shared_out.permute(0, 2, 1) 
        shared_out = self.bn_shared(shared_out)
        shared_out = shared_out.permute(0, 2, 1) 

        # -- Activity LSTM --
        act_outputs, _ = self.lstm_act(shared_out) 
        act_outputs = act_outputs.permute(0, 2, 1) 
        act_outputs = self.bn_act(act_outputs) 
        act_outputs = act_outputs.permute(0, 2, 1) 
        act_probs = self.fc_out_act(act_outputs)

        # -- Remaining RunTime (rrt) LSTM --
        rrt_outputs, _ = self.lstm_rrt(shared_out)
        rrt_outputs = rrt_outputs.permute(0, 2, 1) 
        rrt_outputs = self.bn_rrt(rrt_outputs) 
        rrt_outputs = rrt_outputs.permute(0, 2, 1) 
        rrt_pred = self.fc_out_rrt(rrt_outputs)

        return act_probs, rrt_pred
