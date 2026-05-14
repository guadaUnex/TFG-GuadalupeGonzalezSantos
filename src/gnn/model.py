import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, Sequential, to_hetero

class HybridModel(nn.Module):
    def __init__(self, gnn_input, gnn_output, rnn_hidden_channels, gnn_hidden_channels, num_layers, rnn_type, num_edges, gnn_heads, gnn_concat, gnn_metadata, 
                 linear_layers=[], rnn_activation = 'linear', context_vars = 0, rnn_dropout = 0.0):
        super(HybridModel,self).__init__()

        self.gnn_output = gnn_output
        self.num_edges = num_edges
        self.num_layers = num_layers
        self.context_vars = context_vars

        self.defineGnnBlock(gnn_input, gnn_hidden_channels, gnn_heads, gnn_concat)

        self.gnn_block = to_hetero(self.gnn_block, gnn_metadata, aggr='sum')

        self.defineRnnBlock(rnn_type, rnn_hidden_channels, linear_layers, rnn_activation, rnn_dropout)

    def defineGnnBlock(self, gnn_input, gnn_hidden_channels, gnn_heads, gnn_concat):
        layers = []

        layers.append((GATConv(gnn_input, gnn_hidden_channels, gnn_heads[0], gnn_concat), 'x, edge_index -> x'))
        layers.append(nn.LeakyReLU(inplace=True))

        for idx in range(len(gnn_hidden_channels) - 1):
            input_dim = gnn_hidden_channels[idx] * (gnn_heads[idx] if gnn_concat else 1)
            output_dim = gnn_hidden_channels[idx + 1]
            heads = gnn_heads[idx + 1]
            layers.append((GATConv(input_dim, output_dim, heads, gnn_concat),
                           'x, edge_index -> x'))
            layers.append(nn.LeakyReLU(negative_slope=0.1))

        input_dim = gnn_hidden_channels[-1] * (gnn_heads[-1] if gnn_concat else 1)
        layers.append((GATConv(input_dim, self.gnn_output, heads=1, concat=False), 
                       'x, edge_index -> x'))

        self.gnn_block = Sequential('x, edge_index', layers)


    def defineRnnBlock(self, rnn_type, rnn_hidden_channels, linear_layers, rnn_activation, rnn_dropout):
        if rnn_type == "GRU":
            self.rnn_layer = nn.GRU(self.gnn_output, rnn_hidden_channels, self.num_layers, batch_first=True, dropout=rnn_dropout)
        elif rnn_type == "LSTM":
            self.rnn_layer = nn.LSTM(self.gnn_output, rnn_hidden_channels, self.num_layers,
                                batch_first=True, dropout = rnn_dropout)
            
        self.fc_layers = []
        linear_size = rnn_hidden_channels+self.context_vars
        for l in linear_layers:
            self.fc_layers.append(nn.Linear(linear_size, l))
            linear_size = l
            self.fc_layers.append(nn.LeakyReLU())

        self.fc_layers.append(nn.Linear(linear_size,1))
        self.fc = self.fc_layers[-1]
        if len(linear_layers)>0:
            self.mlp = Sequential(*self.fc_layers)

        if rnn_activation == 'sigmoid':
            self.correct_output = False
            self.activation = nn.Sigmoid()
        elif rnn_activation == 'tanh':
            self.correct_output = True
            self.activation = nn.Tanh()
        else:
            self.correct_output = False
            self.activation = None
        self.context_vars = self.context_vars


    def forward(self, batch_data, slengths):
        gnn_output = self.gnn_block(batch_data.x_dict, batch_data.edge_index_dict)

        scenarios = gnn_output['scenario']

        num_trajectories = len(slengths)
        max_sequence_length = scenarios.shape[0] // num_trajectories

        x_seq = scenarios.view(num_trajectories, max_sequence_length, -1)

        rnn_output, _ = self.rnn_layer(x_seq)

        out = rnn_output[torch.arange(rnn_output.shape[0]), slengths - 1]

        if self.context_vars > 0:
            out = torch.concat((out, x_seq[:, 0, -self.context_vars:]), axis=1)

        for layer in self.fc_layers:
            out = layer(out)

        if self.activation is not None:
            out = self.activation(out)
            
        if hasattr(self, 'correct_output') and self.correct_output:
            out = (out + 1.) / 2.
        
        return out
