import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu
from torch_geometric.nn import GATConv, Linear, to_hetero

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)

        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) 
        x = leaky_relu(x)
        x = self.conv2(x, edge_index) 
        return x


class GNNModel(nn.Module):
    def __init__(self, gnn_hidden_channels, gnn_heads, gnn_output, gnn_concat):
        super(GNNModel,self).__init__()
        self.layers_gat = nn.ModuleList()
        # self.layers_lin = nn.ModuleList()

        self.layers_gat.append((GATConv((-1, -1), gnn_hidden_channels[0], gnn_heads[0], gnn_concat, add_self_loops=False)))
        # self.layers_lin.append(Linear(-1, gnn_hidden_channels[0]))

        for idx in range(len(gnn_hidden_channels) - 1):
            output_dim = gnn_hidden_channels[idx + 1]
            heads = gnn_heads[idx + 1]

            self.layers_gat.append((GATConv((-1, -1), output_dim, heads, gnn_concat, add_self_loops=False)))
            # self.layers_lin.append(Linear(-1, output_dim))

        self.layers_gat.append((GATConv((-1, -1), gnn_output, heads=1, concat=False, add_self_loops=False)))
        # self.layers_lin.append(Linear(-1, gnn_output))


    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers_gat):
            x = layer(x, edge_index)#+self.layers_lin[i](x)
            if i < len(self.layers_gat) - 1:
                x = leaky_relu(x, negative_slope=0.1)
        return x

class HybridModel(nn.Module):
    def __init__(self, num_layers, gnn_output, rnn_hidden_channels, gnn_hidden_channels, rnn_type, num_edges, gnn_heads, gnn_concat, gnn_metadata, 
                 linear_layers=[], rnn_activation = 'linear', context_vars = 0, rnn_dropout = 0.0):
        super(HybridModel,self).__init__()

        self.gnn_output = gnn_output
        self.num_edges = num_edges
        self.num_layers = num_layers
        self.context_vars = context_vars

        # self.defineGnnBlock(gnn_hidden_channels, gnn_heads, gnn_concat)
        # self.gnn_block = GAT(gnn_hidden_channels[0], gnn_output)
        self.gnn_block = GNNModel(gnn_hidden_channels, gnn_heads, gnn_output, gnn_concat)


        print("metadata")
        print(gnn_metadata)
        self.gnn_block = to_hetero(self.gnn_block, gnn_metadata, aggr='sum')

        print(self.gnn_block)
        exit()

        self.defineRnnBlock(rnn_type, rnn_hidden_channels, linear_layers, rnn_activation, rnn_dropout)

    def defineGnnBlock(self, gnn_hidden_channels, gnn_heads, gnn_concat):
        layers = []

        layers.append((GATConv((-1, -1), gnn_hidden_channels[0], gnn_heads[0], gnn_concat, add_self_loops=False), 'x, edge_index -> x'))
        layers.append(nn.LeakyReLU(inplace=True))

        for idx in range(len(gnn_hidden_channels) - 1):
            input_dim = gnn_hidden_channels[idx] * (gnn_heads[idx] if gnn_concat else 1)
            output_dim = gnn_hidden_channels[idx + 1]
            heads = gnn_heads[idx + 1]
            layers.append((GATConv((-1, -1), output_dim, heads, gnn_concat, add_self_loops=False),
                           'x, edge_index -> x'))
            layers.append(nn.LeakyReLU(negative_slope=0.1))

        input_dim = gnn_hidden_channels[-1] * (gnn_heads[-1] if gnn_concat else 1)
        layers.append((GATConv((-1, -1), self.gnn_output, heads=1, concat=False, add_self_loops=False), 
                       'x, edge_index -> x'))

        # self.gnn_block = Sequential('x, edge_index', layers)


    def defineRnnBlock(self, rnn_type, rnn_hidden_channels, linear_layers, rnn_activation, rnn_dropout):
        if rnn_type == "GRU":
            self.rnn_layer = nn.GRU(self.gnn_output, rnn_hidden_channels, self.num_layers, batch_first=True, dropout=rnn_dropout)
        elif rnn_type == "LSTM":
            self.rnn_layer = nn.LSTM(self.gnn_output, rnn_hidden_channels, self.num_layers,
                                batch_first=True, dropout = rnn_dropout)
            
        self.fc_layers = nn.ModuleList()
        linear_size = rnn_hidden_channels+self.context_vars
        for l in linear_layers:
            self.fc_layers.append(nn.Linear(linear_size, l))
            linear_size = l
            self.fc_layers.append(nn.LeakyReLU())

        self.fc_layers.append(nn.Linear(linear_size,1))
        if len(linear_layers)>0:
            self.mlp = nn.Sequential(*self.fc_layers)

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
        # print(slengths)

        # for k, t in batch_data.x_dict.items():
        #     print(k, t.shape)
        # for k, t in batch_data.edge_index_dict.items():
        #     print(k, t.shape, t.dtype)
        #     print(t)

        # print(batch_data.x_dict['scenario'])        
        gnn_output = self.gnn_block(batch_data.x_dict, batch_data.edge_index_dict)

        # for k, t in gnn_output.items():
        #     print(k, t.shape)
        # print(gnn_output)

        scenarios = gnn_output['scenario']

        # print('input shape', batch_data.x_dict.shape)

        num_trajectories = len(slengths)
        max_len = int(torch.max(slengths).item())
        features_dim = scenarios.size(-1)

        # print('num_trajectories', num_trajectories)
        # print('max_len', max_len)
        # print('features_dim', features_dim)
        # print('scenarios shape', scenarios.shape)

        # batch_size x max_len x features_dim
        x_seq = torch.zeros(num_trajectories, max_len, features_dim, device=scenarios.device)

        current_idx = 0
        first_graph_idx = torch.zeros(num_trajectories, device=scenarios.device, dtype=torch.long)
        
        for i, length in enumerate(slengths):
            first_graph_idx[i] = current_idx
            x_seq[i, :length, :] = scenarios[current_idx : current_idx + length, :]
            current_idx += length


        rnn_output, _ = self.rnn_layer(x_seq)

        # print('rnn_output shape', rnn_output.shape)

        out = rnn_output[torch.arange(rnn_output.shape[0]), slengths - 1]

        # print('output shape', out.shape)
        if self.context_vars > 0:
            batch_context = batch_data.x_dict['scenario'][first_graph_idx, -self.context_vars:]
            out = torch.concat((out, batch_context), axis=1)
            # out = torch.concat((out, x_seq[:, 0, -self.context_vars:]), axis=1)

        for layer in self.fc_layers:
            out = layer(out)

        if self.activation is not None:
            out = self.activation(out)
            
        if hasattr(self, 'correct_output') and self.correct_output:
            out = (out + 1.) / 2.
        
        return out
