import sys
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from modelHomo import HybridModel
from datasetHomo import SocNavHomoDataset, collate
import argparse

parser = argparse.ArgumentParser(prog='evaluate_model',
                    description='Evaluates the specified model on a given dataset.')
parser.add_argument('--model', type=str, nargs="?", required = True, help="Model trained for rating prediction.")
parser.add_argument('--context', type=str, nargs="?", required = True,  help="Context file for the generation of the context variables.")
parser.add_argument('--dataset', type=str, nargs="?", default='.', help="Dataset file used for testing the model (TXT)")
parser.add_argument('--dataroot', type=str, nargs="?", default='.', help="Data root directory.")

args = parser.parse_args()


checkpoint_path = args.model
print(checkpoint_path)
model_name = checkpoint_path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

num_layers = checkpoint['num_layers']
gnn_input = checkpoint['gnn_input_size']
gnn_output = checkpoint['gnn_output']
rnn_hidden_size = checkpoint['rnn_hidden_size']
gnn_hidden_size = checkpoint['gnn_hidden_size']
gnn_heads = checkpoint['gnn_heads']
gnn_concat = checkpoint['gnn_concat']

if 'use_new_context' in checkpoint.keys():
    USE_NEW_CONTEXT = checkpoint['use_new_context']
else:
    USE_NEW_CONTEXT = False
if 'rnn_type' in checkpoint.keys():
    RNN_TYPE = checkpoint['rnn_type']
else:
    RNN_TYPE = 'GRU'
if 'frame_threshold' in checkpoint.keys():
    FRAME_THRESHOLD = checkpoint['frame_threshold']
else:
    FRAME_THRESHOLD = 0.1

if 'linear_layers' in checkpoint.keys():
    LINEAR_LAYERS = checkpoint['linear_layers']
else:
    LINEAR_LAYERS = []

if 'activation' in checkpoint.keys():
    ACTIVATION = checkpoint['activation']
else:
    ACTIVATION = 'linear'

if 'context_features' in checkpoint.keys():
    CONTEXT_FEATURES = checkpoint['context_features']
else:
    CONTEXT_FEATURES = 0

if 'metric_features' in checkpoint.keys():
    METRICS_FEATURES = checkpoint['metric_features']
else:
    METRICS_FEATURES = 0

if 'only_gnn' in checkpoint.keys():
    ONLY_GNN = checkpoint['only_gnn']
else:
    ONLY_GNN = False

if 'only_metrics' in checkpoint.keys():
    ONLY_METRICS = checkpoint['only_metrics']
else:
    ONLY_METRICS = False    

# print('only_gnn', ONLY_GNN)    

# print('only_metrics', ONLY_METRICS)    
# print(checkpoint['model_state_dict'])

model = HybridModel(num_layers, gnn_input=gnn_input, gnn_output = gnn_output, rnn_hidden_channels= rnn_hidden_size, 
                    gnn_hidden_channels= gnn_hidden_size, rnn_type = RNN_TYPE, 
                    gnn_heads = gnn_heads, gnn_concat = gnn_concat, linear_layers = LINEAR_LAYERS, 
                    rnn_activation = ACTIVATION, context_vars = CONTEXT_FEATURES, metrics_vars = METRICS_FEATURES,
                    only_gnn = ONLY_GNN, only_metrics = ONLY_METRICS)



model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model = model.to(device)

datafile = args.dataset
contextQ_file = args.context
data_path = args.dataroot

dataset = SocNavHomoDataset(datafile, data_path, contextQ_file, timestamp_threshold = FRAME_THRESHOLD)
loader = DataLoader(dataset,batch_size=32, shuffle=False, collate_fn=collate)

mse_function = torch.nn.MSELoss().to(device)
mae_function = torch.nn.L1Loss().to(device)

model.eval()
loss_mse = 0
loss_mae = 0
with torch.no_grad():
    for val_data in loader:
        traj_batch, metrics_batch, label_batch, seq_length = val_data
        label_batch = label_batch.to(device)
        traj_batch = traj_batch.to(device)
        metrics_batch = metrics_batch.to(device)
        seq_length = seq_length.to(device)

        # Get predictions for the whole batch
        predictions = model(traj_batch, metrics_batch, seq_length)
        print(predictions, label_batch)
        loss_mse += mse_function(predictions, label_batch).item() * label_batch.shape[0]
        loss_mae += mae_function(predictions, label_batch).item() * label_batch.shape[0]
    
loss_mse /= len(dataset)
loss_mae /= len(dataset)

print("MSE", loss_mse)
print("MAE", loss_mae)
