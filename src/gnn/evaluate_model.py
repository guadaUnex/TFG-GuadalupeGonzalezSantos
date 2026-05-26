import sys
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from model import HybridModel
from dataset import SocNavHeteroDataset, collate
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
gnn_output = checkpoint['gnn_output']
rnn_hidden_size = checkpoint['rnn_hidden_size']
gnn_hidden_size = checkpoint['gnn_hidden_size']
num_edges = checkpoint['num_edges']
gnn_heads = checkpoint['gnn_heads']
gnn_concat = checkpoint['gnn_concat']
gnn_metadata = checkpoint['gnn_metadata']

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


model = HybridModel(num_layers, gnn_output, rnn_hidden_size, gnn_hidden_size, RNN_TYPE, num_edges,gnn_heads, gnn_concat, 
                    gnn_metadata, linear_layers = LINEAR_LAYERS, rnn_activation = ACTIVATION, context_vars = CONTEXT_FEATURES)    
model = model.to(device)


model.load_state_dict(checkpoint['model_state_dict'], strict=True)

datafile = args.dataset
contextQ_file = args.context
data_path = args.dataroot

dataset = SocNavHeteroDataset(datafile, data_path, contextQ_file, timestamp_threshold = FRAME_THRESHOLD)
loader = DataLoader(dataset,batch_size=32, shuffle=False, collate_fn=collate)

mse_function = torch.nn.MSELoss().to(device)
mae_function = torch.nn.L1Loss().to(device)

model.eval()
loss_mse = 0
loss_mae = 0
with torch.no_grad():
    for val_data in loader:
        traj_batch, label_batch, seq_length = val_data
        label_batch = label_batch.to(device)
        traj_batch = traj_batch.to(device)
        seq_length = seq_length.to(device)

        # Get predictions for the whole batch
        predictions = model(traj_batch, seq_length).squeeze()
        # print(predictions, label_batch)
        loss_mse += mse_function(predictions, label_batch).item() * len(label_batch)
        loss_mae += mae_function(predictions, label_batch).item() * len(label_batch)
    
loss_mse /= len(dataset)
loss_mae /= len(dataset)

print("MSE", loss_mse)
print("MAE", loss_mae)
