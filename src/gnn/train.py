import argparse
import os
import torch
import yaml

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset import SocNavHeteroDataset, collate
from model import HybridModel

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML = os.path.join(CURRENT_PATH, "train.yaml")

parser = argparse.ArgumentParser(description=help)
parser.add_argument('--task', type=str, default=DEFAULT_YAML, help="yaml file path")
args = parser.parse_args()

config_data = yaml.load(open(args.task, "r"), Loader=yaml.Loader)

DATA_PATH = config_data["DATA_PATH"]
CONTEXT_FILE = config_data["CONTEXT_FILE"]

TRAIN_FILE = config_data["TRAIN_FILE"]
DEV_FILE = config_data["DEV_FILE"]
TEST_FILE = config_data["TEST_FILE"]

ACTIVATION = config_data["ACTIVATION"]
ADD_CONTEXT_TO_OUTPUT = config_data['ADD_CONTEXT_TO_OUTPUT']
DROPOUT = config_data["DROPOUT"]
HIDDEN_SIZE = config_data["HIDDEN_SIZE"]
LINEAR_LAYERS = config_data["LINEAR_LAYERS"]
LOSS = config_data["LOSS"]
LR = config_data["LR"]
NUM_LAYERS = config_data["NUM_LAYERS"]
RNN_TYPE = config_data["RNN_TYPE"]
TIMESTAMP_THRESHOLD = config_data["TIMESTAMP_THRESHOLD"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

SAVE_NAME = '_'.join(['ALL_rm_rf_try', RNN_TYPE, LOSS, str(NUM_LAYERS), str(LR), str(HIDDEN_SIZE)])

def train_model(rnn_data, gnn_data, num_layers, 
                batch_size=32, num_epochs=1000, patience =50, learning_rate=0.00005, 
                checkpoint_dir='checkpoints'):
    # Create checkpoint directory if it doesn't exist
    model_name = SAVE_NAME+'.pytorch'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, model_name)

    train_dataset = SocNavHeteroDataset(data_list_file = TRAIN_FILE, data_path = DATA_PATH, context_path = CONTEXT_FILE, timestamp_threshold = TIMESTAMP_THRESHOLD)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)

    # all_features = train_dataset.get_all_features()
    # print("Num Features:", len(all_features))
    # input_size = len(all_features)

    if ADD_CONTEXT_TO_OUTPUT:
        context_features = len(train_dataset.get_context_features())
    else:
        context_features = 0

    # ----------------------------------------

    val_dataset = SocNavHeteroDataset(data_list_file = DEV_FILE, data_path = DATA_PATH, context_path = CONTEXT_FILE, timestamp_threshold = TIMESTAMP_THRESHOLD)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)

    model = HybridModel(num_layers, gnn_input = gnn_data['input'], gnn_output = gnn_data['output'], rnn_hidden_channels= rnn_data['hidden_channels'], 
                        gnn_hidden_channels= gnn_data['hidden_channels'], rnn_type = rnn_data['type'], num_edges = gnn_data['num_edges'],
                        gnn_heads = gnn_data['heads'], gnn_concat = gnn_data['concant'], gnn_metadata= gnn_data['metadata'], linear_layers = LINEAR_LAYERS, 
                        rnn_activation = ACTIVATION, context_vars = context_features, rnn_dropout = DROPOUT)
    model = model.to(device)
    if LOSS == "mse":
        criterion = nn.MSELoss().to(device)
    elif LOSS == "bce":
        criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    min_val_loss = float('inf')
    best_epoch = 1
    patience_counter = patience  # Using your global Patience variable
    train_losses, val_losses = [], []

    print(f"Starting training for {num_epochs} epochs (or until early stopping)")
    print(f"Model will be saved to {checkpoint_path} when validation loss improves")

    for epoch in range(num_epochs):
        #Training Loop
        model.train()
        epoch_loss = 0

        for trajectories, labels, slengths in train_dataloader:
            trajectories = trajectories.to(device)
            labels = labels.to(device)
            slengths = slengths.to(device)
            outputs = model(trajectories, slengths)

            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*len(labels)

        train_loss = epoch_loss / len(train_dataset)
        train_losses.append(train_loss)

        # Validation Loop

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for trajectories, labels, slengths in val_dataloader:
                trajectories = trajectories.to(device)
                labels = labels.to(device)
                slengths = slengths.to(device)
                outputs = model(trajectories, slengths)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()*len(labels)
        full_preds = []