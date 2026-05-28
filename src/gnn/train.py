import argparse
import os, io
import torch
import wandb
import yaml

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.data import DataLoader

from dataset import SocNavHeteroDataset, collate
from model import HybridModel
from utils import plot_qualitative_multiple

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML = os.path.join(CURRENT_PATH, "train.yaml")

parser = argparse.ArgumentParser(description="Pipeline de entrenamiento para SocNavHybridModel")
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
BATCH_SIZE = config_data["BATCH_SIZE"]
DROPOUT = config_data["DROPOUT"]
GNN_CONCAT = config_data["GNN_CONCAT"]
GNN_HEADS = config_data["GNN_HEADS"]
GNN_HIDDEN_SIZE = config_data["GNN_HIDDEN_SIZE"]
RNN_HIDDEN_SIZE = config_data["RNN_HIDDEN_SIZE"]
LINEAR_LAYERS = config_data["LINEAR_LAYERS"]
LOSS = config_data["LOSS"]
LR = config_data["LR"]
MAX_EPOCHS = config_data["MAX_EPOCHS"]
MAX_PATIENCE = config_data["MAX_PATIENCE"]
NUM_LAYERS = config_data["NUM_LAYERS"]
RNN_TYPE = config_data["RNN_TYPE"]
TIMESTAMP_THRESHOLD = config_data["TIMESTAMP_THRESHOLD"]

SAVE_PLOTS_LOCALLY = config_data["SAVE_PLOTS_LOCALLY"]
UPLOAD_TO_WANDB = config_data["UPLOAD_TO_WANDB"]

qual_loaders = {}
Q_TEST_PATH = config_data["Q_TEST_PATH"]
config_qtest = yaml.load(open(config_data['Q_TEST_FILE'], "r"), Loader=yaml.Loader)
Q_FILES = config_qtest["FILES"]
Q_NAMES = config_qtest["NAMES"]
COLORS = config_qtest["COLORS"]
CONTEXTS = config_qtest["CONTEXTS"]
ABBREVIATED_CONTEXTS = config_qtest["ABBREVIATED_CONTEXTS"]

RELOAD = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

SAVE_NAME = '_'.join(['GNN', RNN_TYPE, LOSS, str(NUM_LAYERS), str(LR)])

def train_model(rnn_data, gnn_data, num_layers, 
                batch_size=32, num_epochs=1000, patience =50, learning_rate=0.00005, 
                checkpoint_dir='checkpoints'):
    # Create checkpoint directory if it doesn't exist
    model_name = SAVE_NAME+'.pytorch'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, model_name)

    train_dataset = SocNavHeteroDataset(data_list_file = TRAIN_FILE, data_path = DATA_PATH, context_path = CONTEXT_FILE, timestamp_threshold = TIMESTAMP_THRESHOLD, reload = RELOAD)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=False)

    if ADD_CONTEXT_TO_OUTPUT:
        context_features = len(train_dataset.get_context_features())
    else:
        context_features = 0

    # print(train_dataset.labels)
    # ----------------------------------------
    # for trajectories, test_labels, slengths in train_dataloader:
    #     print("Muestra de etiquetas reales de tu dataset (primeros 10):", test_labels[:10].tolist())
    #     print("Suma de etiquetas en este batch:", test_labels.sum().item())
    #     print("Muestra de longitudes de secuencias (slengths):", slengths[:10].tolist())
        # break # Solo miramos el primer batch para no saturar la consola
    print("---------------------------------")


    val_dataset = SocNavHeteroDataset(data_list_file = DEV_FILE, data_path = DATA_PATH, context_path = CONTEXT_FILE, timestamp_threshold = TIMESTAMP_THRESHOLD, reload=RELOAD)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate, drop_last=True)

    model = HybridModel(num_layers, gnn_output = gnn_data['output'], rnn_hidden_channels= rnn_data['hidden_channels'], 
                        gnn_hidden_channels= gnn_data['hidden_channels'], rnn_type = rnn_data['type'], num_edges = gnn_data['num_edges'],
                        gnn_heads = gnn_data['heads'], gnn_concat = gnn_data['concat'], gnn_metadata= gnn_data['metadata'], linear_layers = LINEAR_LAYERS, 
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

        # full_preds = []
        # #Get Qualitative predictions and the plot
        # fig, axes = plt.subplots(2, 2, figsize=(15, 6*2))

        # axes = axes.reshape(1, -1)

        # # Flatten the axes array for easy iteration
        # axes_flat = axes.flatten()

        # # Hide any unused subplots
        # for i in range(4, len(axes_flat)):
        #     axes_flat[i].set_visible(False)
        # for i_d, speed in enumerate(list(qual_loaders.keys())):
        #     q_preds = {}
        #     q_list = qual_loaders[speed]
        #     # for file in q_list:
        #     # print(f"File : {q_list}")
        #     for i, q_tuple in enumerate(q_list):
        #         full_preds = []
        #         # print(f"Q_tuple {i}:{len(q_tuple)}")
        #         context, files, qual_loader, abbr_context = q_tuple
        #         # print(f"Context :{context}, files: {files}, qual_loader: {qual_loader}")
        #         with torch.no_grad():
        #             # print(f"======================\n{filename=}")
        #             for trajectories, _, slengths in qual_loader:
        #                 # Pass the whole batched graph sequence to the model at once
        #                 trajectories = trajectories.to(device)
        #                 slengths = slengths.to(device)
        #                 preds = model(trajectories, slengths)
        #                 full_preds += preds.tolist()
        #         # print(f"Predictions for Context {context}: \n {all_preds}")
        #         q_preds[abbr_context] = full_preds
        #     for idx, context in enumerate(q_preds.keys()):
        #         # axes_flat[idx] = plot_qualitative_ratings(q_preds[context], speed, context, COLORS[i_d], ax=axes_flat[idx])
        #         axes_flat[idx] = plot_qualitative_multiple(q_preds[context], speed, context, COLORS[i_d], ax=axes_flat[idx])


        # # model_name = f"{save_dir[:10]}_ep_{str(epoch)}"
        # plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
        # save_dir = os.path.join('plots', SAVE_NAME+'_q_test')
        # save_dir = save_dir if UPLOAD_TO_WANDB is False else os.path.join('plots',wandb.run.name)
        # os.makedirs(save_dir, exist_ok=True)
        # save_loc = os.path.join(save_dir, 'e_'+str(epoch+1))
        # if SAVE_PLOTS_LOCALLY:
        #     plt.savefig(save_loc)
        # if UPLOAD_TO_WANDB:
        #     q_plt_buf = io.BytesIO()
        #     plt.savefig(q_plt_buf, format='png')
        #     q_plt_buf.seek(0)
        #     q_plt_img = wandb.Image(Image.open(q_plt_buf))
        # plt.close()
        
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)

        # Checkpointing - save when validation loss improves
        if val_loss < min_val_loss:
            improvement = min_val_loss - val_loss
            min_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = patience  # Reset patience counter
            
            # Save the checkpoint with detailed information
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'train_loss': train_loss,
                'rnn_type': RNN_TYPE,
                'gnn_input_size': gnn_data['input'],
                'rnn_input_size': rnn_data['input'],
                'rnn_hidden_size': RNN_HIDDEN_SIZE,
                'num_layers': num_layers,
                'linear_layers': LINEAR_LAYERS,
                'frame_threshold': TIMESTAMP_THRESHOLD,
                'activation': ACTIVATION,
                'context_features': context_features,
                'gnn_output': gnn_data['output'],
                'rnn_hidden_size': rnn_data['hidden_channels'],
                'gnn_hidden_size': gnn_data['hidden_channels'],
                'num_edges': gnn_data['num_edges'],
                'gnn_heads': gnn_data['heads'],
                'gnn_concat': gnn_data['concat'],
                'gnn_metadata': gnn_data['metadata']
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            print(f"Checkpoint saved at epoch {epoch+1}")
            print(f"Val Loss: {val_loss:.6f} (improved by {improvement:.6f})")
            
            # Optionally save a timestamped version too if you want a history
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # history_path = os.path.join(checkpoint_dir, f'model_e{epoch+1}_{timestamp}.pth')
            # torch.save(checkpoint, history_path)
        else:
            patience_counter -= 1
            print(f"****Epoch {epoch+1}: Val Loss: {val_loss:.6f} (best: {min_val_loss:.6f}, patience: {patience_counter}/{patience})")
        
        if patience_counter == 0:
            print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {min_val_loss:.6f}")
            break

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
        # Log training loss, validation loss, and gradient statistics to WandB
        if UPLOAD_TO_WANDB is True:
            wandb.log({
                "training_loss": train_loss,
                "validation_loss": val_loss,
                "lr": LR,
                "Qualitative_test": q_plt_img,
                "min_val_loss": min_val_loss,
                "best_epoch": best_epoch,
            })
        


    # Plot Training Progression
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)
    fig_path = os.path.join('plots', SAVE_NAME+'_progression'+'.png')
    plt.savefig(fig_path)
    # plt.show()
    print(f"\nTraining completed. Best model saved at: {checkpoint_path}")
    return model, checkpoint_path


test_dataset = SocNavHeteroDataset(
    data_list_file=TEST_FILE, 
    data_path=DATA_PATH, 
    context_path=CONTEXT_FILE, 
    timestamp_threshold=TIMESTAMP_THRESHOLD,
    reload = RELOAD
)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=True)

sample_graph = test_dataset[0][0][0]  

# Construimos gnn_data dinámicamente
gnn_data = {
    'input': {k: v.shape[1] for k, v in sample_graph.x_dict.items()},
    'output': RNN_HIDDEN_SIZE,                                  
    'hidden_channels': GNN_HIDDEN_SIZE,                         
    'num_edges': len(sample_graph.edge_index_dict),         
    'heads': GNN_HEADS,               
    'concat': GNN_CONCAT,         
    'metadata': sample_graph.metadata()                     
}

# Construimos rnn_data dinámicamente
rnn_data = {
    'type': RNN_TYPE,  
    'input': RNN_HIDDEN_SIZE, # El input de la rnn es igual al output de la gnn                                     
    'hidden_channels': RNN_HIDDEN_SIZE                          
}

def get_qual_loader(file_path):
    # print(f"Processing file: {file_path}")
    q_loaders = []
    with open(file_path) as set_file:
        files = set_file.read().splitlines()
    # context = {"urgency":98., "importance":98., "risk":95 , "distance_from_human":95., "minimum_speed":20, "average_speed":15., "maximum_speed":15.}
    for abbreviated_context, context in zip(ABBREVIATED_CONTEXTS, CONTEXTS):
        print(f"Creating q_test for: {context}")
        qual_set = SocNavHeteroDataset(data_list_file = file_path, data_path = Q_TEST_PATH, context_path = CONTEXT_FILE, timestamp_threshold = TIMESTAMP_THRESHOLD)
        qual_loader = DataLoader(qual_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=True)
        q_loaders.append((context, files, qual_loader, abbreviated_context))
    return q_loaders
# Load and process test datasets
qual_loaders = {}
# for id, path in enumerate(Q_FILES):
#     speed = Q_NAMES[id]
#     print(f"Processing files with {speed}")
#     qual_loaders[speed] = get_qual_loader(os.path.join(Q_TEST_PATH,path))

model, checkpoint = train_model(
    rnn_data=rnn_data, 
    gnn_data=gnn_data, 
    num_layers=NUM_LAYERS,
    batch_size=BATCH_SIZE, 
    num_epochs=MAX_EPOCHS, 
    patience=MAX_PATIENCE, 
    learning_rate=LR
)
if UPLOAD_TO_WANDB is True:
    wandb.finish()
