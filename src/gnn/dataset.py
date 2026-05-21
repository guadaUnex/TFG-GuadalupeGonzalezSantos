import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Dataset, HeteroData, Batch
from transforms import GoalFrameTransform

class SocNavHeteroDataset(Dataset):
    """Heterogeneous Dataset for Social Navigation. 

    It loads social navigation scenarios trajectories from JSON files and transforms
    the data into temporary sequences of heterogeneous graphs (`HeteroData`).

    Attributes:
    data_list_file (str): Name of the file that contains the names of the trajectories files.
    file_names (list): List of JSON-to-process names.
    context_df (pd.DataFrame): DataFrame with context feature vectors. 
    transformer (GoalFrameTransform): Module for spatial transformations and normalization.
    dataset (list): Memory struture for storing the proccessed trajectories.
    timestamp_threshold (float): Maximum allowed time interval between consecutive frames.
        """
    
    def __init__(self, data_list_file, data_path='../../../dataset/labeled', context_path = '../../../dataset/contexts/anthropic_claude_context.csv',
                   transform=None, pre_transform=None, pre_filter=None, timestamp_threshold = 0.3):
        """Inicializa los parámetros del dataset y gestiona la carga de la caché procesada."""
        
        self.data_list_file = data_list_file

        # Reading the dataset file

        with open(os.path.join(data_path, 'split', data_list_file), 'r') as f:
            self.file_names = f.read().splitlines()

        # Reading the context file

        self.context_df = pd.read_csv(context_path, index_col='context')
        self.context_features = ['urgency', 'importance', 'risk', 'distance_from_human', 'distance_from_object', 'speed', 'comfort', 
                                 'bumping_human', 'bumping_object', 'predictability']
        
        self.all_features = {
            'scenario': 2 + len(self.context_features), 
            'goal': 5,
            'robot': 6,
            'human': 3,
            'object': 5,
            'wall': 2
        }

        self.transformer = GoalFrameTransform(scale=10.0, v_max=2.0)
        self.dataset = []
        self.labels = []
        self.slengths = []

        self.timestamp_threshold = timestamp_threshold
            
        super(SocNavHeteroDataset, self).__init__(data_path, transform, pre_transform, pre_filter)

        # Checking if the dataset is already charged, if not, it calls the process method. Otherwise, it loads the pt file

        dataset_path = os.path.join(self.processed_dir, 'dataset.pt')
        if os.path.exists(dataset_path) and len(self.dataset) == 0:
            checkpoint = torch.load(dataset_path, weights_only=False)
            self.dataset = checkpoint['trajectories']
            self.labels = checkpoint['labels']
            self.slengths = checkpoint['slength']


    @property
    def raw_file_names(self):
        return self.file_names

    @property
    def processed_file_names(self):
        return ['dataset.pt']
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'labeled')
    
    @property
    def processed_dir(self):
        return os.path.join(self.root)

    def process(self):
        # Limites de pruebas
        limit = 25
        count = 0
        
        for raw_path in tqdm(self.raw_paths, total=len(self.raw_paths), desc="Procesando JSONs"):            
            with open(raw_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            walls = json_data.get('walls', [])
            context_desc = json_data.get('context_description',[])
            rating = json_data.get('label',[])
            lenght = 0

            context = list(self.context_df.loc[context_desc.rstrip()].to_dict().values())
            self.transformer.normalize_context(context)

            trayectoria = []
            prev_timestamp = json_data['sequence'][0]['timestamp']

            for frame_data in json_data['sequence']:
                if (frame_data['timestamp'] - prev_timestamp) < self.timestamp_threshold:
                    grafo_frame = self._json_to_heterodata(frame_data, walls, context)
                    trayectoria.append(grafo_frame)
                    prev_timestamp = frame_data['timestamp']
                    lenght += 1

            self.dataset.append(trayectoria)
            self.labels.append(rating)
            self.slengths.append(lenght)

            count += 1

            if count == limit:
                break

        

        torch.save({'trajectories':self.dataset, 'labels': self.labels, 'slength': self.slengths}, os.path.join(self.processed_dir, 'dataset.pt'))


    def get_all_features(self):
        return self.all_features

    def get_context_features(self):
        return self.context_features

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        slength = torch.tensor(self.slengths[idx], dtype=torch.long)

        return data, label, slength

    # --- MÉTODOS AUXILIARES ---

    def _json_to_heterodata(self, frame, walls, context, full_conexo=False):
        data = HeteroData()
        s = self.transformer.scale  
        v = self.transformer.v_max

        # We read de robot data first to save some information in the scenario node
        r = frame['robot']

        # Scenario node
        scenario_features = [r['shape']['width']/s, r['shape']['length']/s] + context
        data['scenario'].x = torch.tensor([scenario_features], dtype=torch.float)

        # Goal node
        g = frame['goal']
        gx, gy, ga = torch.tensor(g['x']), torch.tensor(g['y']), torch.tensor(g['angle'])
        data['goal'].x = torch.tensor([[0.0, 0.0, 0, g['pos_threshold']/s, g['angle_threshold']/3.14]], dtype=torch.float)

        # Robot node
        nx, ny, na = self.transformer.transform_pose(r['x'], r['y'], r['angle'], gx, gy, ga)
        nvx, nvy, nva = self.transformer.transform_velocity(r['speed_x'], r['speed_y'], r['speed_a'], ga)
        data['robot'].x = torch.tensor([[nx, ny, na, nvx, nvy, nva]], dtype=torch.float)

        # People nodes
        people = frame.get('people', [])
        p_list = []
        for p in people:
            nx, ny, na = self.transformer.transform_pose(p['x'], p['y'], p['angle'], gx, gy, ga)
            p_list.append([nx, ny, na])
        data['human'].x = torch.tensor(p_list, dtype=torch.float) if p_list else torch.empty((0, 3))

        # Object nodes
        objects = frame.get('objects', [])
        o_list = []
        for o in objects:
            nx, ny, na = self.transformer.transform_pose(o['x'], o['y'], o['angle'], gx, gy, ga)
            o_list.append([nx, ny, na, o['shape']['width']/s, o['shape']['length']/s])
        data['object'].x = torch.tensor(o_list, dtype=torch.float) if o_list else torch.empty((0, 5))

        # Walls nodes
        if walls:
            raw_points = self._sample_walls(walls)
            w_list = []
            for pt in raw_points:
                wx, wy, _ = self.transformer.transform_pose(pt[0].item(), pt[1].item(), 0.0, gx, gy, ga)
                w_list.append([wx, wy])
            data['wall'].x = torch.tensor(w_list, dtype=torch.float)
        else:
            data['wall'].x = torch.empty((0, 2))
       
        data = self._create_edges(data, full_conexo=full_conexo)

        return data
    

    def _sample_walls(self, walls, dist_points=0.2):
        wall_points = []

        for wall in walls:
            x1, y1, x2, y2 = wall
            
            dx, dy = x2 - x1, y2 - y1
            distance = (dx**2 + dy**2)**0.5
            
            if distance < 1e-6:
                wall_points.append([x1, y1])
                continue
                
            num_points = max(int(distance / dist_points), 1)
            
            for i in range(num_points + 1):
                t = i / num_points
                curr_x = x1 + t * dx
                curr_y = y1 + t * dy
                wall_points.append([curr_x, curr_y])
                
        points_tensor = torch.tensor(wall_points, dtype=torch.float)
        unique_points = torch.unique(points_tensor, dim=0)
        
        return unique_points

    def _create_edges(self, data, full_conexo=False, dist_threshold=0.25):
        
        node_types = data.node_types

        if 'robot' in node_types and 'goal' in node_types:
            edge_index_rg = torch.tensor([[0], [0]], dtype=torch.long)
            
            dist_rg = torch.norm(data['robot'].x[0, :2], p=2).reshape(1, 1)
            
            goal_threshold = data['goal'].x[0, 3]
            edge_attr_rg = torch.tensor([[dist_rg, goal_threshold]], dtype=torch.float)

            data['robot', 'targets', 'goal'].edge_index = edge_index_rg
            data['robot', 'targets', 'goal'].edge_attr = edge_attr_rg

        if 'scenario' in node_types:
            for t in node_types:
                if t == 'scenario':
                    continue
                
                num_nodes_of_type = data[t].x.size(0)
                if num_nodes_of_type > 0:
                    row = torch.arange(num_nodes_of_type, dtype=torch.long)
                    col = torch.zeros(num_nodes_of_type, dtype=torch.long)
                    edge_index_in_scenario = torch.stack([row, col], dim=0)
                    
                    data[t, 'in', 'scenario'].edge_index = edge_index_in_scenario

        spatial_nodes = [t for t in node_types if t not in ['goal', 'scenario']]
    
        for i, type_a in enumerate(spatial_nodes):
            for type_b in spatial_nodes[i:]:

                if type_a == 'wall' and type_b == 'wall':
                    continue

                pos_a = data[type_a].x[:, :2]
                pos_b = data[type_b].x[:, :2]
                
                if pos_a.numel() == 0 or pos_b.numel() == 0:
                    continue

                dists = torch.cdist(pos_a, pos_b)
                
                if full_conexo:
                    mask = torch.ones_like(dists, dtype=torch.bool)
                else:
                    mask = dists < dist_threshold
                
                if type_a == type_b:
                    mask = mask & (~torch.eye(pos_a.size(0), dtype=torch.bool))
                
                edge_index = mask.nonzero(as_tuple=False).t()
                
                if edge_index.numel() > 0:
                    edge_values = dists[mask].unsqueeze(1)
                    rel_name = (type_a, 'near_to', type_b)
                    data[rel_name].edge_index = edge_index
                    data[rel_name].edge_attr = edge_values
                    
                    if type_a != type_b:
                        rev_rel = (type_b, 'near_to', type_a)
                        data[rev_rel].edge_index = edge_index.flip(0)
                        data[rev_rel].edge_attr = edge_values                     
                        
        return data


def collate(batch):
    sequences, labels, sequence_lengths = zip(*batch)  

    flat_graphs = [frame for traj in sequences for frame in traj]
    batched_graphs = Batch.from_data_list(flat_graphs)    

    labels_tensor = torch.stack(labels)  
    slengths_tensor = torch.stack(sequence_lengths)

    return batched_graphs, labels_tensor, slengths_tensor
        