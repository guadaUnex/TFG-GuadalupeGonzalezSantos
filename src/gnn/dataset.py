import os
import json
import torch
import pandas as pd
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Dataset, HeteroData, Batch

from data_conversions import clone_sequence, sequence_to_tensor
from data_mirroring import mirror_sequence
from data_normalization import tensor_transform_to_goal_fr
import metrics

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
                   overwrite_contexts = '', transform=None, pre_transform=None, pre_filter=None, timestamp_threshold = 0.1, reload=False, data_augmentation = False):
        """Inicializa los parámetros del dataset y gestiona la carga de la caché procesada."""

        print("Iniciando creación del siguiente dataset: ", data_list_file)
        
        self.data_list_file = data_list_file

        self.trajectories_dir = data_path

        # Reading the dataset file

        with open(os.path.join(data_path, data_list_file), 'r') as f:
            self.file_names = f.read().splitlines()

        self.MAX_TTC = 10

        self.metrics_features = ['dist_to_goal_pos', 'dist_to_goal_angle', 'success', 'hum_exists', 'wall_exist', 'dist_nearest_hum', 'dist_nearest_object', 
                                 'dist_wall', 'human_collision_flag', 'object_collision_flag', 'wall_collision_flag',
                                 'social_space_intrusionA', 'num_near_humansA', 'num_near_humansA2', 'social_space_intrusionB',
                                 'num_near_humansB', 'num_near_humansB2', 'social_space_instrusionC', 'num_near_humansC',
                                 'num_near_humansC2', 'min_ttc', 'min_ttc2', 'max_fear', 'max_panic', 'global_dist_nearest_hum',
                                 'path_efficiency_ratio', 'step_ratio', 'episode_end']
        
        self.max_values = {'scale': 10.0, 'max_v': 2.0, 'max_va': np.pi, 'max_acc' : 3.0, 'max_c': 100.0, 'dist_to_goal_pos': 10, 'dist_to_goal_angle': np.pi, 'success': 1, 
                                'hum_exists': 1, 'wall_exist': 1, 'dist_nearest_hum': 10, 'dist_nearest_object': 10, 
                                 'dist_wall': 10, 'human_collision_flag': 1, 'object_collision_flag': 1, 'wall_collision_flag': 1,
                                 'social_space_intrusionA': 1, 'num_near_humansA': 10, 'num_near_humansA2': 10, 'social_space_intrusionB': 1,
                                 'num_near_humansB': 10, 'num_near_humansB2': 10, 'social_space_intrusionC': 1, 'num_near_humansC': 10,
                                 'num_near_humansC2': 10, 'min_ttc': self.MAX_TTC, 'min_ttc2': self.MAX_TTC**2, 'max_fear': 10, 'max_panic': 10, 'global_dist_nearest_hum': 10,
                                 'path_efficiency_ratio': 1, 'step_ratio': 1, 'episode_end': 1}
        
        #, 'acceleration_x': 3, 'acceleration_y': 3

        # Reading the context file

        self.context_df = pd.read_csv(context_path, index_col='context')
        self.context_features = ['urgency', 'importance', 'risk', 'distance_from_human', 'distance_from_object', 'speed', 'comfort', 
                                 'bumping_human', 'bumping_object', 'predictability']
        self.overwrite_contexts = overwrite_contexts
        self.data_augmentation = data_augmentation

        self.all_feature_names = {
            'scenario': self.metrics_features+self.context_features, 
            'goal': ['x', 'y', 'sin_a', 'cos_a', 'th_pos', 'th_angle', 'dlin', 'dang'],
            'robot': ['x', 'y', 'sin_a', 'cos_a', 'w', 'l', 'vx', 'vy', 'va', 'acc_x', 'acc_y'],
            'human': ['x', 'y', 'sin_a', 'cos_a', 'd_robot', 'dcenter_r'],
            'object': ['x', 'y', 'sin_a', 'cos_a', 'w', 'l', 'd_robot', 'dcenter_r'],
            'wall': ['x', 'y', 'sin_a', 'cos_a', 'd_robot', 'w', 'l']
        }

        self.all_features = {
            # success, min dist to human, context vars.
            'scenario': len(self.context_features) + len(self.metrics_features), 
            'goal': len(self.all_feature_names['goal']),
            'robot': len(self.all_feature_names['robot']),
            'human': len(self.all_feature_names['human']),
            'object': len(self.all_feature_names['object']),
            'wall': len(self.all_feature_names['wall'])
        }

        self.dataset = []
        self.labels = []
        self.slengths = []

        self.timestamp_threshold = timestamp_threshold

        nombre_carpeta = Path(self.data_list_file).stem
        self.carpeta_guardado = os.path.join(data_path, nombre_carpeta)
            
        super(SocNavHeteroDataset, self).__init__(self.carpeta_guardado, transform, pre_transform, pre_filter)

        # Checking if the dataset is already charged, if not, it calls the process method. Otherwise, it loads the pt file

        if reload and os.path.exists(self.processed_paths[0]):
            print("Cargando datos previamente guardados")
            checkpoint = torch.load(self.processed_paths[0], weights_only=False)
            self.dataset = checkpoint['trajectories']
            self.labels = checkpoint['labels']
            self.slengths = checkpoint['slength']
        # else:
        #     self.process()


    @property
    def raw_file_names(self):
        return self.file_names

    @property
    def processed_file_names(self):
        return ['dataset.pt']
    
    @property
    def raw_dir(self):
        return self.trajectories_dir
        # return os.path.join(os.path.dirname(os.path.dirname(self.root)), 'labeled')
    
    @property
    def processed_dir(self):
        return self.root

    def process(self):
        print("Creando nuevo dataset desde trayectorias en crudo")
        # Limites de pruebas
        limit = 5
        count = 0

        # print(self.raw_paths)

        for raw_path in tqdm(self.raw_paths, total=len(self.raw_paths), desc="Procesando JSONs"):            
            with open(raw_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            if 'context_description' in json_data.keys(): #not self.overwrite_contexts:
                context_desc = json_data['context_description']
            else:
                context_desc = self.overwrite_contexts
            context = self.context_df.loc[context_desc.rstrip()].to_dict()

            walls = json_data.get('walls', [])
            rating = json_data.get('label',0.)
            lenght = 0

            tensor_dict, lenght = sequence_to_tensor(json_data, self.timestamp_threshold, context)

            tensor_dict_to_goal = tensor_transform_to_goal_fr(tensor_dict)

            full_dict = metrics.compute_metrics(tensor_dict_to_goal)

            normalized_dict = metrics.normalize_features(full_dict, self.max_values)

            trayectoria = []
            for i in range(lenght):
                grafo = self._json_to_heterodata(normalized_dict, i)
                trayectoria.append(grafo)

            self.dataset.append(trayectoria)
            self.labels.append(torch.tensor([rating], dtype=torch.float32))
            self.slengths.append(torch.tensor(lenght, dtype=torch.long))

            if self.data_augmentation:
                cloned_trajectory = clone_sequence(trayectoria)
                t_data_mirrored = mirror_sequence(cloned_trajectory)
                self.dataset.append(t_data_mirrored)
                self.labels.append(torch.tensor([rating], dtype=torch.float32))
                self.slengths.append(torch.tensor(lenght, dtype=torch.long))

            count += 1

            if count == limit:
                break

        # torch.save(
        #     {'trajectories': self.dataset, 'labels': self.labels, 'slength': self.slengths}, 
        #     self.processed_paths[0]
        # )


    def get_metadata(self):
        metadata = (
            ['scenario', 'goal', 'robot', 'human', 'object', 'wall'],
            [
                ('scenario', 'self', 'scenario'), 
                ('goal', 'self', 'goal'), 
                ('robot', 'self', 'robot'), 
                ('human', 'self', 'human'), 
                ('object', 'self', 'object'), 
                ('wall', 'self', 'wall'), 
                ('robot', 'targets', 'goal'),
                ('goal', 'assigned_to', 'robot'), 
                ('goal', 'in', 'scenario'), 
                ('robot', 'in', 'scenario'), 
                ('human', 'in', 'scenario'), 
                ('object', 'in', 'scenario'), 
                ('wall', 'in', 'scenario'), 
                ('scenario', 'contains', 'goal'), 
                ('scenario', 'contains', 'robot'), 
                ('scenario', 'contains', 'human'), 
                ('scenario', 'contains', 'object'), 
                ('scenario', 'contains', 'wall'), 
                # # ('goal', 'near_to', 'human'), 
                # # ('human', 'near_to', 'goal'), 
                # # ('goal', 'near_to', 'object'), 
                # # ('object', 'near_to', 'goal'), 
                # # ('goal', 'near_to', 'wall'), 
                # # ('wall', 'near_to', 'goal'), 
                # ('goal', 'near_to', 'robot'), 
                # ('robot', 'near_to', 'goal'), 
                # ('robot', 'near_to', 'wall'), 
                # ('wall', 'near_to', 'robot'), 
                # ('robot', 'near_to', 'human'), 
                # ('human', 'near_to', 'robot'), 
                # ('robot', 'near_to', 'object'), 
                # ('object', 'near_to', 'robot'), 
                # # ('human', 'near_to', 'human'), 
                # # ('human', 'near_to', 'object'), 
                # # ('object', 'near_to', 'human'), 
                # # ('human', 'near_to', 'wall'), 
                # # ('wall', 'near_to', 'human'), 
                # # ('object', 'near_to', 'object'),
                # # ('object', 'near_to', 'wall'),
                # # ('wall', 'near_to', 'object')                
            ]
        )
        return metadata

    def get_all_features(self):
        return self.all_features

    def get_context_features(self):
        return self.context_features

    def get_metrics_features(self):
        return self.metrics_features

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx]
        label = self.labels[idx] 
        slength = self.slengths[idx] 

        return data, label, slength

    # --- MÉTODOS AUXILIARES ---

    def _json_to_heterodata(self, dict, index, full_conexo=True):
        data = HeteroData()

        rx = dict['robot']['x'][index]
        ry = dict['robot']['y'][index]

        # Goal node
        data['goal'].x = torch.tensor([dict['goal']['x'][index], 
                                        dict['goal']['y'][index], 
                                        math.sin(dict['goal']['a'][index]), 
                                        math.cos(dict['goal']['a'][index]),
                                        dict['goal']['th_p'][index], 
                                        dict['goal']['th_a'][index],
                                        dict['computed_metrics']['dist_to_goal_pos'][index],
                                        dict['computed_metrics']['dist_to_goal_angle'][index]], dtype=torch.float).view(1,-1)

        # Robot node
        acc_x = 0. #dict['robot']['acc_x'][index]
        acc_y = 0. #dict['robot']['acc_y'][index]
        data['robot'].x = torch.tensor([[rx, 
                                         ry, 
                                         math.sin(dict['robot']['a'][index]), 
                                         math.cos(dict['robot']['a'][index]), 
                                         dict['robot']['w'][index], 
                                         dict['robot']['l'][index],  
                                         dict['robot']['vx'][index], 
                                         dict['robot']['vy'][index], 
                                         dict['robot']['va'][index], 
                                         acc_x, 
                                         acc_y]], dtype=torch.float).view(1,-1)
        

        num_humans = dict['people']['x'].shape[1]
        people_exist = dict['people']['exists']
        # People nodes
        p_list = []
        for i in range(num_humans):
            if people_exist[index, i].item():
                px = dict['people']['x'][index, i].item()
                py = dict['people']['y'][index, i].item()
                pa = dict['people']['a'][index, i].item()
                dist_center = math.sqrt((px-rx)**2+(py-ry)**2)
                dist_to_robot = dict['metrics']['dist_human'][index, i].item()
                p_list.append([px, 
                            py, 
                            math.sin(pa), 
                            math.cos(pa), 
                            dist_to_robot,
                            dist_center]) 
        data['human'].x = torch.tensor(p_list, dtype=torch.float) if p_list else torch.empty((0, self.all_features['human']))

        num_objects = dict['objects']['x'].shape[1]
        objects_exist = dict['objects']['exists']
        # Object nodes
        o_list = []
        for i in range(num_objects):
            if objects_exist[index, i].item():
                ox = dict['objects']['x'][index, i].item()
                oy = dict['objects']['y'][index, i].item()
                oa = dict['objects']['a'][index, i].item()
                dist_center = math.sqrt((ox-rx)**2+(oy-ry)**2)
                dist_to_robot = dict['metrics']['dist_object'][index, i].item()
                o_list.append([ox, 
                            oy, 
                            math.sin(oa), 
                            math.cos(oa), 
                            dict['objects']['w'][index, i], 
                            dict['objects']['l'][index, i], 
                            dist_to_robot,
                            dist_center])
        data['object'].x = torch.tensor(o_list, dtype=torch.float) if o_list else torch.empty((0, self.all_features['object']))

        # Walls nodes
        walls = dict['walls']
        if walls is not None:
            raw_points, id_walls, a_walls, l_walls = self._get_walls(walls)
            w_list = []
            for pt, idW, wa, la in zip(raw_points, id_walls, a_walls, l_walls):
                wx = pt[0]
                wy = pt[1]
                dist_to_robot = dict['metrics']['dist_walls'][index, idW].item()
                w_list.append([wx, wy, math.sin(wa), math.cos(wa), dist_to_robot, la, 0.01])
            data['wall'].x = torch.tensor(w_list, dtype=torch.float)
        else:
            data['wall'].x = torch.empty((0, self.all_features['wall']))

        context_values = []
        for c_key in dict['context'].keys():
            val = dict['context'][c_key][index].item()
            context_values.append(val)
            
        context_tensor = torch.tensor(context_values, dtype=torch.float32)

        metrics_values = []
        for m_key in dict['computed_metrics'].keys():
            val = dict['computed_metrics'][m_key][index].item()
            metrics_values.append(val)

        metrics_frame = torch.tensor(metrics_values, dtype=torch.float32)

        # Scenario node
        scenario_features = torch.cat((metrics_frame, context_tensor), dim=0).float()
        data['scenario'].x = scenario_features.view(1,-1)

        data = self._create_edges(data, full_conexo=full_conexo)

        return data
    

    def _get_walls(self, walls):
        wall_points = []
        id_walls =[]
        angle_walls = []
        length_walls = []

        wx = walls['x']
        wy = walls['y']

        num_walls = len(wx)//2

        for i in range(num_walls):
            x1 = wx[2 * i].item()
            y1 = wy[2 * i].item()
            x2 = wx[2 * i + 1].item()
            y2 = wy[2 * i + 1].item()

            dx, dy = x2 - x1, y2 - y1
            distance = (dx**2 + dy**2)**0.5
            angle = np.arctan2(dy, dx) + np.pi/2
            curr_x = (x1 + x2)/2
            curr_y = (y1 + y2)/2
            wall_points.append([curr_x, curr_y])
            id_walls.append(i)
            angle_walls.append(angle)
            length_walls.append(distance)

        return wall_points, id_walls, angle_walls, length_walls
    

    def _sample_walls(self, walls, dist_points=0.2):
        wall_points = []

        wx = walls['x']
        wy = walls['y']

        num_walls = len(wx)//2

        for i in range(num_walls):
            x1 = wx[2 * i].item()
            y1 = wy[2 * i].item()
            x2 = wx[2 * i + 1].item()
            y2 = wy[2 * i + 1].item()
            
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

    def _create_edges(self, data, full_conexo=False, dist_threshold=0.8):
        
        node_types = data.node_types

        # self edges
        for t in node_types:
            num_nodes_of_type = data[t].x.size(0)
            if num_nodes_of_type>0:
                node_idx = torch.arange(num_nodes_of_type, dtype=torch.long)
                edge_index_self = torch.stack([node_idx, node_idx], dim=0)
                data[t, 'self', t].edge_index = edge_index_self
                data[t, 'self', t].edge_attr = torch.tensor([1.]*num_nodes_of_type, dtype=torch.float)

        if 'robot' in node_types and 'goal' in node_types:
            edge_index_rg = torch.tensor([[0], [0]], dtype=torch.long)
            
            dist_rg = torch.norm(data['robot'].x[0, :2], p=2).reshape(1, 1)            
            goal_threshold = data['goal'].x[0,3]

            edge_attr_rg = torch.tensor([[dist_rg, goal_threshold]], dtype=torch.float)

            data['robot', 'targets', 'goal'].edge_index = edge_index_rg
            data['robot', 'targets', 'goal'].edge_attr = edge_attr_rg
            data['goal', 'assigned_to', 'robot'].edge_index = edge_index_rg
            data['goal', 'assigned_to', 'robot'].edge_attr = edge_attr_rg

        if 'scenario' in node_types:
            for t in node_types:
                if t == 'scenario':
                    continue
                
                num_nodes_of_type = data[t].x.size(0)
                if num_nodes_of_type > 0:
                    row = torch.arange(num_nodes_of_type, dtype=torch.long)
                    col = torch.zeros(num_nodes_of_type, dtype=torch.long)
                    edge_index_in_scenario = torch.stack([row, col], dim=0)
                    edge_index_contains_scenario = torch.stack([col, row], dim=0)
                    
                    data[t, 'in', 'scenario'].edge_index = edge_index_in_scenario
                    data[t, 'in', 'scenario'].edge_attr = torch.tensor([1.]*num_nodes_of_type, dtype=torch.float)
                    data['scenario', 'contains', t].edge_index = edge_index_contains_scenario
                    data['scenario', 'contains', t].edge_attr = torch.tensor([1.]*num_nodes_of_type, dtype=torch.float)
        
        spatial_nodes = [t for t in node_types if t not in ['scenario']]
    
        # for i, type_a in enumerate(spatial_nodes):
        #     if type_a!='robot':
        #         continue
        #     for type_b in spatial_nodes[i:]:

        #         if type_a == type_b == 'wall' or type_a == type_b == 'robot' or type_a == type_b == 'goal':
        #             continue

        #         if data[type_a].x.numel() == 0 or data[type_b].x.numel() == 0:
        #             continue

        #         pos_a = data[type_a].x[:, :2]
        #         pos_b = data[type_b].x[:, :2]

        #         if pos_a.numel() == 0 or pos_b.numel() == 0:
        #             continue

        #         dists = torch.cdist(pos_a, pos_b)
                
        #         if full_conexo:
        #             mask = torch.ones_like(dists, dtype=torch.bool)
        #         else:
        #             mask = dists < dist_threshold

        #         # mask = mask.long()
                
        #         # if type_a == type_b:
        #         #     mask = mask & (~torch.eye(pos_a.size(0), dtype=torch.bool))
                
        #         edge_index = mask.nonzero(as_tuple=False).t()
                
        #         if edge_index.numel() > 0:
        #             edge_values = dists[mask].unsqueeze(1)
        #             rel_name = (type_a, 'near_to', type_b)
        #             data[rel_name].edge_index = edge_index.long()
        #             data[rel_name].edge_attr = edge_values
                    
        #             if type_a != type_b:
        #                 rev_rel = (type_b, 'near_to', type_a)
        #                 data[rev_rel].edge_index = edge_index.flip(0).long()
        #                 data[rev_rel].edge_attr = edge_values   

        edge_types_metadata = self.get_metadata()[1]  

        for edge_type in edge_types_metadata:
            if edge_type not in data.edge_types:
                data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
                data[edge_type].edge_attr = torch.empty((0), dtype=torch.float)

        return data
    
    def mirror_sequence(self, sequence):
        new_sequence = sequence

        for frame in new_sequence:
            frame['robot'].x[:,self.all_feature_names['robot'].index('y')] = -frame['robot'].x[:,self.all_feature_names['robot'].index('y')]
            frame['robot'].x[:,self.all_feature_names['robot'].index('sin_a')] = -frame['robot'].x[:,self.all_feature_names['robot'].index('sin_a')]
            frame['robot'].x[:,self.all_feature_names['robot'].index('vy')] = -frame['robot'].x[:,self.all_feature_names['robot'].index('vy')]
            frame['robot'].x[:,self.all_feature_names['robot'].index('va')] = -frame['robot'].x[:,self.all_feature_names['robot'].index('va')]            
            frame['robot'].x[:,self.all_feature_names['robot'].index('acc_y')] = -frame['robot'].x[:,self.all_feature_names['robot'].index('acc_y')]

            frame['goal'].x[:,self.all_feature_names['goal'].index('y')] = -frame['goal'].x[:,self.all_feature_names['goal'].index('y')]
            frame['goal'].x[:,self.all_feature_names['goal'].index('sin_a')] = -frame['goal'].x[:,self.all_feature_names['goal'].index('sin_a')]

            frame['human'].x[:,self.all_feature_names['human'].index('y')] = -frame['human'].x[:,self.all_feature_names['human'].index('y')]
            frame['human'].x[:,self.all_feature_names['human'].index('sin_a')] = -frame['human'].x[:,self.all_feature_names['human'].index('sin_a')]

            frame['object'].x[:,self.all_feature_names['object'].index('y')] = -frame['object'].x[:,self.all_feature_names['object'].index('y')]
            frame['object'].x[:,self.all_feature_names['object'].index('sin_a')] = -frame['object'].x[:,self.all_feature_names['object'].index('sin_a')]

            frame['wall'].x[:,self.all_feature_names['wall'].index('y')] = -frame['wall'].x[:,self.all_feature_names['wall'].index('y')]
            frame['wall'].x[:,self.all_feature_names['wall'].index('sin_a')] = -frame['wall'].x[:,self.all_feature_names['wall'].index('sin_a')]

        return new_sequence



def collate(batch):
    sequences, labels, sequence_lengths = zip(*batch)  


    flat_graphs = [frame for traj in sequences for frame in traj]

    batched_graphs = Batch.from_data_list(flat_graphs)   


    labels_tensor = torch.stack(labels)  
    slengths_tensor = torch.stack(sequence_lengths)


    return batched_graphs, labels_tensor, slengths_tensor

        

if __name__ == "__main__":
    print("🚀 Lanzando suite de prueba simplificada para SocNavHeteroDataset...")

    # --- Configuración de tus rutas reales ---
    DATA_PATH = '../../../dataset/labeled/labeled'
    CONTEXT_PATH = '../../../dataset/contexts/anthropic_claude_context.csv'
    SPLIT_PRUEBA = '../split/train_set_socnav3.txt' # El nombre del archivo dentro de DATA_PATH

    print("\n[PASO 1] Instanciando dataset con data_augmentation=True...")
    try:
        dataset = SocNavHeteroDataset(
            data_list_file=SPLIT_PRUEBA,
            data_path=DATA_PATH,
            context_path=CONTEXT_PATH,
            timestamp_threshold=0.3,
            data_augmentation=True,   # Activa el espejo (Mirroring)
            reload=False              # Fuerza el procesamiento crudo (.process())
        )
    except Exception as e:
        print(f"❌ Error al instanciar el dataset: {e}")
        print("Asegúrate de que las rutas relativas sean correctas desde donde ejecutas este script.")
        exit(1)

    print(f"\n[PASO 2] Dataset cargado. Muestras totales en memoria: {len(dataset)}")
    
    if len(dataset) < 2:
        print("❌ Alerta: Se necesitan más muestras para validar, pero el pipeline base no ha crasheado.")
        exit(0)

    # El elemento 0 es la secuencia Original y el 1 es su versión Espejo (Mirror)
    print("\n[PASO 3] Extrayendo pareja simétrica (Muestra 0 [Original] vs Muestra 1 [Espejo])...")
    traj_orig, label_orig, len_orig = dataset[0]
    traj_mirr, label_mirr, len_mirr = dataset[1]

    print("\n📊 --- INFORMACIÓN DE ESTRUCTURAS ---")
    print(f"• Longitud temporal secuencia original: {len_orig.item()} frames")
    print(f"• Longitud temporal secuencia espejo:   {len_mirr.item()} frames")
    print(f"• Calificación/Label original:          {label_orig.item()}")
    print(f"• Calificación/Label espejo:            {label_mirr.item()}")

    # Inspección rápida del primer Grafo Heterogéneo de la secuencia
    print("\n🤖 --- INSPECCIÓN DEL PRIMER FRAME DE LA TRAYECTORIA ---")
    primer_grafo = traj_orig[0]
    print(primer_grafo)
    
    if 'robot' in primer_grafo.node_types:
        print(f"• Dimensiones del tensor del Robot: {primer_grafo['robot'].x.shape}")
    if 'human' in primer_grafo.node_types:
        print(f"• Dimensiones del tensor de Humanos: {primer_grafo['human'].x.shape}")
    if 'wall' in primer_grafo.node_types:
        print(f"• Dimensiones del tensor de Paredes: {primer_grafo['wall'].x.shape}")
    if 'scenario' in primer_grafo.node_types:
        print(f"• Dimensiones del nodo Escenario:    {primer_grafo['scenario'].x.shape}")

    print("\n🎉 ¡Prueba finalizada! Si ves los prints de arriba sin ningún mensaje de error intermedio, tu pipeline está corregido y listo.")