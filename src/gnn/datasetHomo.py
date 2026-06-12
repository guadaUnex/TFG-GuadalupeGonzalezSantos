import os
import json
import torch
import pandas as pd
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Dataset, Data, Batch

from data_conversions import clone_sequence, sequence_to_tensor
from data_normalization import tensor_transform_to_goal_fr
import metrics

class SocNavHomoDataset(Dataset):
    """Homogeneous Dataset for Social Navigation. 

    It loads social navigation scenarios trajectories from JSON files and transforms
    the data into temporary sequences of homogeneous graphs (`Data`).

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

        self.metrics_features = ['dist_to_goal_pos', 'success', 'hum_exists', 'wall_exist', 'dist_nearest_hum', 'dist_nearest_object', 
                                 'dist_wall', 'human_collision_flag', 'object_collision_flag', 'wall_collision_flag',
                                 'social_space_intrusionA', 'num_near_humansA', 'num_near_humansA2', 'social_space_intrusionB',
                                 'num_near_humansB', 'num_near_humansB2', 'social_space_instrusionC', 'num_near_humansC',
                                 'num_near_humansC2', 'min_ttc', 'min_ttc2', 'max_fear', 'max_panic', 'global_dist_nearest_hum',
                                 'path_efficiency_ratio', 'step_ratio', 'episode_end']
        
        self.max_values = {'scale': 10.0, 'max_v': 2.0, 'max_va': np.pi, 'max_acc' : 3.0, 'max_c': 100.0, 'dist_to_goal_pos': 10, 'success': 1, 'hum_exists': 1, 'wall_exist': 1, 'dist_nearest_hum': 10, 'dist_nearest_object': 10, 
                                 'dist_wall': 10, 'human_collision_flag': 1, 'object_collision_flag': 1, 'wall_collision_flag': 1,
                                 'social_space_intrusionA': 1, 'num_near_humansA': 10, 'num_near_humansA2': 10, 'social_space_intrusionB': 1,
                                 'num_near_humansB': 10, 'num_near_humansB2': 10, 'social_space_intrusionC': 1, 'num_near_humansC': 10,
                                 'num_near_humansC2': 10, 'min_ttc': self.MAX_TTC, 'min_ttc2': self.MAX_TTC**2, 'max_fear': 10, 'max_panic': 10, 'global_dist_nearest_hum': 10,
                                 'path_efficiency_ratio': 1, 'step_ratio': 1, 'episode_end': 1}
        

        self.context_df = pd.read_csv(context_path, index_col='context')
        self.context_features = ['urgency', 'importance', 'risk', 'distance_from_human', 'distance_from_object', 'speed', 'comfort', 
                                 'bumping_human', 'bumping_object', 'predictability']
        self.overwrite_contexts = overwrite_contexts
        self.data_augmentation = data_augmentation

        self.type_features = ['robot', 'goal', 'human', 'object', 'wall']
        self.geometric_features = ['x', 'y','sin_a', 'cos_a', 'vx', 'vy', 'acc_x', 'acc_y',
                                   'w', 'l', 'd_robot', 'th_pos', 'th_angle']

        self.all_features = self.type_features + self.geometric_features

        self.graphs = []
        self.metrics = []        
        self.labels = []
        self.slengths = []

        self.timestamp_threshold = timestamp_threshold

        nombre_carpeta = Path(self.data_list_file).stem
        self.carpeta_guardado = os.path.join(data_path, nombre_carpeta)
            
        super(SocNavHomoDataset, self).__init__(self.carpeta_guardado, transform, pre_transform, pre_filter)

        # Checking if the dataset is already charged, if not, it calls the process method. Otherwise, it loads the pt file

        if reload and os.path.exists(self.processed_paths[0]):
            print("Cargando datos previamente guardados")
            checkpoint = torch.load(self.processed_paths[0], weights_only=False)
            self.graphs = checkpoint['trajectories']
            self.metrics = checkpoint['metrics']
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
            traj_metrics = []
            for i in range(lenght):
                grafo, frame_metrics = self._json_to_graph(normalized_dict, i)
                trayectoria.append(grafo)
                traj_metrics.append(frame_metrics)

            traj_metrics = torch.stack(traj_metrics)
            self.graphs.append(trayectoria)
            self.metrics.append(traj_metrics)
            self.labels.append(torch.tensor([rating], dtype=torch.float32))
            self.slengths.append(torch.tensor(lenght, dtype=torch.long))

            if self.data_augmentation:
                cloned_trajectory = clone_sequence(trayectoria)
                t_data_mirrored = self.mirror_sequence(cloned_trajectory)
                self.graphs.append(t_data_mirrored)
                self.metrics.append(traj_metrics)
                self.labels.append(torch.tensor([rating], dtype=torch.float32))
                self.slengths.append(torch.tensor(lenght, dtype=torch.long))

            count += 1

            if count == limit:
                break

        # torch.save(
        #     {'trajectories': self.graphs, 'metrics': self.metrics, 'labels': self.labels, 'slength': self.slengths}, 
        #     self.processed_paths[0]
        # )


    def get_all_features(self):
        return self.all_features

    def get_context_features(self):
        return self.context_features

    def get_metrics_features(self):
        return self.metrics_features

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        data = self.graphs[idx]
        metrics = self.metrics[idx]
        label = self.labels[idx] 
        slength = self.slengths[idx] 

        return data, metrics, label, slength

    # --- MÉTODOS AUXILIARES ---

    def _json_to_graph(self, dict, index, full_conexo=True):


        num_node_features = len(self.all_features)
        # x = torch.zeros((num_nodes, num_node_features), dtype=torch.float32)

        entity_idx = {}
        graph_feats = []

        self.geometric_features = ['x', 'y','sin_a', 'cos_a', 'vx', 'vy', 'acc_x', 'acc_y',
                                   'w', 'l', 'd_robot', 'th_pos', 'th_angle']

        id = 0
        # robot node
        entity_idx['robot'] = [id]
        id += 1
        node_feats = torch.zeros(num_node_features, dtype=torch.float32)
        rx, ry = dict['robot']['x'][index], dict['robot']['y'][index]
        node_feats[self.all_features.index('robot')] = 1.
        node_feats[self.all_features.index('x')] = rx
        node_feats[self.all_features.index('y')] = ry
        node_feats[self.all_features.index('sin_a')] = math.sin(dict['robot']['a'][index])
        node_feats[self.all_features.index('cos_a')] = math.cos(dict['robot']['a'][index])
        node_feats[self.all_features.index('vx')] = dict['robot']['vx'][index]
        node_feats[self.all_features.index('vy')] = dict['robot']['vy'][index]
        node_feats[self.all_features.index('acc_x')] = dict['robot']['acc_x'][index]
        node_feats[self.all_features.index('acc_y')] = dict['robot']['acc_y'][index]
        node_feats[self.all_features.index('w')] = dict['robot']['w'][index]
        node_feats[self.all_features.index('l')] = dict['robot']['l'][index]

        graph_feats.append(node_feats)

        # goal node
        entity_idx['goal'] = [id]
        id += 1
        node_feats = torch.zeros(num_node_features, dtype=torch.float32)
        node_feats[self.all_features.index('goal')] = 1.
        node_feats[self.all_features.index('x')] = dict['goal']['x'][index]
        node_feats[self.all_features.index('y')] = dict['goal']['y'][index]
        node_feats[self.all_features.index('sin_a')] = math.sin(dict['goal']['a'][index])
        node_feats[self.all_features.index('cos_a')] = math.cos(dict['goal']['a'][index])
        node_feats[self.all_features.index('d_robot')] = dict['computed_metrics']['dist_to_goal_pos'][index]
        node_feats[self.all_features.index('th_pos')] = dict['goal']['th_p'][index]
        node_feats[self.all_features.index('th_angle')] = dict['goal']['th_a'][index]

        graph_feats.append(node_feats)

        # human nodes
        num_humans = dict['people']['x'].shape[1]
        people_exist = dict['people']['exists']
        entity_idx['human'] = []
        for i in range(num_humans):
            if people_exist[index, i].item():
                entity_idx['human'].append(id)
                id += 1
                node_feats = torch.zeros(num_node_features, dtype=torch.float32)
                px = dict['people']['x'][index, i].item()
                py = dict['people']['y'][index, i].item()
                pa = dict['people']['a'][index, i].item()
                dist_to_robot = math.sqrt((px-rx)**2+(py-ry)**2)
                node_feats[self.all_features.index('human')] = 1.
                node_feats[self.all_features.index('x')] = px
                node_feats[self.all_features.index('y')] = py
                node_feats[self.all_features.index('sin_a')] = math.sin(pa)
                node_feats[self.all_features.index('cos_a')] = math.cos(pa)
                node_feats[self.all_features.index('d_robot')] = dist_to_robot

                graph_feats.append(node_feats)

        # object nodes
        num_objects = dict['objects']['x'].shape[1]
        objects_exist = dict['objects']['exists']
        entity_idx['object'] = []        
        for i in range(num_objects):
            if objects_exist[index, i].item():
                entity_idx['object'].append(id)
                id += 1
                node_feats = torch.zeros(num_node_features, dtype=torch.float32)
                ox = dict['objects']['x'][index, i].item()
                oy = dict['objects']['y'][index, i].item()
                oa = dict['objects']['a'][index, i].item()
                dist_to_robot = math.sqrt((ox-rx)**2+(oy-ry)**2)
                node_feats[self.all_features.index('object')] = 1.
                node_feats[self.all_features.index('x')] = ox
                node_feats[self.all_features.index('y')] = oy
                node_feats[self.all_features.index('sin_a')] = math.sin(oa)
                node_feats[self.all_features.index('cos_a')] = math.cos(oa)
                node_feats[self.all_features.index('d_robot')] = dist_to_robot
                node_feats[self.all_features.index('w')] = dict['objects']['w'][index, i]
                node_feats[self.all_features.index('l')] = dict['objects']['l'][index, i]

                graph_feats.append(node_feats)

        # wall nodes
        walls = dict['walls']
        entity_idx['wall'] = []
        if walls is not None:
            raw_points = self._sample_walls(walls)
            for pt in raw_points:
                entity_idx['wall'].append(id)
                id += 1
                node_feats = torch.zeros(num_node_features, dtype=torch.float32)
                wx = pt[0].item()
                wy = pt[1].item()
                dist_to_robot = math.sqrt((wx-rx)**2+(wy-ry)**2)
                node_feats[self.all_features.index('wall')] = 1.
                node_feats[self.all_features.index('x')] = wx
                node_feats[self.all_features.index('y')] = wy
                node_feats[self.all_features.index('d_robot')] = dist_to_robot

                graph_feats.append(node_feats)

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

        metrics_frame = torch.cat((metrics_frame, context_tensor), dim=0).float()

        graph_feats = torch.stack(graph_feats, dim = 0)
        num_nodes = graph_feats.shape[0]

        list_N = list(range(1, num_nodes))
        list_0 = [0]*(num_nodes - 1)
        list_ALL = list(range(num_nodes))
        graph_edge_index = np.zeros((2, 2*(num_nodes-1)+num_nodes))
        graph_edge_index[0, :] = list_N + list_0 + list_ALL
        graph_edge_index[1, :] = list_0 + list_N + list_ALL
        graph_edge_index = torch.tensor(graph_edge_index, dtype=torch.long)


        data = Data(x=graph_feats, edge_index=graph_edge_index)

        return data, metrics_frame
    

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
    
    def mirror_sequence(self, sequence):
        new_sequence = sequence

        for frame in new_sequence:
            frame.x[:self.all_features.index('y')] = -frame.x[:self.all_features.index('y')]
            frame.x[:self.all_features.index('sin_a')] = -frame.x[:self.all_features.index('sin_a')]
            frame.x[:self.all_features.index('vy')] = -frame.x[:self.all_features.index('vy')]
            frame.x[:self.all_features.index('acc_y')] = -frame.x[:self.all_features.index('acc_y')]

        return new_sequence



def collate(batch):
    sequences, metrics, labels, sequence_lengths = zip(*batch)  

    flat_graphs = [frame for traj in sequences for frame in traj]

    batched_graphs = Batch.from_data_list(flat_graphs)   

    metrics_tensor = torch.cat(metrics, dim=0)  
    labels_tensor = torch.stack(labels)  
    slengths_tensor = torch.stack(sequence_lengths)


    return batched_graphs, metrics_tensor, labels_tensor, slengths_tensor

        

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