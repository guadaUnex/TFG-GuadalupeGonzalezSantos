import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
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
                   transform=None, pre_transform=None, pre_filter=None, timestamp_threshold = 0.3, reload=False):
        """Inicializa los parámetros del dataset y gestiona la carga de la caché procesada."""

        print("Iniciando creación del siguiente dataset: ", data_list_file)
        
        self.data_list_file = data_list_file

        self.trajectories_dir = data_path

        # Reading the dataset file

        with open(os.path.join(data_path, data_list_file), 'r') as f:
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
        else:
            self.process()


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
        limit = 100
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
                if (frame_data['timestamp'] - prev_timestamp) > self.timestamp_threshold:
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

        torch.save(
            {'trajectories': self.dataset, 'labels': self.labels, 'slength': self.slengths}, 
            self.processed_paths[0]
        )


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
    

    def _sample_walls(self, walls, dist_points=2.):
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

        # self edges
        for t in node_types:
            num_nodes_of_type = data[t].x.size(0)
            if True: #num_nodes_of_type>0:
                edge_index_self = torch.arange(num_nodes_of_type, dtype=torch.long)
                data[t, 'self', t].edge_index = edge_index_self

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
                if True: #num_nodes_of_type > 0:
                    row = torch.arange(num_nodes_of_type, dtype=torch.long)
                    col = torch.zeros(num_nodes_of_type, dtype=torch.long)
                    edge_index_in_scenario = torch.stack([row, col], dim=0)
                    
                    data[t, 'in', 'scenario'].edge_index = edge_index_in_scenario

        # spatial_nodes = [t for t in node_types if t not in ['scenario']]
    
        # for i, type_a in enumerate(spatial_nodes):
        #     for type_b in spatial_nodes[i:]:

        #         if type_a == 'wall' and type_b == 'wall':
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
                        
        return data


def collate(batch):
    sequences, labels, sequence_lengths = zip(*batch)  

    # print("Prueba etiquetas:", labels[:10])

    node_names = ['scenario', 'goal', 'robot', 'human', 'object', 'wall']
    edge_names = [('robot', 'targets', 'goal'), ('scenario', 'in', 'scenario'), ('goal', 'in', 'scenario'), ('robot', 'in', 'scenario'), ('human', 'in', 'scenario'), ('object', 'in', 'scenario'), ('wall', 'in', 'scenario')]

    flat_graphs = [frame for traj in sequences for frame in traj]
    # print(flat_graphs[0].x_dict)
    # exit()
    # for graph in flat_graphs:
    #     for edge in edge_names:
    #         src = edge[0]
    #         dst = edge[2]
    #         if src in graph.x_dict.keys() and dst in graph.x_dict.keys():
    #         # print(src, dst)
    #         # print(graph[src])
    #             len_src = graph.x_dict[src].shape[0]
    #             len_dst = graph.x_dict[dst].shape[0]
    #             indices = graph.edge_index_dict[edge]
    #             for i in range(indices.shape[1]):
    #                 i_src = indices[0,i]
    #                 i_dst = indices[1,i]
    #                 if i_src >= len_src or i_dst>=len_dst:
    #                     print('error in indices', 'i_src', i_src, 'len_src', len_src, 
    #                         'i_dst', i_dst, 'len_dst', len_dst)
    #                     exit()
            # else:
            #     print(src,dst)

    batched_graphs = Batch.from_data_list(flat_graphs)   
    # for k, t in batched_graphs.x_dict.items():
    #     print(k, t.shape)
    # print(batched_graphs.edge_index_dict) 

    labels_tensor = torch.stack(labels)  
    slengths_tensor = torch.stack(sequence_lengths)

    # print("Prueba etiquetas 2:", labels_tensor[:10])

    return batched_graphs, labels_tensor, slengths_tensor

        

# if __name__ == "__main__":
#     import os
#     from pathlib import Path

#     # --- CONFIGURACIÓN DE RUTAS ---
#     DATA_PATH = '../../../dataset/labeled'
#     CONTEXT_PATH = '../../../dataset/contexts/anthropic_claude_context.csv'
    
#     # Splits oficiales de tu TFG
#     splits_a_cargar = ['train_set_socnav3.txt', 'val_set_socnav3.txt', 'test_set_socnav3.txt']
    
#     print("🚀 Iniciando el gestor de Datasets de SocNavHetero...")
    
#     # Diccionario para guardar tus datasets en memoria si quisieras usarlos más adelante
#     datasets_en_memoria = {}

#     for split_file in splits_a_cargar:
#         nombre_split = Path(split_file).stem
#         print(f"\n{'-'*60}")
#         print(f"📂 Preparando split: {nombre_split.upper()}")
#         print(f"{'-'*60}")
        
#         # Comprobamos de antemano si el archivo .pt ya existe para darte un feedback claro
#         ruta_esperada_pt = os.path.join(DATA_PATH, "loading_files", nombre_split, "dataset.pt")
#         if os.path.exists(ruta_esperada_pt):
#             print(f"ℹ️  Caché detectada en: {ruta_esperada_pt}")
#             print(f"⏳ Cargando datos directamente desde el fichero .pt (esto será ultra rápido)...")
#         else:
#             print(f"⚠️  No se encontró caché previa para el split '{nombre_split}'.")
#             print(f"⚙️  Iniciando procesamiento completo de los archivos JSON...")

#         try:
#             # Instanciamos el dataset. Aquí PyG toma el control:
#             # - Si el .pt existe: se salta 'process()' y va directo a tu constructor.
#             # - Si el .pt NO existe: ejecuta 'process()', guarda el archivo y luego construye el objeto.
#             dataset = SocNavHeteroDataset(
#                 data_list_file=split_file,
#                 data_path=DATA_PATH,
#                 context_path=CONTEXT_PATH,
#                 timestamp_threshold=0.3
#             )
            
#             # Guardamos la instancia en el diccionario por si necesitas usarla en este script
#             datasets_en_memoria[nombre_split] = dataset
            
#             print(f"✅ ¡Split '{nombre_split}' listo! Total de trayectorias en memoria: {len(dataset)}")
            
#             # --- PRUEBA DE INTEGRIDAD DE LOS DATOS ---
#             # Si el dataset tiene elementos, inspeccionamos el primero para verificar que todo cuadre
#             if len(dataset) > 0:
#                 primer_elemento = dataset[0]  # Esto ejecuta tu método get(0)
#                 trayectoria, label, slength = primer_elemento
                
#                 print(f"📊 Verificación del primer elemento:")
#                 print(f"   • Número de frames en la secuencia: {len(trayectoria)} (slength original: {slength.item()})")
#                 print(f"   • Tipo de dato del primer frame: {type(trayectoria[0]).__name__}")
#                 print(f"   • Etiqueta asociada (label): {label.item()}")
            
#         except FileNotFoundError as e:
#             print(f"❌ Error: Comprueba que el archivo de split '{split_file}' exista en '{DATA_PATH}/split/'")
#             print(f"   Detalle del error: {e}")
#         except Exception as e:
#             print(f"❌ Error inesperado procesando el split '{split_file}': {e}")

#     print(f"\n{'-'*60}")
#     print("🎯 Proceso de inicialización finalizado.")
#     print(f"{'-'*60}")