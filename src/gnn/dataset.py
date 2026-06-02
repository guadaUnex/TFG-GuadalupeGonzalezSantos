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
from transforms import GoalFrameTransform

from data_conversions import clone_sequence
from data_mirroring import mirror_sequence

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
                   overwrite_contexts = '', transform=None, pre_transform=None, pre_filter=None, timestamp_threshold = 0.3, reload=False, data_augmentation = False):
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
        self.overwrite_contexts = overwrite_contexts
        self.data_augmentation = data_augmentation
        
        self.all_features = {
            # success, min dist to human, context vars.
            'scenario': 2 + len(self.context_features), 
            'goal': 5,
            'robot': 10,
            'human': 5,
            'object': 7,
            'wall': 3
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
        limit = 25
        count = 0

        # print(self.raw_paths)

        for raw_path in tqdm(self.raw_paths, total=len(self.raw_paths), desc="Procesando JSONs"):            
            with open(raw_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            walls = json_data.get('walls', [])
            context_desc = json_data.get('context_description', self.overwrite_contexts)
            rating = json_data.get('label',0.)
            lenght = 0

            context = list(self.context_df.loc[context_desc.rstrip()].to_dict().values())
            self.transformer.normalize_context(context)

            trayectoria = []
            prev_timestamp = -np.inf #json_data['sequence'][0]['timestamp']
            last_i = len(json_data['sequence'])-1
            for i, frame_data in enumerate(json_data['sequence']):
                if (frame_data['timestamp'] - prev_timestamp) > self.timestamp_threshold or i == last_i:
                    grafo_frame = self._json_to_heterodata(frame_data, walls, context)
                    trayectoria.append(grafo_frame)
                    prev_timestamp = frame_data['timestamp']
                    lenght += 1

            self.dataset.append(trayectoria)
            self.labels.append([rating])
            self.slengths.append(lenght)

            if self.data_augmentation:
                cloned_trajectory = clone_sequence(trayectoria)
                t_data_mirrored = mirror_sequence(cloned_trajectory)
                self.dataset.append(t_data_mirrored)
                self.labels.append([rating])
                self.slengths.append(lenght)

            # count += 1

            # if count == limit:
            #     break

        torch.save(
            {'trajectories': self.dataset, 'labels': self.labels, 'slength': self.slengths}, 
            self.processed_paths[0]
        )


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
                ('goal', 'near_to', 'human'), 
                ('human', 'near_to', 'goal'), 
                ('goal', 'near_to', 'object'), 
                ('object', 'near_to', 'goal'), 
                ('goal', 'near_to', 'wall'), 
                ('wall', 'near_to', 'goal'), 
                ('goal', 'near_to', 'robot'), 
                ('robot', 'near_to', 'goal'), 
                ('robot', 'near_to', 'wall'), 
                ('wall', 'near_to', 'robot'), 
                ('robot', 'near_to', 'human'), 
                ('human', 'near_to', 'robot'), 
                ('robot', 'near_to', 'object'), 
                ('object', 'near_to', 'robot'), 
                ('human', 'near_to', 'human'), 
                ('human', 'near_to', 'object'), 
                ('object', 'near_to', 'human'), 
                ('human', 'near_to', 'wall'), 
                ('wall', 'near_to', 'human'), 
                ('object', 'near_to', 'object'),
                ('object', 'near_to', 'wall'),
                ('wall', 'near_to', 'object')                
            ]
        )
        return metadata

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
        g = frame['goal']

        dist_to_goal_pos = math.sqrt((r['x']-g['x'])**2 + (r['y']-g['y'])**2) 
        angle_diff = r['angle']-g['angle']
        dist_to_goal_angle = abs(math.atan2(math.sin(angle_diff), math.cos(angle_diff)))
        success = 1. if (dist_to_goal_pos<g['pos_threshold']+0.1) and (dist_to_goal_angle<g['angle_threshold']) else 0.

        # Goal node
        gx, gy, ga = torch.tensor(g['x']), torch.tensor(g['y']), torch.tensor(g['angle'])
        data['goal'].x = torch.tensor([[0.0, 0.0, 0, g['pos_threshold']/s, g['angle_threshold']/3.14]], dtype=torch.float)

        # Robot node
        rx, ry, ra = self.transformer.transform_pose(r['x'], r['y'], r['angle'], gx, gy, ga)
        nvx, nvy, nva = self.transformer.transform_velocity(r['speed_x'], r['speed_y'], r['speed_a'], ga)
        dist_to_goal = math.sqrt(rx**2+ry**2)
        data['robot'].x = torch.tensor([[rx, ry, math.sin(ra), math.cos(ra), r['shape']['width']/s, r['shape']['length']/s, nvx, nvy, nva, dist_to_goal]], dtype=torch.float)

        # People nodes
        min_HR_dist = 1.
        people = frame.get('people', [])
        p_list = []
        for p in people:
            nx, ny, na = self.transformer.transform_pose(p['x'], p['y'], p['angle'], gx, gy, ga)
            dist_to_robot = math.sqrt((nx-rx)**2+(ny-ry)**2)
            min_HR_dist = min(dist_to_robot, min_HR_dist)
            p_list.append([nx, ny, math.sin(na), math.cos(na), dist_to_robot])
        data['human'].x = torch.tensor(p_list, dtype=torch.float) if p_list else torch.empty((0, 5))

        # Object nodes
        objects = frame.get('objects', [])
        o_list = []
        for o in objects:
            nx, ny, na = self.transformer.transform_pose(o['x'], o['y'], o['angle'], gx, gy, ga)
            dist_to_robot = math.sqrt((nx-rx)**2+(ny-ry)**2)
            o_list.append([nx, ny, math.sin(na), math.cos(na), o['shape']['width']/s, o['shape']['length']/s, dist_to_robot])
        data['object'].x = torch.tensor(o_list, dtype=torch.float) if o_list else torch.empty((0, 7))

        # Walls nodes
        if walls:
            raw_points = self._sample_walls(walls)
            w_list = []
            for pt in raw_points:
                wx, wy, _ = self.transformer.transform_pose(pt[0].item(), pt[1].item(), 0.0, gx, gy, ga)
                dist_to_robot = math.sqrt((wx-rx)**2+(wy-ry)**2)
                w_list.append([wx, wy, dist_to_robot])
            data['wall'].x = torch.tensor(w_list, dtype=torch.float)
        else:
            data['wall'].x = torch.empty((0, 3))

        # Scenario node
        scenario_features = [success, min_HR_dist]+ context
        data['scenario'].x = torch.tensor([scenario_features], dtype=torch.float)

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

    def _create_edges(self, data, full_conexo=False, dist_threshold=0.1):
        
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
            
            goal_threshold = data['goal'].x[0, 3]
            edge_attr_rg = torch.tensor([[dist_rg, goal_threshold]], dtype=torch.float)

            data['robot', 'targets', 'goal'].edge_index = edge_index_rg
            data['robot', 'targets', 'goal'].edge_attr = edge_attr_rg
            data['goal', 'assigned_to', 'goal'].edge_index = edge_index_rg
            data['goal', 'assigned_to', 'goal'].edge_attr = edge_attr_rg


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
    
        for i, type_a in enumerate(spatial_nodes):
            for type_b in spatial_nodes[i:]:

                if type_a == type_b == 'wall' or type_a == type_b == 'robot' or type_a == type_b == 'goal':
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

                # mask = mask.long()
                
                # if type_a == type_b:
                #     mask = mask & (~torch.eye(pos_a.size(0), dtype=torch.bool))
                
                edge_index = mask.nonzero(as_tuple=False).t()
                
                if edge_index.numel() > 0:
                    edge_values = dists[mask].unsqueeze(1)
                    rel_name = (type_a, 'near_to', type_b)
                    data[rel_name].edge_index = edge_index.long()
                    data[rel_name].edge_attr = edge_values
                    
                    if type_a != type_b:
                        rev_rel = (type_b, 'near_to', type_a)
                        data[rev_rel].edge_index = edge_index.flip(0).long()
                        data[rev_rel].edge_attr = edge_values                     

        edge_types_metadata = self.get_metadata()[1]  

        for edge_type in edge_types_metadata:
            if edge_type not in data.edge_types:
                data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
                data[edge_type].edge_attr = torch.empty((0), dtype=torch.float)
        return data


def collate(batch):
    sequences, labels, sequence_lengths = zip(*batch)  


    flat_graphs = [frame for traj in sequences for frame in traj]

    batched_graphs = Batch.from_data_list(flat_graphs)   


    labels_tensor = torch.stack(labels)  
    slengths_tensor = torch.stack(sequence_lengths)


    return batched_graphs, labels_tensor, slengths_tensor

        

# if __name__ == "__main__":
#     print("🚀 Lanzando suite de prueba para verificación de Mirroring (Data Augmentation)...")

#     # --- Configuración de rutas de pruebas ---
#     DATA_PATH = '../../../dataset/labeled/labeled'
#     CONTEXT_PATH = '../../../dataset/contexts/anthropic_claude_context.csv'
#     SPLIT_PRUEBA = '../split/train_set_socnav3.txt' # Asegúrate de que este archivo existe para la prueba

#     # 1. Instanciamos el Dataset FORZANDO el Data Augmentation y recreando el procesamiento
#     print("\n[PASO 1] Instanciando dataset con data_augmentation=True...")
#     try:
#         dataset = SocNavHeteroDataset(
#             data_list_file=SPLIT_PRUEBA,
#             data_path=DATA_PATH,
#             context_path=CONTEXT_PATH,
#             timestamp_threshold=0.3,
#             data_augmentation=True,   # 🌟 ACTIVAMOS EL AUMENTO
#             reload=False              # 🌟 FORZAMOS REPROCESAMIENTO para ver el espejo en acción
#         )
#     except Exception as e:
#         print(f"❌ Error al instanciar el dataset: {e}")
#         print("Asegúrate de que las rutas relativas sean correctas desde la carpeta donde ejecutas este script.")
#         exit(1)

#     print(f"\n[PASO 2] Dataset cargado. Muestras totales generadas: {len(dataset)}")
    
#     if len(dataset) < 2:
#         print("❌ Error: Se necesitan al menos 2 muestras en el dataset para validar el reflejo.")
#         exit(1)

#     # 2. Extracción de la pareja de prueba
#     # Debido a tu lógica en 'process()', la muestra 0 es la Original y la muestra 1 es su versión Espejo.
#     print("\n[PASO 3] Extrayendo pareja simétrica (Muestra 0 [Original] vs Muestra 1 [Espejo])...")
#     traj_orig, label_orig, len_orig = dataset[0]
#     traj_mirr, label_mirr, len_mirr = dataset[1]

#     # --- TEST 1: Longitudes y Etiquetas ---
#     print("\n🔍 --- TEST 1: Consistencia de Metadatos y Estructura temporal ---")
#     print(f"• Longitud temporal original: {len_orig.item()} | Longitud espejo: {len_mirr.item()}")
#     print(f"• Etiqueta original: {label_orig.item()} | Etiqueta espejo: {label_mirr.item()}")
    
#     assert len_orig.item() == len_mirr.item(), "❌ FALLO: Las longitudes temporales difieren."
#     assert len(traj_orig) == len(traj_mirr), "❌ FALLO: El conteo de grafos (frames) internos difiere."
#     assert torch.equal(label_orig, label_orig), "❌ FALLO: Las etiquetas sufrieron alteraciones."
#     print("✅ TEST 1 COMPLETADO: Estructura y dimensiones idénticas.")

#     # --- TEST 2: Inversión Geométrica de Tensores (Matemática de Nodos) ---
#     print("\n🔍 --- TEST 2: Verificación analítica de los signos en el Eje Y ---")
    
#     # Analizamos el primer frame de la secuencia para verificar los nodos
#     frame_orig = traj_orig[0]
#     frame_mirr = traj_mirr[0]
    
#     # 📝 Verificación del Robot
#     # Estructura de data['robot'].x según tu código: [rx, ry, sin(ra), cos(ra), w, l, nvx, nvy, nva, dist]
#     # Deben invertirse: ry (index 1), sin(ra) (index 2), nvy (index 7), nva (index 8)
#     r_orig = frame_orig['robot'].x[0]
#     r_mirr = frame_mirr['robot'].x[0]
    
#     print("\n🤖 Datos del Robot (Primer Frame):")
#     print(f"  • Posición Y -> Original: {r_orig[1].item():.4f} | Espejo: {r_mirr[1].item():.4f}")
#     print(f"  • Seno Ángulo -> Original: {r_orig[2].item():.4f} | Espejo: {r_mirr[2].item():.4f}")
#     print(f"  • Vel. Lineal Y -> Original: {r_orig[7].item():.4f} | Espejo: {r_mirr[7].item():.4f}")
#     print(f"  • Vel. Angular -> Original: {r_orig[8].item():.4f} | Espejo: {r_mirr[8].item():.4f}")
    
#     # Comprobaciones matemáticas tolerantes a precisión flotante
#     assert torch.allclose(r_orig[1], -r_mirr[1]), "❌ FALLO: La posición 'y' del robot no se invirtió correctamente."
#     assert torch.allclose(r_orig[2], -r_mirr[2]), "❌ FALLO: El ángulo (sin) del robot no se invirtió correctamente."
#     assert torch.allclose(r_orig[7], -r_mirr[7]), "❌ FALLO: La velocidad 'vy' del robot no se invirtió correctamente."
#     assert torch.allclose(r_orig[8], -r_mirr[8]), "❌ FALLO: La velocidad angular 'va' del robot no se invirtió correctamente."
#     print("  => OK: El robot se refleja de manera simétrica perfecta.")

#     # 📝 Verificación de Humanos (Si existen en el escenario)
#     # Estructura: [nx, ny, sin(na), cos(na), dist_to_robot]
#     if frame_orig['human'].x.size(0) > 0:
#         h_orig = frame_orig['human'].x[0]
#         h_mirr = frame_mirr['human'].x[0]
#         print("\n👥 Datos del Humano 0 (Primer Frame):")
#         print(f"  • Posición Y -> Original: {h_orig[1].item():.4f} | Espejo: {h_mirr[1].item():.4f}")
#         print(f"  • Seno Ángulo -> Original: {h_orig[2].item():.4f} | Espejo: {h_mirr[2].item():.4f}")
        
#         assert torch.allclose(h_orig[1], -h_mirr[1]), "❌ FALLO: La posición 'y' del humano no se invirtió."
#         assert torch.allclose(h_orig[2], -h_mirr[2]), "❌ FALLO: El ángulo 'sin(na)' del humano no se invirtió."
#         print("  => OK: Los seres humanos se reflejan correctamente.")
#     else:
#         print("\n👥 Datos de Humanos: No hay humanos en este frame de prueba (saltando aserción).")

#     # 📝 Verificación de Muros / Paredes (Si existen)
#     # Estructura: [wx, wy, dist_to_robot]
#     if frame_orig['wall'].x.size(0) > 0:
#         w_orig = frame_orig['wall'].x[0]
#         w_mirr = frame_mirr['wall'].x[0]
#         print("\n🧱 Datos del Muro 0 (Primer Frame):")
#         print(f"  • Posición Y -> Original: {w_orig[1].item():.4f} | Espejo: {w_mirr[1].item():.4f}")
        
#         assert torch.allclose(w_orig[1], -w_mirr[1]), "❌ FALLO: La posición 'y' de la pared no se invirtió."
#         print("  => OK: Los puntos de los muros se reflejan correctamente.")

#     print("\n🔍 --- TEST 3: Verificación de no-corrupción (Aislamiento de memoria) ---")
#     # Si modificaste la función para que no use dobles clones ni punteros cruzados,
#     # el segundo elemento del dataset (índice 1) debe tener el signo cambiado, pero el elemento 0 debe seguir intacto.
#     assert not torch.allclose(r_orig[1], r_mirr[1]) or r_orig[1] == 0, "❌ FALLO: ¡Los datos originales y espejo son idénticos! Punteros cruzados en memoria."
#     print("✅ TEST 3 COMPLETADO: Aislamiento de memoria verificado exitosamente.")

#     print(f"\n{'-'*60}")
#     print("🎉 ¡ENHORABUENA! El sistema de Mirroring funciona a la perfección.")
#     print("Los datos se clonan, aíslan y reflejan respetando las leyes geométricas de SocNav.")
#     print(f"{'-'*60}\n")