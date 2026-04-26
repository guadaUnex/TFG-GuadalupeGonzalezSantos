import os
import json
import torch
from tqdm import tqdm
from torch_geometric.data import Dataset, HeteroData
from transforms import GoalFrameTransform

class SocNavHeteroDataset(Dataset):
    def __init__(self, data_list_file, path='../../../dataset/labeled', transform=None, pre_transform=None, pre_filter=None):
        
        self.data_list_file = data_list_file

        with open(os.path.join(path, 'split', data_list_file), 'r') as f:
            self.file_names = f.read().splitlines()

        self.transformer = GoalFrameTransform(scale=10.0, v_max=2.0)
            
        super(SocNavHeteroDataset, self).__init__(path, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.file_names

    @property
    def processed_file_names(self):
        return [f"data_{name.replace('/', '_')}.pt" for name in self.file_names]
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'labeled')
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    def process(self):

        # Limites de pruebas
        limit = 50
        count = 0

        paths_to_process = list(zip(self.raw_paths, self.processed_paths))[:limit]

        for raw_path, processed_path in tqdm(zip(self.raw_paths, self.processed_paths), 
                                             total=len(self.raw_paths), 
                                             desc="Procesando JSONs"):            
            with open(raw_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            walls = json_data.get('walls', [])

            trayectoria = []

            for frame_data in json_data['sequence']:
                grafo_frame = self._json_to_heterodata(frame_data, walls)
                trayectoria.append(grafo_frame)

            torch.save(trayectoria, processed_path)

            count += 1

            if count == limit:
                break


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        return data

    # --- MÉTODOS AUXILIARES ---

    def _json_to_heterodata(self, frame, walls, full_conexo=False):
        data = HeteroData()
        s = self.transformer.scale  
        v = self.transformer.v_max

        # Goal node
        g = frame['goal']
        gx, gy, ga = torch.tensor(g['x']), torch.tensor(g['y']), torch.tensor(g['angle'])
        data['goal'].x = torch.tensor([[0.0, 0.0, 0, g['pos_threshold']/s, g['angle_threshold']/3.14]], dtype=torch.float)

        # Robot node
        r = frame['robot']
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

        # Scenario node
        shape_id = 1.0 if r['shape']['type'] == 'rectangle' else 0.0
        data['scenario'].x = torch.tensor([[shape_id, r['shape']['width']/s, r['shape']['length']/s]], dtype=torch.float)
       
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



# if __name__ == "__main__":
#     # Configuración de rutas
#     ruta_raiz = 'C:/Users/Usuario/Desktop/Universidad/TFG/dataset/labeled'
#     dataset = SocNavHeteroDataset(path=ruta_raiz, data_list_file='train_set_socnav3.txt')

#     print(f"\n🚀 Iniciando auditoría del Dataset. Total trayectorias: {len(dataset)}")

#     # 1. Intentamos cargar la primera trayectoria procesada
#     try:
#         trayectoria = dataset[0]
#         print(f"✅ Trayectoria 0 cargada correctamente ({len(trayectoria)} frames).")
#     except Exception as e:
#         print(f"❌ Error al cargar el primer elemento: {e}")
#         print("💡 Consejo: Borra la carpeta 'processed' para forzar un nuevo procesado.")
#         exit()

#     if isinstance(trayectoria, list) and len(trayectoria) > 0:
#         f0 = trayectoria[0]
        
#         print("\n--- 📝 TEST DE ESTRUCTURA (FRAME 0) ---")
        
#         # Test: Nodo Goal (Meta)
#         if 'goal' in f0.node_types:
#             g_x = f0['goal'].x
#             print(f"🎯 Meta: {g_x.shape} | Threshold pos: {g_x[0, 3]:.4f} | Threshold ang: {g_x[0, 4]:.4f}")
#             if g_x[0, 0] != 0 or g_x[0, 1] != 0:
#                 print("⚠️  Aviso: La meta no está en (0,0). Revisa transform_pose.")

#         # Test: Nodo Robot
#         r_x = f0['robot'].x
#         print(f"🤖 Robot: {r_x.shape} | Pos: {r_x[0, :2].tolist()} | Vel: {r_x[0, 3:5].tolist()}")

#         # Test: Nodo Scenario (Dimensiones normalizadas)
#         s_x = f0['scenario'].x
#         print(f"🏢 Escenario: {s_x.shape} | ShapeID: {s_x[0,0]} | Size Robot Norm: {s_x[0, 1:]}")

#         print("\n--- 🔗 TEST DE ARISTAS (RELACIONES) ---")
        
#         # Test: Enlace Crítico Robot -> Goal
#         rel_target = ('robot', 'targets', 'goal')
#         if rel_target in f0.edge_types:
#             e_idx = f0[rel_target].edge_index
#             e_attr = f0[rel_target].edge_attr
#             print(f"🔗 Robot-Goal: {e_idx.shape} | Atributos [Dist, Thres]: {e_attr[0].tolist()}")
#         else:
#             print("❌ Error: No se encontró el enlace 'targets' entre Robot y Goal.")

#         # Test: Enlaces de Proximidad (Humanos/Paredes)
#         for rel in f0.edge_types:
#             if 'near_to' in rel[1]:
#                 num_edges = f0[rel].edge_index.shape[1]
#                 print(f"📡 Relación {rel}: {num_edges} enlaces detectados.")

#         print("\n--- 📏 TEST DE RANGOS (NORMALIZACIÓN) ---")
#         # Comprobar que los valores no explotan (deberían estar cerca de [-1, 1])
#         all_node_features = torch.cat([f0[nt].x.flatten() for nt in f0.node_types if f0[nt].x.numel() > 0])
#         max_val = all_node_features.max().item()
#         min_val = all_node_features.min().item()
#         print(f"📊 Rango de valores en el grafo: [{min_val:.2f}, {max_val:.2f}]")
        
#         if max_val > 5.0:
#             print("⚠️  Alerta: Tienes valores muy altos. Revisa las divisiones por 's' o 'v'.")

#     else:
#         print("❌ Error: El dataset devolvió una lista vacía o un objeto no válido.")
        