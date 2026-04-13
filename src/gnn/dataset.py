import os
import json
import torch
from tqdm import tqdm
from torch_geometric.data import Dataset, HeteroData


class SocNavHeteroDataset(Dataset):
    def __init__(self, data_list_file, path='../../../dataset/labeled', transform=None, pre_transform=None, pre_filter=None):
        
        self.data_list_file = data_list_file

        with open(os.path.join(path, 'split', data_list_file), 'r') as f:
            self.file_names = f.read().splitlines()
            
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


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        return data

    # --- MÉTODOS AUXILIARES ---

    def _json_to_heterodata(self, frame, walls):
        data = HeteroData()

        # Robot node
        r = frame['robot']
        data['robot'].x = torch.tensor([[r['x'], r['y'], r['angle'], r['speed_x'], r['speed_y'], r['speed_a']]], dtype=torch.float)

        # People nodes
        people = frame.get('people', [])
        data['human'].x = torch.tensor([[p['x'], p['y'], p['angle']] for p in people], dtype=torch.float)

        # Object nodes
        objects = frame.get('objects', [])
        data['object'].x = torch.tensor([[o['x'], o['y'], o['angle'], o['shape']['width'], o['shape']['length']] for o in objects], dtype=torch.float)

        # Goal node
        g = frame['goal']
        data['goal'].x = torch.tensor([[g['x'], g['y'], g['angle']]], dtype=torch.float)

        # Walls nodes
        if walls:
            data['wall'].x = self._sample_walls(walls)

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

    def _create_edges(self, data):
        # robot interactions

        robot_idx = torch.tensor([0], dtype=torch.long)

        # robot <-> people

        if 'human' in data.node_types and data['human'].x.shape[0] > 0:
            num_people = data['human'].x.shape[0]
            human_indices = torch.arange(num_people, dtype=torch.long)
            # robot sees human
            edge_index_r_h = torch.stack([robot_idx.repeat(num_people), human_indices], dim=0)
            data['robot', 'sees', 'human'].edge_index = edge_index_r_h

        if 'object' in data.node_types and data['object'].x.shape[0] > 0:
            num_o = data['object'].x.shape[0]
            o_indices = torch.arange(num_o, dtype=torch.long)
            # robot sees object
            data['robot', 'sees', 'object'].edge_index = torch.stack([robot_idx.repeat(num_o), o_indices], dim=0)

        if 'goal' in data.node_types and data['goal'].x.shape[0] > 0:
            num_g = data['goal'].x.shape[0]
            g_indices = torch.arange(num_g, dtype=torch.long)
            data['robot', 'targets', 'goal'].edge_index = torch.stack([robot_idx.repeat(num_g), g_indices], dim=0)


""" if __name__ == "__main__":
    ruta_raiz = 'C:/Users/Usuario/Desktop/Universidad/TFG/dataset/labeled'
    dataset = SocNavHeteroDataset(path=ruta_raiz, data_list_file='train_set_socnav3.txt')

    print(f"\n✅ Dataset cargado. Total de trayectorias: {len(dataset)}")

    # 1. Cogemos la primera trayectoria
    trayectoria = dataset[0]

    # 2. Comprobamos que es una lista
    if isinstance(trayectoria, list):
        print(f"📦 Trayectoria 0: ES UNA LISTA")
        print(f"⏱️  Número de frames (grafos) detectados: {len(trayectoria)}")
        
        # 3. Inspeccionamos el primer y el último frame para ver si cambian las posiciones
        if len(trayectoria) > 1:
            f0 = trayectoria[0]
            f_final = trayectoria[-1]
            
            print("\n--- Comparativa Temporal ---")
            print(f"Posición Robot Frame 0:     {f0['robot'].x[0, :2].tolist()}")
            print(f"Posición Robot Último Frame: {f_final['robot'].x[0, :2].tolist()}")
            
            # Verificamos que la estructura interna sigue siendo HeteroData
            print(f"\nEstructura interna del frame 0:\n{f0}")
    else:
        print("❌ Error: El elemento recuperado NO es una lista. Revisa el torch.save.") """
        