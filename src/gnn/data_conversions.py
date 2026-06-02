import torch
import copy
from torch_geometric.data import HeteroData

def clone_sequence(trayectoria):
    nueva_trayectoria = []

    for frame in trayectoria:
        nuevo_grafo = HeteroData()
        
        for node_type in frame.node_types:
            if hasattr(frame[node_type], 'x') and frame[node_type].x is not None:
                nuevo_grafo[node_type].x = frame[node_type].x.clone()
                
        for edge_type in frame.edge_types:
            if hasattr(frame[edge_type], 'edge_index') and frame[edge_type].edge_index is not None:
                nuevo_grafo[edge_type].edge_index = frame[edge_type].edge_index.clone()
            
            if hasattr(frame[edge_type], 'edge_attr') and frame[edge_type].edge_attr is not None:
                nuevo_grafo[edge_type].edge_attr = frame[edge_type].edge_attr.clone()

        nueva_trayectoria.append(nuevo_grafo)

    return nueva_trayectoria
