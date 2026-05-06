# ----------------------------------------------------------------

# Codigo de ejemplo dado por gemini, es solo a modo de orientación

# ----------------------------------------------------------------

import os
import json
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from usefulCode.baseline.data_conversions import sequence_to_tensor, clone_sequence
from usefulCode.baseline.data_mirroring import mirror_tDic_sequence
from usefulCode.baseline.data_normalization import tensor_transform_to_goal_fr

class Dataset(Dataset):
    def __init__(self, data_file, contextQ_file, path = '../dataset', limit= -1,  frame_threshold = 0.1, label_exists = True, 
                 overwrite_context = False, data_augmentation = False, load_time_augmentation = True, reload = False):
        self.data = []
        self.labels = []
        self.path = path
        self.data_file = data_file

        if type(data_file) is str:
            DA = '_DA_' if data_augmentation and load_time_augmentation else ''
            self.reload_fname = '.'.join(data_file.split('.')[:-1]) + '_' + contextQ_file.split('/')[-1].split('.csv')[0] + DA + '.pytorch'
        else:
            self.reload_fname = ''
            reload = False
        self.label_exists = label_exists
        self.limit = limit
        self.overwrite_contexts = overwrite_context
        self.frame_threshold = frame_threshold
        self.data_augmentation = data_augmentation
        self.load_time_augmentation = load_time_augmentation
        self.context_df = pd.read_csv(contextQ_file, index_col='context') 

        self.robot_features = ['robot_x', 'robot_y', 'robot_a', 'speed_x', 'speed_y', 'speed_a', 'acceleration_x', 'acceleration_y']

        self.context_features = ['urgency', 'importance', 'risk', 'distance_from_human', 'distance_from_object', 'speed', 'comfort', 
                                 'bumping_human', 'bumping_object', 'predictability']
        self.goal_features = ['goal_pos_threshold', 'goal_angle_threshold']
        
        self.all_features = self.robot_features + self.metrics_features + self.goal_features + self.context_features

        self.MAX_TTC = 10
        self.MAX_LSPEED = 2

        self.max_metric_values = {'robot_x': 10, 'robot_y': 10, 'robot_a': np.pi, 'speed_x': self.MAX_LSPEED, 
                            'speed_y': self.MAX_LSPEED, 'speed_a': np.pi, 'acceleration_x': 3, 
                            'acceleration_y': 3, 'success': 1, 'hum_exists': 1, 'wall_exists': 1,
                            'dist_nearest_hum': 10, 'dist_nearest_obj': 10, 'dist_wall': 10, 'dist_goal': 10,
                            'hum_collision_flag': 1, 'object_collision_flag': 1, 'wall_collision_flag': 1, 
                            'social_space_intrusionA': 1, 'social_space_intrusionB': 1, 'social_space_intrusionC': 1,
                            'num_near_humansA': 10, 'num_near_humansB': 10, 'num_near_humansC': 10, 
                            'min_time_to_collision': self.MAX_TTC, 'max_fear': 10, 'max_panic': 10,
                            'num_near_humansA2': 100, 'num_near_humansB2': 100, 'num_near_humansC2': 100,
                            'min_time_to_collision2': self.MAX_TTC**2,
                            'global_dist_nearest_hum': 10, 'path_efficiency_ratio': 1, 'step_ratio': 1, 'episode_end': 1,
                            'goal_pos_threshold': 10, 'goal_angle_threshold': np.pi}

        for var in self.context_features:
            self.max_metric_values[var] = 100

        if reload is True:
            if os.path.exists(self.reload_fname):
                loaded = torch.load(self.reload_fname)
                self.data = loaded['data']
                self.labels = loaded['labels']
                print("number of trajectories for ", self.data_file, len(self.data))
                return


        if type(self.data_file) is str and self.data_file.endswith('.txt'):
            print(self.data_file)
            with open(self.data_file) as set_file:
                ds_files = set_file.read().splitlines()

            print("number of files for ", self.data_file, len(ds_files))
        elif type(self.data_file) is str and self.data_file.endswith('.json'):
            ds_files = [self.data_file]
        elif type(self.data_file) is list:
            ds_files = self.data_file

        self.ds_files = ds_files
        for i, filename in enumerate(ds_files):
            if filename.endswith('.json'):
                file_path = os.path.join(self.path, filename)
                with open(file_path, 'r', encoding="utf-8") as f:
                    try:
                        t_data = json.load(f)
                    except:
                        print("FileName :", file_path)


                    if 'context_description' in t_data.keys(): #not self.overwrite_contexts:
                        context_desc = t_data['context_description']
                    else:
                        context_desc = self.overwrite_contexts
                    context = self.context_df.loc[context_desc.rstrip()].to_dict()

                    tensor_dict = sequence_to_tensor(t_data, self.frame_threshold, context)
                    tensor_dict_to_goal = tensor_transform_to_goal_fr(tensor_dict)
                    if self.label_exists and 'label' in t_data: 
                        rating = t_data['label']
                    else:
                        rating = 0.0
                    self.data.append(tensor_dict_to_goal)
                    self.labels.append(rating)

                    if self.data_augmentation and self.load_time_augmentation:
                        tensor_dict_clone = clone_sequence(tensor_dict_to_goal)
                        t_data_mirrored = mirror_tDic_sequence(tensor_dict_clone)
                        self.data.append(t_data_mirrored)
                        self.labels.append(rating)

                    
            if i%1000 == 0:
                print(i)
            if i + 1 >= self.limit and self.limit > 0:
                print('Stop including more samples to speed up dataset loading')
                break
        if reload:
            torch.save({
                'data': self.data,
                'labels': self.labels
                }, self.reload_fname)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 1. Cargar el archivo JSON crudo
        file_path = os.path.join(self.json_dir, self.file_list[idx])
        with open(file_path, 'r') as f:
            raw_data = json.load(f)

        # 2. Convertir a diccionario de tensores usando tu código base
        # Esto calcula velocidades, TTC, pánico, etc.
        tensor_dict = sequence_to_tensor(raw_data, self.frame_threshold, self.context)

        # 3. Extraer el label (la nota que le puso el humano a la trayectoria)
        # Asumiendo que el JSON tiene un campo 'score' o similar
        label = torch.tensor(raw_data.get('score', 0.5), dtype=torch.float)

        return tensor_dict, label
    

