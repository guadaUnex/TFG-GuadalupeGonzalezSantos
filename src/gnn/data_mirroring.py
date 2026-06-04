import torch
import random
from data_conversions import clone_sequence

def mirror_sequence(sequence):
    new_sequence = sequence

    for frame in new_sequence:
        if frame['robot'].x.numel() > 0:
            frame['robot'].x[:,1] = -frame['robot'].x[:,1]
            frame['robot'].x[:,2] = -frame['robot'].x[:,2]
            frame['robot'].x[:,7] = -frame['robot'].x[:,7]
            frame['robot'].x[:,8] = -frame['robot'].x[:,8]

        if frame['human'].x.numel() > 0:
            frame['human'].x[:,1] = -frame['human'].x[:,1]
            frame['human'].x[:,2] = -frame['human'].x[:,2]
        if frame['object'].x.numel() > 0:
            frame['object'].x[:,1] = -frame['object'].x[:,1]
            frame['object'].x[:,2] = -frame['object'].x[:,2]
        if frame['wall'].x.numel() > 0:
            frame['wall'].x[:,1] = -frame['wall'].x[:,1]

    return new_sequence


# def tensor_transform_with_random_mirroring(tDict_sequence):
#     if random.randint(0,1)==1:
#         tensor_dict_clone = clone_sequence(tDict_sequence)                        
#         tDict_mirrored = mirror_tDic_sequence(tensor_dict_clone)
#         return tDict_mirrored
#     else:
#         return tDict_sequence
    



