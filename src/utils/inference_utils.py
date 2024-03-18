import torch
import numpy as np
from collections import defaultdict

def map_index(key: torch.LongTensor, val: torch.LongTensor, t: torch.LongTensor) -> torch.LongTensor:
    
    # From `https://stackoverflow.com/questions/13572448`.
    '''
    :param torch.LongTensor (length of dict) key: sorted keys
    :param torch.LongTensor (length of dict) val: value
    :param torch.LongTensor (length of input) t: input 1d-tensor
    
    :return: mapped indices
    '''
    index = torch.bucketize(t.ravel(), key)
    remapped = val[index].reshape(t.shape)

    return remapped

def filter_sequence(input_sequence: torch.LongTensor, ref: torch.LongTensor) -> torch.LongTensor:
    
    '''
    filter input indices
    
    :param torch.LongTensor (L) input_sequence: input indices
    :param torch.LongTensor (L) ref: reference keys
    
    :return: filtered indices
    '''
    
    mask = (input_sequence.squeeze().unsqueeze(1) == ref.unsqueeze(0)).sum(1)
    masked_sequence = input_sequence * mask
    masked_sequence = masked_sequence[masked_sequence>0]
    
    return masked_sequence
