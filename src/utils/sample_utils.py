import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import time
from multiprocessing import Pool
def neg_assign(x):
    (m,c) = x
    return c[m][:100]

class NEGSampler():

    def __init__(self, args):
        
        self.device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_seqs = args.num_seqs
        self.num_items = args.num_items

    def sample_negative_items_online(self, seen_items, how_many):
        
        cand = torch.randint(1, self.num_items,
                             (len(seen_items),max(how_many*2,500)),
                              device=self.device).cuda()
        
        indices = torch.searchsorted(seen_items, cand)
        neq_indices = (cand!=torch.gather(seen_items, 1, indices)).long()
        topk_indices = torch.topk(neq_indices, k = how_many, dim=1).indices
        neg_items = torch.gather(cand, 1, topk_indices)
        
        return neg_items