import os
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import collections
from collections import namedtuple
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix

def _load_cache(load_dir_path, name):

    with open(load_dir_path+name+".pkl",'rb') as f:
        data_dict = pickle.load(f)
    
    return data_dict

def select_seq_knowledge(args):

    load_dir_path = args.model_img_path
    
    try:
        best_img, best_conf, best_perf = None, None, 0.


        for root, dirs, files in os.walk(load_dir_path):
            print(root, dirs, files)
            #knowledge/data/
            #search model_config_performance
            if root == load_dir_path:
                
                for f in files:
                    model,conf,perf = f.split('_')
                    
                    #if( not (args.model == model and args.window_length == int(length) and args.dims == int(dims) ) ): continue
                    if( not (args.model == model) ): continue
                    length,dims = conf.split('-')
                    if( not(args.dims_t == int(dims)) ): continue
                    perf = float(perf)
                    if(best_perf < perf):
                        best_perf = perf
                        best_img = f
                        #best_conf = conf

        #knowledge/hyperparams/data/performance_.pkl
        #arg_dict = _load_cache(load_dir_path+'/hyperparams/', f"{best_perf:.4f}_")
        
        arg_dict = _load_cache(load_dir_path+'/hyperparams/',f"{best_img}")
        return (best_img, arg_dict['hyperparams'])
    except Exception as e:
        if(best_perf == 0.0): print("Check hyperparameters (dims,window_length)")
        print('load error',e)
        #import pdb;pdb.set_trace()
        
def load_mapping_info(args) -> collections.namedtuple:
    
    meta_dict = None
    try:
        with open(args.data_path+"unit_meta_cache.pkl",'rb') as f:
            meta_dict = pickle.load(f)
    except Exception as e:
        print('load error', e)
    
    meta_tuple = namedtuple('mapper', ["ordered_iid_keys", "ordered_iid_vals",
                                       "ordered_iid_rkeys", "ordered_iid_rvals"])
    
    #store mapping tensor(db_id->__id)
    del(meta_dict["iid_map"]["pad"])
    ordered_iid_keys, ordered_iid_vals = zip(*sorted(meta_dict["iid_map"].items()))
    
   
    meta_tuple.ordered_iid_keys = torch.LongTensor(list(ordered_iid_keys))
    meta_tuple.ordered_iid_vals = torch.LongTensor(list(ordered_iid_vals))
    
    #store mapping tensor(model_id->db_id)
    del(meta_dict["iid_rmap"][0])
    ordered_iid_rkeys, ordered_iid_rvals = zip(*sorted(meta_dict["iid_rmap"].items()))
    meta_tuple.ordered_iid_rkeys = torch.LongTensor(list(ordered_iid_rkeys))
    meta_tuple.ordered_iid_rvals = torch.LongTensor(list(ordered_iid_rvals))
    
    return meta_tuple

class SEQDataset(Dataset):

    def __init__(self, args):
       
        load_dir_path = args.data_path
        name = args.dataset
        L = args.window_length
        T = args.target_length
        try:
            data_dict = _load_cache(load_dir_path, name+'_cache')
            self.num_users = data_dict["num_users"]
            self.num_items = data_dict["num_items"]
            self.train_seqs = data_dict["train_seqs"]
            self.valid_seqs = data_dict["valid_seqs"]
            self.test_seqs = data_dict["test_seqs"]
            
            self.valid_seen_mat = data_dict["valid_seen_mat"]
            self.valid_gt_mat = data_dict["valid_gt_mat"]
            self.test_seen_mat = data_dict["test_seen_mat"]
            self.test_gt_mat = data_dict["test_gt_mat"]
            
            self.sorted_pos_lists = data_dict["sorted_pos_lists"]
            
            self.valid_items = data_dict["valid_items"]
            self.test_items = data_dict["test_items"]
            self.item_pop = data_dict["item_pop"]
            
            #self.ladj_mat = data_dict["ladj_mat"]
            #self.radj_mat = data_dict["radj_mat"]
            self.adj_mat = data_dict["adj_mat"]
            
            self.ladj_mat1 = data_dict["ladj_mat1"]
            self.ladj_mat2 = data_dict["ladj_mat2"]
            self.ladj_mat3 = data_dict["ladj_mat3"]
            
            self.radj_mat1 = data_dict["radj_mat1"]
            self.radj_mat2 = data_dict["radj_mat2"]
            self.radj_mat3 = data_dict["radj_mat3"]
            
        except Exception as e:
            print('load error', e)
        self.num_seqs = self.num_users
        self.train_instances = None #filled by make_subsequences
        
        self.train_last_subseqs = None #filled by make_subsequences
        self.valid_last_subseqs = None
        self.test_last_subseqs = None
        
        self.train_last_subseqs, self.train_instances = self.make_subsequences(self.train_seqs, L+T)  
        self.valid_last_subseqs, _ = self.make_subsequences(self.valid_seqs, L+T)
        self.test_last_subseqs, _ = self.make_subsequences(self.test_seqs, L+T)
          
    def make_subsequences(self, seqs, max_subseq_length = 5):
        
        num_subseqs = sum([len(seq)- max_subseq_length + 1 
                                if len(seq) >= max_subseq_length else 1 
                                for seq in seqs])
        
        subseqs = np.zeros((num_subseqs, max_subseq_length), dtype=np.int64)
        subseqs_users = np.empty(num_subseqs, dtype=np.int64)
        last_subseqs = np.zeros((self.num_seqs, max_subseq_length), dtype=np.int64)

        idx=0
        # sliding window
        for uid, seq in enumerate(seqs):
            if len(seq) < max_subseq_length:
                subseqs[idx][max_subseq_length-len(seq):] = seq #(fix)
                subseqs_users[idx] = uid
                last_subseqs[uid][max_subseq_length-len(seq):] = seq #copy safe
                idx+=1
                continue
            
            for s in range(0, len(seq) - max_subseq_length +1 ):
                #[s,e) 
                e = s + max_subseq_length
                subseqs[idx] = seq[s:e] #copy safe
                subseqs_users[idx] = uid #copy safe (fix)
                idx+=1
                
            last_subseqs[uid] = subseqs[idx-1] #copy safe (fix)
        
        sorted_subseqs = np.concatenate((np.sort(subseqs,axis=1), [[self.num_items]]* len(subseqs) ), axis=1)
        instances = np.concatenate(
            (np.expand_dims(subseqs_users,axis=1),subseqs, sorted_subseqs ),
            axis=1)

        return last_subseqs, instances
    
    def __len__(self):
        
        return len(self.train_instances)
    
    def __getitem__(self,idx):
        
        return self.train_instances[idx]