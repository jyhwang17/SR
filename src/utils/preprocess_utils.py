import pandas as pd
import numpy as np
import pickle
from collections import defaultdict, Counter


def drop_by_count(df, feature_names, threshold_list, num_iters = 2 ):
    
    '''
     drop rows having sparse feature

    :param dataframe df:
    :param list feature_names:
    :param list threshold_list:
    :param int num_iters:
    
    '''
    
    assert len(feature_names) == len(threshold_list), 'check input'
    assert type(feature_names) is list, 'check input type'
    assert type(threshold_list) is list, 'check input type'
    
    filtered_df = df.copy()
    
    for _ in range(num_iters):
        for i,feature_id in enumerate(feature_names):
            counts = filtered_df[feature_id].value_counts()
            satisfied = filtered_df[feature_id].isin( counts[ counts >= threshold_list[i] ].index)
            filtered_df = filtered_df[satisfied]
    
    return filtered_df

def drop_by_value(df, value):
    
    '''
     drop rows having specific value
     
    :param dataframe df:
    :param value:
    '''
    _df = df[~(df == value).any(axis=1)]
    
    return _df

def filter_by_reference(df, key, ref):
    
    '''
     filter rows based on the key,ref
     
    :param dataFrame df:
    :param key:
    :param list ref:
    '''
    _df = df[df[key].isin(ref)]

    return _df

def extract_features(df, feature_names):
    
    '''
     extract features with numpy form
     
    :param dataframe df:
    :param list feature_names:
    '''
    features = df.loc[:, feature_names ].values
    return features

def map_feature_id(feature_list, org_type, mapped_type):
    
    '''
     make hash table for id conversion
     
    :param list feature_list:
    :param org_type:
    :param mapped_type:
    '''
    
    f_cnt = 1
    id_map = defaultdict(mapped_type)
    id_rmap = defaultdict(org_type)
    id_map['pad'] = 0
    id_rmap[0] = 'pad'
    for feature_id in feature_list:
        if feature_id not in id_map:
            id_map[feature_id] = f_cnt
            id_rmap[f_cnt] = feature_id
            f_cnt +=1
            
    return f_cnt,id_map,id_rmap

def build_seq_dict(raw_interactions, uid_map, iid_map, tid_map):
    
    '''
     build seq_dict from raw_interactions
     
    :param list raw_interactions: this contains (user, item, time) tuple 
    :param dict uid_map: hash table for user ids
    :param dict iid_map: hash table for item ids
    :param dict tid_map: hash table for time ids
    '''
    sorted_interactions = sorted(raw_interactions, key=lambda t: 100000000*uid_map[t[0]]+tid_map[t[2]])
    seq_dict = defaultdict(list)
    
    for u,i,t in sorted_interactions:
        seq_dict[uid_map[u]].append(iid_map[i])

    return seq_dict

def build_seq_list(raw_interactions, uid_map, iid_map, tid_map):
    
    '''
     build seq_list from raw_interactions
    
    :param list raw_interactions: this contains (user, item, time) tuple
    :param dict uid_map: hash table for user ids
    :param dict iid_map: hash table for item ids
    :param dict tid_map: hash table for time ids
    '''
    
    sorted_interactions = sorted(raw_interactions, key=lambda t: (t[2]) )
    #sorted_interactions = sorted(raw_interactions, key=lambda t:uid_map[t[0]],tid_map[t[2]] )
    
    seq_list = [ [] for _ in range(len(uid_map)) ]

    for u,i,t in sorted_interactions:
        seq_list[uid_map[u]].append(iid_map[i])
    
    return seq_list

def split_train_test(lists, train_ratio, opt = 'ordered' ):
    
    '''
     split train/test
     
    :param lists lists: user history
    :param float train_ratio: (# of training samples)/(# of the interactions in user history)
    :param str opt: 'ordered' or 'unique'
    '''
    
    train_lists,test_lists = [],[]
    
    for items in lists:
        threshold = int(train_ratio*len(items))
        train_segment = set(items[:threshold])
        test_segment = set(items[threshold:])

        if opt == 'ordered': train_lists.append(items[:threshold])
        else: train_lists.append(list(train_segment))
        
        test_lists.append(list(test_segment.difference(train_segment)))
    
    return train_lists,test_lists

def split_seqs(seqs, train_ratio):
    
    early_seqs = [] # sequences to return
    later_seqs = []
    for seq in seqs:
        threshold = int(train_ratio*len(seq))
        early_seqs.append(seq[:threshold])
        later_seqs.append(seq[threshold:])
        
    return early_seqs, later_seqs

def extract_valid_seqs(early_seqs, later_seqs, train_popularity):
    
    ret_seq = []
    ret_item = []
    
    for uidx,seq in enumerate(later_seqs):
        seen = set(early_seqs[uidx])
        
        valid_seq = []
        valid_item = []
        for iidx, item in enumerate(seq):
            
            if item not in seen and train_popularity[item] > 0:
                valid_seq = early_seqs[uidx][:] + later_seqs[uidx][:iidx]
                valid_item.append(later_seqs[uidx][iidx])
                break
            seen.add(later_seqs[uidx][iidx])# 추가된 라인.
            
        
        
        ret_seq.append(valid_seq)
        ret_item.append(valid_item)
        
    return (ret_seq,ret_item)

def extract_cdr_validation(zipped_seqs):
    
    
    # (early_item_seqs, early_dom_seqs, later_item_seqs, layer_dom_seqs)
    ret_seq = []
    ret_item = []
    for uidx, (early_seq,early_dseq,
        later_seq,later_dseq) in enumerate(zipped_seqs):
        
        seen_items = Counter(early_seq)
        seen_domains = Counter(early_dseq)
        
        next_domains = Counter(later_dseq)
        valid_seq = []
        valid_item = []
        print(seen_domains, next_domains)
        
        for iidx, dom in enumerate(later_dseq):
            cond1 = (iidx ==0)&early_dseq[-1] != later_dseq[iidx]
            cond2 = later_dseq[iidx-1] != later_dseq[iidx]
            cond3 =  seen_domains[dom] <= max(0.1*(len(early_seq)+iidx),2)
            cond4 = len(seen_domains.keys()) >= 2
            
            if (cond1 or cond2):
                valid_seq = early_seq[:] + later_seq[:iidx]
                valid_item.append(later_seq[iidx])
                break
            seen_items.update([ later_seq[iidx] ])
            seen_domains.update([later_dseq[iidx]])
            
        ret_seq.append(valid_seq)
        ret_item.append(valid_item)
    
    return (ret_seq, ret_item)

def split_train_test_loo(lists, opt = 'ordered' ):
    
    '''
     split train/test
     
    :param lists lists: user history
    :param str opt: 'ordered'
    '''
    
    train_lists,test_lists = [],[]
    
    for items in lists:
        
        train_lists.append(items[:-1])
        test_lists.append(items[-1:])
        
    return train_lists,test_lists

def filter_testset(train_seqs, test_lists, item_features, num_recent_items = 1, filter_feature_idx = 0):
    
    '''
     filter testset based on the feature of recent items
     
     :param lists train_seqs: sequential interactions in training set
     :param lists test_lists: interactions in test set
     :param np.ndarray item_features: item featatures
     :param int num_recent_items: the number of recent items
     :param int filter_feature_idx: feature index 
    '''
    assert len(train_seqs) == len(test_lists), 'check shape'
    ret_lists = [] 
    for user, items in enumerate(test_lists):
        
        feat = item_features[train_seqs[user], filter_feature_idx ]
        ref = feat[-num_recent_items:] 
        #TODO
        test_features = np.expand_dims(item_features[items, filter_feature_idx ],1)
        test_items = np.expand_dims(np.array(items),1)
        _df = pd.DataFrame(np.concatenate((test_items, test_features),1) )
        ret_lists.append(list(_df[~_df[1].isin(ref)].values[:,0]))
        
    return ret_lists
    
def remove_redundant_items(item_sequences, state_sequences = None):

    '''
     remove consecutive redundant interactions
     
    :param lists item_sequences: sequential interactions
    :param lists state_sequences: state for sequential interactions
    
    '''
    if state_sequences == None: state_sequences = item_sequences
    
    return [[item for i, item in enumerate(items) if i==0 or state_sequences[u][i] != state_sequences[u][i-1]] for u,items in enumerate(item_sequences)]

def generate_pairs_from_sequences(item_sequences):
    
    '''
     generate_pairs_from_sequences
     
    :param lists item_sequences: sequential interactions
    '''
    for user,items in enumerate(item_sequences):
        for item in items:
            yield [user,item]

def generate_interaction_triplets(num_users, num_items, pos_items, neg_items):
    
    '''
     generate (user, positive_item, 1) from pos_items, generate (user, negative_item, 0) from neg_items
     
    :param int num_users: # of users
    :param int num_items: # of items
    :param lists pos_items: positive items
    :param lists neg_items: negative items
    '''
        
    assert len(pos_items) == num_users, 'check shape of pos_items'
    assert len(neg_items) == num_users, 'check shape of neg_items'
    
    for user,item in generate_pairs_from_sequences(pos_items):
        yield [user, item, 1.]
    for user,item in generate_pairs_from_sequences(neg_items):
        yield [user, item, 0.]
        
def save_cache(save_dir_path, name, objs):
    
    with open(save_dir_path+name+'.pkl','wb') as f:
        pickle.dump(objs, f)
