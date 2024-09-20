import numpy as np
import torch
import time
import copy
from tqdm import tqdm
from collections import defaultdict
from torch.distributions import Categorical
import torch.nn as nn
from collections import Counter

def _get_recall(batch_hits, num_tests, cond):

    return batch_hits.sum(1,keepdims=True)[cond]/(num_tests)[cond]

def _get_ndcg(batch_hits, num_tests, cond):
    
    topk = batch_hits.size(1)
    denominator = torch.log2(1+torch.arange(1,topk+1)).unsqueeze(0).cuda()
    DCG_K = (batch_hits/denominator)[cond]
    
    IDCG_K = torch.cumsum((1./denominator),1).flatten()
    
    IDCG_I = num_tests[cond].flatten().long() -1
    IDCG_I[IDCG_I>=topk] = topk-1
    
    norm = IDCG_K[IDCG_I].unsqueeze(1)
    
    NDCG_K = (DCG_K).sum(1,keepdims=True)/norm
    
    return NDCG_K
    
def _get_rr(batch_hits, cond):
    denominator = (1.0 / torch.arange(1, batch_hits.size(1) + 1)).cuda()
    denominator = denominator.unsqueeze(0)

    # rr is reciprocal rank
    rr = ((batch_hits >= 1.0) * denominator * batch_hits)[cond]
    rr = torch.max(rr, 1).values

    return rr
    
def evaluate_asmip_topk(model, seq_list, dataset, scenario = 'valid', pred_opt = 'mip', topk = 20):

    '''
     evaluate topk ranking performance
     (currently, only recall is implemented)
     
     :param model: recommender system
     :param list seq_list: user list for test
     :param dataset: dataset
     :param int topk: # of items to show
     "
    '''
    
    model.eval()
    if scenario == 'valid':
        coo = dataset.valid_seen_mat.tocoo()       
        ind = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        val = torch.LongTensor(coo.data.astype(np.int32))
        seen_mat = torch.sparse_coo_tensor(ind, val, (dataset.num_seqs, dataset.num_items)).cuda()
        
        coo = dataset.valid_gt_mat.tocoo()
        ind = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        val = torch.LongTensor(coo.data.astype(np.int32))
        gt_mat = torch.sparse_coo_tensor(ind, val, (dataset.num_seqs, dataset.num_items)).cuda()
        
        last_subseqs = torch.LongTensor(dataset.valid_last_subseqs[:,1:]).cuda()
        gt_analysis = torch.cuda.LongTensor(dataset.valid_items)
        
    if scenario == 'test':
        coo = dataset.test_seen_mat.tocoo()
        ind = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        val = torch.LongTensor(coo.data.astype(np.int32))
        seen_mat = torch.sparse_coo_tensor(ind, val, (dataset.num_seqs, dataset.num_items)).cuda()
        
        coo = dataset.test_gt_mat.tocoo()
        ind = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        val = torch.LongTensor(coo.data.astype(np.int32))
        gt_mat = torch.sparse_coo_tensor(ind, val, (dataset.num_seqs, dataset.num_items)).cuda()
        #check last subseqs
        last_subseqs = torch.LongTensor(dataset.test_last_subseqs[:,1:]).cuda()
        gt_analysis = torch.cuda.LongTensor(dataset.test_items)
    
    #prepare sequences
    seq_indices = torch.split(torch.LongTensor(seq_list),256)
    all_item_indices = torch.arange(dataset.num_items).unsqueeze(1).cuda()
    
    recall = torch.cuda.DoubleTensor()
    ndcg = torch.cuda.DoubleTensor()
    mrr = torch.cuda.DoubleTensor()
    
    #additional metric for analysis
    hits_pop = torch.cuda.DoubleTensor()
    topk_pop = torch.cuda.DoubleTensor()
    entropy = torch.cuda.DoubleTensor()
    attention_entropy1 = torch.cuda.DoubleTensor() # Layer1
    attention_entropy2 = torch.cuda.DoubleTensor() # Layer2
    
    smx = torch.nn.Softmax(dim=1)
    item_pop = torch.cuda.DoubleTensor(dataset.item_pop)
    
    with torch.no_grad():
        for i, batch_seq_indices in enumerate(tqdm(seq_indices, desc="Evaluation", disable = True)):
            batch_seq_indices = batch_seq_indices.cuda()
            
            batch_scores, entropy1, entropy2 = model(batch_seq_indices,
                                     last_subseqs[batch_seq_indices],
                                     all_item_indices, # NX1
                                     pred_opt=pred_opt)
            
            batch_entropy = Categorical(probs = smx(batch_scores[:,1:]) ).entropy()
            
            batch_seen_array = torch.index_select(seen_mat, 0, batch_seq_indices).to_dense()
            batch_scores.unsqueeze(0)[:, batch_seen_array > 0 ] = -1000.0 # ignore seen items
            batch_topk = torch.topk(batch_scores, k = topk, dim=1, sorted=True).indices
            
            batch_gt_array = torch.index_select(gt_mat, 0, batch_seq_indices).to_dense().long()
            
            num_tests = batch_gt_array.sum(1,keepdims=True)
            batch_hits = torch.gather(batch_gt_array, 1, batch_topk)
            
            #select validation user
            cond = (num_tests > 0.).flatten()
            
            batch_recall = _get_recall(batch_hits, num_tests, cond)
            recall = torch.cat((recall, batch_recall),0)
            
            batch_ndcg = _get_ndcg(batch_hits, num_tests, cond)
            ndcg = torch.cat((ndcg, batch_ndcg),0)
            
            batch_mrr = _get_rr(batch_hits, cond)
            mrr = torch.cat((mrr, batch_mrr.type('torch.DoubleTensor').cuda()), 0)
            
            
            #for analysis.
            batch_hits_mask = batch_hits.sum(1).bool()
            batch_gt = gt_analysis[batch_seq_indices]
            
            hits_items = batch_gt[batch_hits_mask].flatten()
            hits_pop = torch.cat((hits_pop, item_pop[hits_items]),0)
            topk_pop = torch.cat((topk_pop, item_pop[batch_topk].flatten()),0)
            
            attention_entropy1 = torch.cat((attention_entropy1, entropy1),0)
            attention_entropy2 = torch.cat((attention_entropy2, entropy2),0)
            
            entropy = torch.cat((entropy, batch_entropy), 0)
            
    return {"recall":recall.mean(),"ndcg":ndcg.mean(), "mrr":mrr.mean(), "entropy":entropy.mean(),
            "attention_entropy1": attention_entropy1.mean(), "attention_entropy2": attention_entropy2.mean(),
            "hits_list":recall, "hits_pop": hits_pop.mean(), "topk_pop": topk_pop.mean()}


def evaluate_topk(model, seq_list, dataset, scenario = 'valid',topk = 20):

    '''
     evaluate topk ranking performance
     (currently, only recall is implemented)
     
     :param model: recommender system
     :param list seq_list: user list for test
     :param dataset: dataset
     :param int topk: # of items to show
     "
    '''
    
    model.eval()
    if scenario == 'valid':
        coo = dataset.valid_seen_mat.tocoo()       
        ind = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        val = torch.LongTensor(coo.data.astype(np.int32))
        seen_mat = torch.sparse_coo_tensor(ind, val, (dataset.num_seqs, dataset.num_items)).cuda()
        
        coo = dataset.valid_gt_mat.tocoo()
        ind = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        val = torch.LongTensor(coo.data.astype(np.int32))
        gt_mat = torch.sparse_coo_tensor(ind, val, (dataset.num_seqs, dataset.num_items)).cuda()
        
        last_subseqs = torch.LongTensor(dataset.valid_last_subseqs[:,1:]).cuda()
        gt_analysis = torch.cuda.LongTensor(dataset.valid_items)
        
    if scenario == 'test':
        coo = dataset.test_seen_mat.tocoo()
        ind = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        val = torch.LongTensor(coo.data.astype(np.int32))
        seen_mat = torch.sparse_coo_tensor(ind, val, (dataset.num_seqs, dataset.num_items)).cuda()
        
        coo = dataset.test_gt_mat.tocoo()
        ind = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        val = torch.LongTensor(coo.data.astype(np.int32))
        gt_mat = torch.sparse_coo_tensor(ind, val, (dataset.num_seqs, dataset.num_items)).cuda()
        #check last subseqs
        last_subseqs = torch.LongTensor(dataset.test_last_subseqs[:,1:]).cuda()
        gt_analysis = torch.cuda.LongTensor(dataset.test_items)
    
    #prepare sequences
    seq_indices = torch.split(torch.LongTensor(seq_list),256)
    all_item_indices = torch.arange(dataset.num_items).unsqueeze(1).cuda()
    
    recall = torch.cuda.DoubleTensor()
    ndcg = torch.cuda.DoubleTensor()
    mrr = torch.cuda.DoubleTensor()
    
    #additional metric for analysis
    hits_pop = torch.cuda.DoubleTensor()
    topk_pop = torch.cuda.DoubleTensor()
    entropy = torch.cuda.DoubleTensor()
    cross_entropy = torch.cuda.DoubleTensor()
    
    hits_items = torch.cuda.LongTensor()
    smx = torch.nn.Softmax(dim=1)
    item_pop = torch.cuda.DoubleTensor(dataset.item_pop)
    
    with torch.no_grad():
        for i, batch_seq_indices in enumerate(tqdm(seq_indices, desc="Evaluation", disable = True)):
            batch_seq_indices = batch_seq_indices.cuda()
            
            batch_scores = model(batch_seq_indices,
                                     last_subseqs[batch_seq_indices],
                                     all_item_indices, # NX1
                                     pred_opt='eval')
            
           
            batch_gt_array = torch.index_select(gt_mat, 0, batch_seq_indices).to_dense().long()
            
            #batch_entropy = Categorical(probs = smx(batch_scores[:,1:]) ).entropy()
            
            batch_cross_entropy = -(batch_gt_array[:,1:]*smx(batch_scores[:,1:]).log()).sum(-1)
            
            #breakpoint()
            
            batch_seen_array = torch.index_select(seen_mat, 0, batch_seq_indices).to_dense()
            batch_scores.unsqueeze(0)[:, batch_seen_array > 0 ] = -1000.0 # ignore seen items
            batch_topk = torch.topk(batch_scores, k = topk, dim=1, sorted=True).indices
            
            num_tests = batch_gt_array.sum(1,keepdims=True)
            batch_hits = torch.gather(batch_gt_array, 1, batch_topk)
            
            #select validation user
            cond = (num_tests > 0.).flatten()
            
            batch_recall = _get_recall(batch_hits, num_tests, cond)
            recall = torch.cat((recall, batch_recall),0)
            
            batch_ndcg = _get_ndcg(batch_hits, num_tests, cond)
            ndcg = torch.cat((ndcg, batch_ndcg),0)
            
            batch_mrr = _get_rr(batch_hits, cond)
            mrr = torch.cat((mrr, batch_mrr.type('torch.DoubleTensor').cuda()), 0)
            
            
            #For Analysis.
            batch_hits_mask = batch_hits.sum(1).bool()
            batch_gt = gt_analysis[batch_seq_indices]
            batch_hits_items = batch_gt[batch_hits_mask].flatten()# Check 해보기.
            
            hits_pop = torch.cat((hits_pop, item_pop[batch_hits_items]),0)
            topk_pop = torch.cat((topk_pop, item_pop[batch_topk].flatten()),0)
            
            #entropy = torch.cat((entropy, batch_entropy), 0)
            cross_entropy = torch.cat((cross_entropy, batch_cross_entropy), 0)
            hits_items = torch.cat((hits_items, batch_hits_items), 0)
            
    return {"recall":recall.mean(),"ndcg":ndcg.mean(), "mrr":mrr.mean(), "cross_entropy":cross_entropy.mean(), "hits_list":recall,
            "hits_pop": hits_pop.mean(), "topk_pop": topk_pop.mean(), "hits_items":hits_items.cpu()}