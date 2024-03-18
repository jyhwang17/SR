#문제의 코드
import numpy as np
import torch
import time
import copy
from tqdm import tqdm
from collections import defaultdict

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
    


def evaluate_topk(model, seq_list, dataset, scenario = 'valid', topk = 20):

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
    
    #prepare sequences
    seq_indices = torch.split(torch.LongTensor(seq_list),256)
    all_item_indices = torch.arange(dataset.num_items).unsqueeze(1).cuda()
    
    recall = torch.cuda.DoubleTensor()
    ndcg = torch.cuda.DoubleTensor()
    mrr = torch.cuda.DoubleTensor()
    
    with torch.no_grad():
        for i, batch_seq_indices in enumerate(tqdm(seq_indices, desc="Evaluation", disable = True)):
            batch_seq_indices = batch_seq_indices.cuda()
            
            batch_scores = model(batch_seq_indices,
                                     last_subseqs[batch_seq_indices],
                                     all_item_indices, # NX1
                                     pred_opt='eval')
            #print(predicted_scores)
            #breakpoint()
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

    return {"recall":recall.mean(),"ndcg":ndcg.mean(), "mrr":mrr.mean() }