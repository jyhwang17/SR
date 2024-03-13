import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .layers import TransformerEncoder as TE

def gcn_norm(coo_tensor: torch.Tensor):
    """get normalized adj matrix

    Args:
        coo_tensor (torch.Tensor): torch.coo_tensor
    """
    coo_tensor = coo_tensor.coalesce()
    indices = coo_tensor.indices()
    values = coo_tensor.values()
    row, col = indices[0], indices[1]

    # calculate degree
    N = coo_tensor.size(0)
    out = torch.zeros((N, ), device = "cuda" )
    one = torch.ones((col.size(0), ), dtype=out.dtype, device=out.device)
    deg = out.scatter_add_(0, col, one)
    deg_inv_sqrt = deg.pow(-0.5)
    # Handling zero division
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    try:
        edge_weight_normalized = values * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    except:
        breakpoint()
    return torch.sparse_coo_tensor(indices, edge_weight_normalized, size=coo_tensor.size())

class LGCN(nn.Module):
    
    def __init__(self, args):
        super(LGCN, self).__init__()

        self.args = args
        self.mtype = 'JOIN'
        
                
        #For multi-domain
        self.user_embeddings = []
        self.item_embeddings = []
        for _ in range(self.args.num_domains):
            usr_ebd = nn.Embedding(self.args.num_seqs, self.args.dims)
            usr_ebd.weight.data.normal_(0., 1./usr_ebd.embedding_dim)
            
            itm_ebd = nn.Embedding(self.args.num_items, self.args.dims, padding_idx = 0)
            itm_ebd.weight.data.normal_(0., 1./itm_ebd.embedding_dim)
            itm_ebd.weight.data[0] = 0.

            self.user_embeddings.append(usr_ebd)
            self.item_embeddings.append(itm_ebd)
        
        self.user_embeddings = nn.ModuleList(self.user_embeddings)
        self.item_embeddings = nn.ModuleList(self.item_embeddings)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.args.dropout)
        
        self.ii_mat_tensor = self.args.ii_mat_tensor
        self.ui_mat_tensor = self.args.ui_mat_tensor
        self.item2dom = torch.LongTensor(self.args.item2dom).cuda()

    def graph_convolution(self, target_domain):

        usr_ebd = self.user_embeddings[target_domain].weight # u,D
        itm_ebd = self.item_embeddings[target_domain].weight # I,D
        ebd = torch.cat((usr_ebd,itm_ebd),0)
        
        ebd_lst = torch.Tensor(ebd.unsqueeze(1)).cuda()
        mat = self.ui_mat_tensor[target_domain]
        mat = gcn_norm(mat)
        for nlayer in range(self.args.graph_layers):
            
            ebd = torch.sparse.mm(mat, ebd)
            ebd_lst = torch.cat([ebd_lst,ebd.unsqueeze(1)],1)
        
        ret_ebd = ebd_lst.mean(1)

        return ret_ebd
    
    def forward(self, user_indices, item_seq_indices, target_item_indices, target_domain = None, pred_opt = ''):
        
        g_ebd = self.graph_convolution(0)
        
        B,L = item_seq_indices.size()

        usr_ebd = g_ebd[user_indices].unsqueeze(1).expand(-1,L,-1) #B,L,D
        tgt_ebd = g_ebd[target_item_indices]# B,L,N,D

        rec_heads = usr_ebd
        
        if pred_opt == 'training':
            rec_heads = rec_heads.unsqueeze(2).contiguous()# B x L x 1 x dims
            
            rec_heads = rec_heads.view(rec_heads.size(0)*rec_heads.size(1),rec_heads.size(2),rec_heads.size(3))
            tgt_ebd = tgt_ebd.view(tgt_ebd.size(0)*tgt_ebd.size(1),-1,tgt_ebd.size(3))
            #seq_rep : BL x 1 x dims
            #tgt_ebd : BL x N x dims
            rel_score = rec_heads.bmm(tgt_ebd.permute([0,2,1])) # BL x 1 x N
            rel_score = rel_score.squeeze(1).view(B,L,-1) # B,L,N
            return rel_score
        else:
            # rec_heads: batch x dims (e.g., 500, 64)
            # tgt_ebd : all_Items x dims (e.g., 32980, 1, 64)

            rel_score = rec_heads[:,-1,:].mm(tgt_ebd.squeeze(1).t()) # batch x all_Items
            return rel_score

    def loss(self, user_indices,sorted_item_seq_indices,item_seq_indices, pos_item_indices, neg_item_indices ):

        '''
        compute loss

        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (B x 1) pos_item_indices:
        :param torch.LongTensor (B x N) neg_item_indices:
        
        :return: loss
        '''
        
        B,L = item_seq_indices.size() 
        total_sequence = torch.cat((item_seq_indices, pos_item_indices),1) #[B,L+1]
        neg_tgt_item_indices = neg_item_indices.unsqueeze(1).expand(-1,item_seq_indices.size(1),-1) #[B,L,L]
        #neg_tgt_item_indices = torch.diagonal(neg_tgt_item_indices, offset = 0, dim1=1, dim2=2).unsqueeze(-1)

        pad_filter = (total_sequence.sum(-1)!=0)

        user_indices = user_indices[pad_filter]
        total_sequence = total_sequence[pad_filter]
        neg_tgt_item_indices = neg_tgt_item_indices[pad_filter]
        
        
        loss = torch.Tensor().cuda()
        for t in range(1,self.args.num_domains):
            loss_mask = (total_sequence[:,:-1] != 0)# B L
#            pos_domain_mask = (self.item2dom[total_sequence[:,1:].unsqueeze(-1)] == t).squeeze()
#            neg_domain_mask = (self.item2dom[neg_tgt_item_indices] == t).squeeze()
            loss_mask = loss_mask# & pos_domain_mask & neg_domain_mask

            pos_score = self.forward(user_indices,
                                     total_sequence[:,:-1],#[B,L]
                                     total_sequence[:,1:].unsqueeze(-1),#[B,L,1]
                                     target_domain = t,
                                     pred_opt='training') #=>[B,L,1]

            neg_score = self.forward(user_indices,
                                     total_sequence[:,:-1],#[B,L]
                                     neg_tgt_item_indices,#[B,L,N]
                                     target_domain = t,
                                     pred_opt='training') #=>[B,L,N]
            sigm = nn.Sigmoid()
            domain_loss = -torch.log( sigm(pos_score-neg_score))[loss_mask].flatten()
            loss = torch.cat((loss, domain_loss),0)
        
        return loss.mean()