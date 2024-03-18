import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .layers import TransformerEncoder as TE
    
class HGN(nn.Module):
    
    def __init__(self, args):
        super(HGN, self).__init__()
        
        #SASREC..
        self.args = args
        self.mtype = 'JOIN'
        
        self.U = nn.Embedding(self.args.num_seqs, self.args.dims)
        self.U.weight.data.normal_(0., 1./self.U.embedding_dim)
        
        self.V = nn.Embedding(self.args.num_items, self.args.dims, padding_idx = 0)
        self.V.weight.data.normal_(0., 1./self.V.embedding_dim)
        self.V.weight.data[0] = 0.
        
        self.g1 = nn.Linear(self.args.dims, self.args.dims, bias=False)
        self.g2 = nn.Linear(self.args.dims, self.args.dims)
        self.g3 = nn.Linear(self.args.dims, self.args.dims, bias=False)
        self.g4 = nn.Linear(self.args.dims, self.args.dims, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.args.dropout)
    
    def forward(self, user_indices, item_seq_indices, target_item_indices, pred_opt = ''):
        
        B,L = item_seq_indices.size()
        tgt_ebd = self.V(target_item_indices)#[B,L,N,D] or #[B,1,D]
        seq_ebd = self.V(item_seq_indices)#[B,L,D]
        usr_ebd = self.U(user_indices).unsqueeze(1) # B,1,D
        
        f_seq_ebd = seq_ebd * self.sigmoid(self.g1(seq_ebd) + self.g2(usr_ebd))
        seq_rep = f_seq_ebd * self.sigmoid(self.g3(f_seq_ebd) + self.g4(usr_ebd))


        rec_heads = usr_ebd + seq_rep.cumsum(1)/( ((item_seq_indices != 0).cumsum(1).unsqueeze(-1)) +1.0) + seq_ebd#.cumsum(1)
        #rec_heads = usr_ebd + seq_ebd.cumsum(1)
        if pred_opt == 'training':
            rec_heads = rec_heads.unsqueeze(2)# B x L x 1 x dims
            rec_heads = rec_heads.view(rec_heads.size(0)*rec_heads.size(1), rec_heads.size(2), rec_heads.size(3))
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

    def loss(self,
             user_indices,
             item_seq_indices,
             pos_item_indices,
             neg_item_indices
            ):
        
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
        neg_tgt_item_indices = torch.diagonal(neg_tgt_item_indices, offset = 0, dim1=1, dim2=2).unsqueeze(-1)

        pad_filter = (total_sequence.sum(-1)!=0)

        user_indices = user_indices[pad_filter]
        total_sequence = total_sequence[pad_filter]
        neg_tgt_item_indices = neg_tgt_item_indices[pad_filter]
        
        loss_mask = (total_sequence[:,:-1] != 0)# B L
        pos_score = self.forward(user_indices,
                                 total_sequence[:,:-1],#[B,L]
                                 total_sequence[:,1:].unsqueeze(-1),#[B,L,1]
                                 pred_opt='training') #=>[B,L,1]
        
        neg_score = self.forward(user_indices,
                                 total_sequence[:,:-1],#[B,L]
                                 neg_tgt_item_indices,#[B,L,N]
                                 pred_opt='training') #=>[B,L,N]
        sigm = nn.Sigmoid()
        loss = -torch.log( sigm(pos_score-neg_score))[loss_mask].flatten()

        return loss.mean()