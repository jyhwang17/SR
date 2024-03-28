import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .layers import TransformerEncoder as TE
    
class CT4REC(nn.Module):
    
    def __init__(self, args):
        super(CT4REC, self).__init__()
        
        #Next-Item Prediction Task with SASREC Architecture.
        
        self.args = args
        self.mtype = 'JOIN'
        
        self.V = nn.Embedding(self.args.num_items, self.args.dims, padding_idx = 0)
        self.V.weight.data.normal_(0., 1./self.V.embedding_dim)
        self.V.weight.data[0] = 0.
        
        self.P = nn.Embedding(self.args.window_length, self.args.dims)
        self.P.weight.data.normal_(0., 1./self.P.embedding_dim)
        
        self.seq_transformer_encoder = TE(n_layers = self.args.encoder_layers,
                                          n_heads = 2,
                                          hidden_size = self.args.dims,
                                          inner_size = self.args.dims*4,
                                          hidden_dropout_prob = self.args.dropout,
                                          attn_dropout_prob = 0.0,
                                          hidden_act = 'gelu',
                                          bidirectional=False)
                
        
        self.criterion = nn.BCELoss(reduction='none')
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.args.dropout)
        self.layer_norm = nn.LayerNorm(self.args.dims)
    
    def get_attention_mask(self, item_seq, bidirectional = False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask
    
    
    def get_contextualized_rep(self, user_indices, item_seq_indices):
        
        '''
         predict the score at mask [*, N]
        
        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (* X N) target_item_indices:loss mask로 마스크된 위치의 target들
        :param torch.BoolTensor (B x L) loss_mask: True if the element is masked
        '''
        B,L = item_seq_indices.size()
        position_ids = torch.arange(L, dtype=torch.long, device = item_seq_indices.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq_indices)
        position_ebd = self.P(position_ids)
        
        seq_ebd = self.V(item_seq_indices) + position_ebd
        
        seq_ebd = self.layer_norm(seq_ebd)
        seq_ebd = self.dropout_layer(seq_ebd)
        seq_rep = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask(item_seq_indices))[-1]
        
        return seq_rep
    
    def forward(self, user_indices, item_seq_indices, target_item_indices, pred_opt = ''):
        B,L = item_seq_indices.size()
        
        position_ids = torch.arange(item_seq_indices.size(1), dtype=torch.long, device = item_seq_indices.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq_indices)
        position_ebd = self.P(position_ids)
        
        tgt_ebd = self.V(target_item_indices)#[B,L,N,D] or #[B,1,D]
        
        seq_ebd = self.V(item_seq_indices) + position_ebd #[B,L,D]
        seq_ebd = self.dropout_layer(self.layer_norm(seq_ebd))
        seq_rep = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask(item_seq_indices))[-1]

        rec_heads = seq_rep[:,-1,:] # B x dims
        
        if pred_opt == 'training':
            seq_rep = seq_rep.unsqueeze(2)# B x L x 1 x dims
            seq_rep = seq_rep.view(seq_rep.size(0)*seq_rep.size(1),seq_rep.size(2),seq_rep.size(3))
            tgt_ebd = tgt_ebd.view(tgt_ebd.size(0)*tgt_ebd.size(1),-1,tgt_ebd.size(3))
            #seq_rep : BL x 1 x dims
            #tgt_ebd : BL x N x dims
            rel_score = seq_rep.bmm(tgt_ebd.permute([0,2,1])) # BL x 1 x N
            rel_score = rel_score.squeeze(1).view(B,L,-1) # B,L,N
            return rel_score
        else:
            # rec_heads: batch x dims (e.g., 500, 64)
            # tgt_ebd : all_Items x dims (e.g., 32980, 1, 64)
            rel_score = rec_heads.mm(tgt_ebd.squeeze(1).t()) # batch x all_Items
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
        
        #B,L = item_seq_indices.size() 
        rec_loss = torch.cuda.DoubleTensor() # recommendation loss
        
        total_sequence = torch.cat((item_seq_indices, pos_item_indices),1) #[B,L+1]
        neg_tgt_item_indices = neg_item_indices.unsqueeze(1).expand(-1,item_seq_indices.size(1),-1) #[B,L,L]
        
        pad_filter = (total_sequence.sum(-1)!=0)
        total_sequence = total_sequence[pad_filter]
        neg_tgt_item_indices = neg_tgt_item_indices[pad_filter]
        
        B,L = total_sequence[:,:-1].size()
        
        tgt_item_indices = torch.cat((total_sequence[:,1:].unsqueeze(-1), neg_tgt_item_indices),2)
        loss_mask = (total_sequence[:,:-1] != 0)# B L
        
        #get representation
        seq_rep1 = self.get_contextualized_rep(user_indices,total_sequence[:,:-1])
        seq_rep2 = self.get_contextualized_rep(user_indices,total_sequence[:,:-1])
        tgt_ebd = self.V(tgt_item_indices)

        #prediction
        seq_rep1 = seq_rep1.unsqueeze(2)# B x L x 1 x dims
        seq_rep1 = seq_rep1.view(seq_rep1.size(0)*seq_rep1.size(1),seq_rep1.size(2),seq_rep1.size(3))
        
        seq_rep2 = seq_rep2.unsqueeze(2)# B x L x 1 x dims
        seq_rep2 = seq_rep2.view(seq_rep2.size(0)*seq_rep2.size(1),seq_rep2.size(2),seq_rep2.size(3))
        tgt_ebd = tgt_ebd.view(tgt_ebd.size(0)*tgt_ebd.size(1),-1,tgt_ebd.size(3))
        
        score1 = seq_rep1.bmm(tgt_ebd.permute([0,2,1])) # BL x 1 x N
        score1 = score1.squeeze(1).view(B,L,-1) # B,L,N
        score2 = seq_rep2.bmm(tgt_ebd.permute([0,2,1])) # BL x 1 x N
        score2 = score2.squeeze(1).view(B,L,-1) # B,L,N
        
        #Basic loss
        pos_score1 = score1[:,:,[0]]
        neg_score1 = score1[:,:,1:]
        numerator1 =  pos_score1 # (*, 1)
        denominator1 = (neg_score1.exp().sum(-1,keepdims=True) + pos_score1.exp()).log() #[*, 1]

        rec_loss = torch.cat((rec_loss, -((numerator1 - denominator1))),0)
        

        pos_score2 = score2[:,:,[0]]
        neg_score2 = score2[:,:,1:]
        numerator2 =  pos_score2 # (*, 1)
        denominator2 = (neg_score2.exp().sum(-1,keepdims=True) + pos_score2.exp()).log() #[*, 1]
        rec_loss = torch.cat((rec_loss, -((numerator2 - denominator2))),0)
        
        #Regularized dropout loss
        soft_targets1 = F.softmax(score1, dim=-1).detach()
        log_probs1 = F.log_softmax(score1, dim = -1)

        soft_targets2 = F.softmax(score2, dim=-1).detach()
        log_probs2 = F.log_softmax(score2, dim = -1)

        rd_loss = F.kl_div(log_probs2, soft_targets1, reduction='none')[loss_mask.squeeze(-1)].mean() + F.kl_div(log_probs1, soft_targets2, reduction='none')[loss_mask.squeeze(-1)].mean()
        
        #Seququence to Sequence Distribution align
        seq_rep_last_1 = seq_rep1[:,-1,:] # B,D
        seq_rep_last_2 = seq_rep2[:,-1,:] # B,D
        
        score_uu1 = seq_rep_last_1.mm(torch.transpose(seq_rep_last_1,0,1))  # user-user score1.
        soft_targets_uu1 = F.softmax(score_uu1, dim=-1).detach()  # user-user distribution
        log_probs_uu1 = F.log_softmax(score_uu1, dim = -1)
        
        score_uu2 = seq_rep_last_2.mm(torch.transpose(seq_rep_last_2,0,1))  # user-user score1.
        soft_targets_uu2 = F.softmax(score_uu2, dim=-1).detach()  # user-user distribution
        log_probs_uu2 = F.log_softmax(score_uu2, dim = -1)
        
        dr_loss = F.kl_div(log_probs_uu2, soft_targets_uu1, reduction='none').mean() + F.kl_div(log_probs_uu1, soft_targets_uu2, reduction='none').mean()
        
        return rec_loss.mean(), 0.5*rd_loss.mean(), 0.5*dr_loss.mean()