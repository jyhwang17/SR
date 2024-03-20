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
        
        B,L = item_seq_indices.size() 
        total_sequence = torch.cat((item_seq_indices, pos_item_indices),1) #[B,L+1]
        neg_tgt_item_indices = neg_item_indices.unsqueeze(1).expand(-1,item_seq_indices.size(1),-1) #[B,L,L]
        
        pad_filter = (total_sequence.sum(-1)!=0)
        total_sequence = total_sequence[pad_filter]
        neg_tgt_item_indices = neg_tgt_item_indices[pad_filter]
        
        tgt_item_indices = torch.cat((total_sequence[:,1:].unsqueeze(-1), neg_tgt_item_indices),2)
        
        loss_mask = (total_sequence[:,:-1] != 0)# B L
        score = self.forward(user_indices,
                                 total_sequence[:,:-1],#[B,L]
                                 tgt_item_indices,#[B,L,1+N]
                                 pred_opt='training') #=>[B,L,1+N]
        
        pos_score = score[:,:,[0]]
        neg_score = score[:,:,1:]
        numerator =  pos_score # (*, 1)
        denominator = (neg_score.exp().sum(-1,keepdims=True) + pos_score.exp()).log() #[*, 1]
        loss = -((numerator - denominator)) # [*,1]
        
        
        score2 = self.forward(user_indices,
                              total_sequence[:,:-1],#[B,L]
                              tgt_item_indices,#[B,L,1+N]
                              pred_opt='training') #=>[B,L,1+N]
            
        soft_targets = F.softmax(sp_scores[:,:,:self.args.negs], dim=-1)
        student_log_probs = F.log_softmax(sh_scores[:,:,:self.args.negs], dim = -1)

        distillation = F.kl_div(student_log_probs, soft_targets, reduction='none')
        ret_reg = torch.cat((ret_reg, distillation[loss_mask.squeeze(-1)]),0)
            
        return loss.mean()