import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pickle
from .layers import TransformerEncoder as TE
from .layers import FeedForward as FF

from utils.inference_utils import map_index
from utils.inference_utils import filter_sequence
from utils.loader_utils import load_mapping_info
from collections import namedtuple

class TC4REC(nn.Module):
    
    def __init__(self, args):
        
        super(TC4REC, self).__init__()
        
        self.args = args
        self.mask_prob = self.args.mask_prob
        self.mtype = 'JOIN'
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        torch.cuda.set_device(self.device)
        
        #model parameters
        self.V = nn.Embedding(self.args.num_items+1, self.args.dims, padding_idx = 0)# +1 is mask token
        self.V.weight.data.normal_(0., 1./self.V.embedding_dim)
        self.V.weight.data[0] = 0.
        
        self.P = nn.Embedding(self.args.window_length+1, self.args.dims , padding_idx=0)
        self.P.weight.data.normal_(0., 1./self.V.embedding_dim)
        


        self.seq_transformer_encoder = TE(n_layers = self.args.encoder_layers,
                                          n_heads = 2,
                                          hidden_size = self.args.dims,
                                          inner_size = self.args.dims*4,
                                          hidden_dropout_prob = self.args.dropout,
                                          attn_dropout_prob = 0.0,
                                          hidden_act = 'gelu',
                                          bidirectional=True)
 
        self.mask_index = torch.cuda.LongTensor([self.args.num_items])
        
        self.dropout_layer = nn.Dropout(self.args.dropout)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(self.args.dims)

        #etc
        self.pad_sequence = torch.zeros(self.args.window_length, dtype = int, device = self.device)
        self.all_items = torch.arange(self.args.num_items, device = self.device)
        
    def get_attention_mask(self, item_seq, bidirectional = True):

        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask
    
    def mask_sequence(self, seq, mask_prob):
        
        # false is the mask
        B,L = seq.size()
        mask = torch.rand((B,L), device = self.device ).ge(mask_prob).cuda()
        mask = (seq == 0) | mask # pad에는 mask를 씌우지 않음.
        masked_seq = mask*seq + (mask==0)*self.mask_index
        return (masked_seq, mask)
    
    def append_mask(self, seq, mask):
        
        B,L = seq.size()
        masked_seq = torch.cat((seq, self.mask_index.repeat(B,1)),1)
        mask = torch.cat(( mask, torch.zeros(B,1, dtype=torch.bool).cuda()),1)#[B,L+1]
        
        return (masked_seq, mask)
    
    def get_contextualized_rep(self, user_indices, item_seq_indices):
        
        '''
         predict the score at mask [*, N]
        
        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (* X N) target_item_indices:loss mask로 마스크된 위치의 target들
        :param torch.BoolTensor (B x L) loss_mask: True if the element is masked
        '''
        B,L = item_seq_indices.size()
        position_ids = torch.arange( L , dtype=torch.long, device = item_seq_indices.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq_indices)
        position_ebd = self.P(position_ids)
        
        seq_ebd = self.V(item_seq_indices) + position_ebd
        
        seq_ebd = self.layer_norm(seq_ebd)
        seq_ebd = self.dropout_layer(seq_ebd)
        seq_rep = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask(item_seq_indices))[-1]
        
        return seq_rep
    
    def masked_prediction(self, user_indices, item_seq_indices, target_item_indices, loss_mask, pred_opt = 'Forward'):
        
        '''
         predict the score at mask [*, N]
        
        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (* X N) target_item_indices:loss mask로 마스크된 위치의 target들
        :param torch.BoolTensor (B x L) loss_mask: True if the element is masked
        '''
        
        B,L = item_seq_indices.size()
        position_ids = torch.arange( L , dtype=torch.long, device = item_seq_indices.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq_indices)
        position_ebd = self.P(position_ids)
        
        tgt_ebd = self.V(target_item_indices)
        seq_ebd = self.V(item_seq_indices) + position_ebd
        
        seq_ebd = self.layer_norm(seq_ebd)
        seq_ebd = self.dropout_layer(seq_ebd)
        seq_rep = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask(item_seq_indices))[-1]
        
        # [#of mask, dims]
        #if pred_opt == 'Forward':
        seq_rep = seq_rep[loss_mask] # [*, dims]
        '''    
        else:
            seq_rep = self.projection(seq_rep[loss_mask])
        '''
        rec_heads = seq_rep.unsqueeze(1)#[*, 1, dims]
        rel_score = rec_heads.bmm(tgt_ebd.permute([0,2,1])).squeeze(1)  #[*, N]
        
        return rel_score
    
    def next_item_prediction(self, user_indices, item_seq_indices, target_item_indices):
        
        '''
         predict next item score
        
        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (N x 1) target_item_indices:
        
        '''
        
        item_seq_indices,_ = self.append_mask(item_seq_indices, 
                                              torch.ones_like(item_seq_indices, dtype=torch.bool,
                                                              device = self.device))
        
        B,L = item_seq_indices.size()
        position_ids = torch.arange( L , dtype=torch.long, device = item_seq_indices.device)
        position_ids = position_ids.unsqueeze(0).expand_as((item_seq_indices))
        position_ebd = self.P(position_ids)
        
        tgt_ebd = self.V(target_item_indices) #[N(1), 1, 64]
        
        seq_ebd = self.V(item_seq_indices)+position_ebd
        
        seq_ebd = self.layer_norm(seq_ebd)
        seq_ebd = self.dropout_layer(seq_ebd)
        seq_rep = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask(item_seq_indices))[-1]
        
        # rec_heads: batch x dims (e.g., 500, 64)
        # tgt_ebd : all_Items x dims (e.g., 32980, 1, 64)
        rec_heads = seq_rep[:,-2,:]
        
        rel_score = rec_heads.mm(tgt_ebd.squeeze(1).t()) # batch x all_Items
        return rel_score
        
    def forward(self, user_indices, item_seq_indices, target_item_indices, pred_opt = 'eval'):

        '''
        compute model outputs
        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (B x L x N) or (N x 1) target_item_indices: The shape is [B,L,N] in training, and [N,1] in test.
        '''

        if pred_opt == 'training':
            #refactoring..
            return None
            #return self.masked_prediction(user_indices, item_seq_indices, target_item_indices, input_mask = None)
        else:
            return self.next_item_prediction(user_indices, item_seq_indices, target_item_indices)
    
    def loss(self, user_indices, item_seq_indices, pos_item_indices, neg_item_indices):
        

        '''
        compute loss

        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (B x 1) pos_item_indices:
        :param torch.LongTensor (B x N) neg_item_indices:
        :return: loss
        '''
        
        mask_prob = self.mask_prob
        amip_loss = torch.cuda.DoubleTensor()
        mip_loss = torch.cuda.DoubleTensor()
        consistency_loss = torch.cuda.DoubleTensor()
        
        pad_filter = (item_seq_indices.sum(-1)!=0)
        item_seq_indices = item_seq_indices[pad_filter]
        pos_item_indices = pos_item_indices[pad_filter]
        neg_item_indices = neg_item_indices[pad_filter]
        
        
        for i in range(self.args.shots):
            masked_item_seq_indices, input_mask = self.mask_sequence(item_seq_indices, mask_prob)
            masked_item_seq_indices, input_mask = self.append_mask(masked_item_seq_indices, input_mask)#맨마지막이 mask
            
            pos_tgt_item_indices1 = torch.cat((item_seq_indices[:,1:],
                                               pos_item_indices, pos_item_indices),1).unsqueeze(2) #(B,L+1,1) #Next Item
            pos_tgt_item_indices2 = torch.cat((pos_item_indices, item_seq_indices[:,:]),1).unsqueeze(2) #(B,L+1,1) #Prev Item
            
            pos_tgt_item_indices3 = torch.cat((item_seq_indices[:,:],pos_item_indices),1).unsqueeze(2) #(B,L+1,1) #Self Item
            
            neg_tgt_item_indices = neg_item_indices.unsqueeze(1).expand(-1,item_seq_indices.size(1)+1,-1) # B,L+1,N
            
            tgt_item_indices1 = torch.cat((pos_tgt_item_indices1,neg_tgt_item_indices),-1)
            tgt_item_indices2 = torch.cat((pos_tgt_item_indices2,neg_tgt_item_indices),-1)
            tgt_item_indices3 = torch.cat((pos_tgt_item_indices3,neg_tgt_item_indices),-1)
            
            false_mask = torch.zeros((input_mask.size(0),1) ,device = input_mask.device).bool()
            loss_mask1 = (torch.cat((~input_mask[:,1:],false_mask),-1))  # next 아이템이 mask인지 아닌지
            loss_mask2 = (torch.cat((false_mask, ~input_mask[:,:-1]),-1)) # prev, 아이템이 mask인지 아닌지
            loss_mask3 = ~input_mask
            
            #Get representation
            transformer_rep = self.get_contextualized_rep(user_indices, masked_item_seq_indices)
            
            # Get score (next-token)
            rep1  = transformer_rep[loss_mask1].unsqueeze(1) # [*,1,dims]
            tgt_ebd1 = self.V(tgt_item_indices1)[loss_mask1]
            score1 = rep1.bmm(tgt_ebd1.permute([0,2,1])).squeeze(1) #[*,1+N]
            numerator1 =  score1[:,[0]] # (*, 1)
            denominator1 = (score1[:,1:].exp().sum(-1,keepdims=True) + score1[:,[0]].exp() ).log() #[*, 1]
            loss1 = -((numerator1 - denominator1))
            # (prev-token)
            rep2  = transformer_rep[loss_mask2].unsqueeze(1) # [*,1,dims]
            tgt_ebd2 = self.V(tgt_item_indices2)[loss_mask2]
            score2 = rep2.bmm(tgt_ebd2.permute([0,2,1])).squeeze(1) #[*,1+N]
            numerator2 =  score2[:,[0]] # (*, 1)
            denominator2 = (score2[:,1:].exp().sum(-1,keepdims=True) + score2[:,[0]].exp() ).log() #[*, 1]
            loss2 = -((numerator2 - denominator2))
            
            rep3 = transformer_rep[loss_mask3].unsqueeze(1)
            tgt_ebd3 = self.V(tgt_item_indices3)[loss_mask3]
            score3 = rep3.bmm(tgt_ebd3.permute([0,2,1])).squeeze(1)
            numerator3 = score3[:,[0]]
            denominator3 = (score3[:,1:].exp().sum(-1,keepdims=True) + score3[:,[0]].exp() ).log() #[*, 1]
            loss3 = -((numerator3 - denominator3))
            
            amip_loss = torch.cat((amip_loss, loss1, loss2),0)
            mip_loss = torch.cat((mip_loss, loss3),0)
        
            
            # Task-oriented Consistency
            
            # Self-Prediction
            task_vector3 = loss_mask3[:,1:self.args.window_length] # self, B,L-1
            task_vector3 = torch.cat((false_mask,task_vector3,false_mask),-1)# B,L+1
            
            #
            task_vector2 = torch.roll(loss_mask3,-1,-1)[:,0:self.args.window_length-1] #right-item prediction
            task_vector2 = torch.cat((task_vector2, false_mask, false_mask),-1)
            
            task_vector3 = torch.roll(loss_mask3,1,-1)[:,2:] #size check self-item
            task_vector3 = torch.cat((false_mask, false_mask, task_vector3),-1)
            
            #task_rep = 0.333333333333*(transformer_rep[task_vector1] + transformer_rep[task_vector2]
            #+ transformer_rep[task_vector3]) #[*, D]
            r_predictor = transformer_rep[task_vector1] 
            l_predictor = transformer_rep[task_vector2]
            s_predictor = transformer_rep[task_vector3]
            
            task_rep = 0.5*(transformer_rep[task_vector1]+ transformer_rep[task_vector3]) #[*, D]
            
            task_items = tgt_item_indices3[task_vector1][:,0] #[*]
            ans_ebd = self.V(task_items) #[*,D] 
            
            # ITEM[*,D] x SEQ[*,D] = item-seq score[*,*]
            task_scores = ans_ebd.mm(torch.transpose(task_rep,0,1))
            diag_mask =  torch.eye(task_scores.size(0)).cuda()# 이게 문제인데.....내일 다시 검토하기.
            ones_mask =  torch.ones(task_scores.size()).cuda()
            diag_mask = ones_mask - diag_mask
            task_neg_scores = (task_scores.exp()*diag_mask).sum(1,keepdims=True) + torch.transpose((task_scores.exp()*diag_mask),0,1).sum(1,keepdims=True)
            
            #[*,1,D] x [*,D,1+N]: [*, 1, 1+N]
            task_pos_scores = torch.diagonal(task_scores,0).unsqueeze(1)
            numerator4 = task_pos_scores
            denominator4 = (task_neg_scores + task_pos_scores.exp()).log()
            loss4 = -((numerator4 - denominator4))
            
            consistency_loss = torch.cat((consistency_loss, loss4),0)

        return amip_loss.mean(), mip_loss.mean(), consistency_loss.mean()