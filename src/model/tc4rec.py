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
        self.P.weight.data.normal_(0., 1./self.P.embedding_dim)  
        self.T = nn.Embedding(3, self.args.dims)# Task-specific embedding
        self.T2 = nn.Embedding(3, self.args.dims)# Task-specific embedding
        
       
        
        self.task_norm = nn.ModuleList([nn.LayerNorm(self.args.dims),
                                        nn.LayerNorm(self.args.dims),
                                        nn.LayerNorm(self.args.dims)])
        
        self.task_dropout = nn.ModuleList([nn.Dropout(self.args.dropout),
                                           nn.Dropout(self.args.dropout),
                                           nn.Dropout(self.args.dropout)])
        
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
        
        self.apply(self._init_weights)
        self.V.weight.data[0] = 0.
        self.T.weight.data.normal_(1., 1./self.T.embedding_dim)
        self.T2.weight.data.normal_(0., 1./self.T2.embedding_dim)

    def _init_weights(self, module):
        
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=(1./self.args.dims))
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

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
    
    def get_task_contextualized_rep(self, rep, task_number = 0):
        
        task_layer = self.T.weight[task_number]
        task_layer = task_layer.expand_as(rep)
        
        task_bias = self.T2.weight[task_number]
        task_bias = task_bias.expand_as(rep)
        
        ret = self.task_dropout[task_number](self.task_norm[task_number]((task_layer*rep + task_bias)))

        return ret
    
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
        rep4next = seq_rep[:,-2,:]
        rep4next = self.get_task_contextualized_rep(rep4next, task_number = 2)

        rep4self = seq_rep[:,-1,:]
        rep4self = self.get_task_contextualized_rep(rep4self, task_number = 1)
        rec_heads = rep4next + rep4self
        
        rel_score = rec_heads.mm(tgt_ebd.squeeze(1).t()) # batch x all_Items
        
        #prob = nn.Softmax(dim=-1)
        #rel_score = rec_heads.mm(tgt_ebd.squeeze(1).t()) # batch x all_Items
        #rel_score = prob(rep4next.mm(tgt_ebd.squeeze(1).t())) + prob(rep4self.mm(tgt_ebd.squeeze(1).t()))
        
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

        
        pad_filter = (item_seq_indices.sum(-1)!=0)
        item_seq_indices = item_seq_indices[pad_filter]
        pos_item_indices = pos_item_indices[pad_filter]
        neg_item_indices = neg_item_indices[pad_filter]
        
        #masked_item_seq_indices = torch.cuda.LongTensor()
        #input_mask = torch.cuda.BoolTensor()
        # Sequence Batch Construction
        '''
        for i in range(self.args.shots):
            masked_seqs, masks = self.mask_sequence(item_seq_indices, mask_prob)
            masked_seqs, masks = self.append_mask(masked_seqs, masks)#맨마지막이 mask
            masked_item_seq_indices = torch.cat((masked_item_seq_indices, masked_seqs),0)
            input_mask = torch.cat(( input_mask, masks),0)
        '''
        
        #Sequence Batch Construction
        masked_item_seq_indices, input_mask = self.mask_sequence(item_seq_indices.repeat(self.args.shots,1), mask_prob)
        masked_item_seq_indices, input_mask = self.append_mask(masked_item_seq_indices, input_mask)#맨마지막이 mask
        # Target Item Constuction
        neg_item_indices = neg_item_indices.unsqueeze(1).expand(-1,item_seq_indices.size(1)+1,-1) # B,L+1,N   
        #Next
        next_pos_item_indices = torch.cat((item_seq_indices[:,1:],
                                           pos_item_indices,pos_item_indices),1).unsqueeze(2) #(B,L+1,1) #Next Item
        
        next_tgt_item_indices = torch.cat((next_pos_item_indices,neg_item_indices),-1)
        #Prev
        prev_pos_item_indices = torch.cat((pos_item_indices, item_seq_indices[:,:]),1).unsqueeze(2) #(B,L+1,1) #Prev Item
        prev_tgt_item_indices = torch.cat((prev_pos_item_indices, neg_item_indices),-1)  
        #Self
        self_pos_item_indices = torch.cat((item_seq_indices[:,:], pos_item_indices),1).unsqueeze(2) #(B,L+1,1) #Self Item
        self_tgt_item_indices = torch.cat((self_pos_item_indices, neg_item_indices),-1)
        #Prediction
        next_tgt_item_indices = next_tgt_item_indices.repeat(self.args.shots,1,1)
        prev_tgt_item_indices = prev_tgt_item_indices.repeat(self.args.shots,1,1)
        self_tgt_item_indices = self_tgt_item_indices.repeat(self.args.shots,1,1)
        # For consen
        next_pos_item_indices = next_pos_item_indices.repeat(self.args.shots,1,1)
        prev_pos_item_indices = prev_pos_item_indices.repeat(self.args.shots,1,1)
        self_pos_item_indices = self_pos_item_indices.repeat(self.args.shots,1,1)
        
        # Loss Mask Construction
        false_mask = torch.zeros((input_mask.size(0),1) ,device = input_mask.device).bool()
        item_loss_mask = (masked_item_seq_indices != 0) & (masked_item_seq_indices != self.mask_index)        
        loss_mask_next = (torch.cat((~input_mask[:,1:],false_mask),-1)) & item_loss_mask  # next 아이템이 mask인지 아닌지
        loss_mask_prev = (torch.cat((false_mask, ~input_mask[:,:-1]),-1)) & item_loss_mask # prev, 아이템이 mask인지 아닌지
        loss_mask_self = ~input_mask

        #Get representation
        transformer_rep = self.get_contextualized_rep(user_indices, masked_item_seq_indices)
        
        # Get score (next-token)
        rep4next  = transformer_rep[loss_mask_next].unsqueeze(1) # [*,1,dims]
        rep4next = self.get_task_contextualized_rep(rep4next, task_number = 2)# Next
            
        tgt_next = self.V(next_tgt_item_indices)[loss_mask_next]
        score_next = rep4next.bmm(tgt_next.permute([0,2,1])).squeeze(1) #[*,1+N]
        numerator_next =  score_next[:,[0]] # (*, 1)
        denominator_next = (score_next[:,1:].exp().sum(-1,keepdims=True) + score_next[:,[0]].exp() ).log() #[*, 1]
        loss_next = -((numerator_next - denominator_next))
            
        #Get score (prev-token)
        rep4prev = transformer_rep[loss_mask_prev].unsqueeze(1) # [*,1,dims]
        rep4prev = self.get_task_contextualized_rep(rep4prev, task_number = 0)

        tgt_prev = self.V(prev_tgt_item_indices)[loss_mask_prev]
        score_prev = rep4prev.bmm(tgt_prev.permute([0,2,1])).squeeze(1) #[*,1+N]
        numerator_prev =  score_prev[:,[0]] # (*, 1)
        denominator_prev = (score_prev[:,1:].exp().sum(-1,keepdims=True) + score_prev[:,[0]].exp() ).log() #[*, 1]
        loss_prev = -((numerator_prev - denominator_prev))
        amip_loss = torch.cat((amip_loss, loss_next, loss_prev),0)
        
        #Get score (self-token)
        rep4self = transformer_rep[loss_mask_self].unsqueeze(1)
        rep4self = self.get_task_contextualized_rep(rep4self, task_number = 1)
        
        tgt_self = self.V(self_tgt_item_indices)[loss_mask_self]
        score_self = rep4self.bmm(tgt_self.permute([0,2,1])).squeeze(1)
        numerator_self = score_self[:,[0]]
        denominator_self = (score_self[:,1:].exp().sum(-1,keepdims=True) + score_self[:,[0]].exp() ).log() #[*, 1]
        loss_self = -((numerator_self - denominator_self))
        mip_loss = torch.cat((mip_loss, loss_self),0)
        
        ###########################################################################################################################
        
        #1. Consensus와 item 사이의 relation을 distillation
        #2. item과 consensus들 사이의 relation을 distillation
        
        #Consensus Representation Constuction
        self.eval()
        transformer_rep_t = self.get_contextualized_rep(user_indices, masked_item_seq_indices)
        rep4next_ = transformer_rep_t[loss_mask_next].unsqueeze(1) # [*,1,dims]
        rep4next_ = self.get_task_contextualized_rep(rep4next_, task_number = 2)# Next
        rep4prev_ = transformer_rep_t[loss_mask_prev].unsqueeze(1) # [*,1,dims]
        rep4prev_ = self.get_task_contextualized_rep(rep4prev_, task_number = 0)
        rep4self_ = transformer_rep_t[loss_mask_self].unsqueeze(1)
        rep4self_ = self.get_task_contextualized_rep(rep4self_, task_number = 1)
        batch_rep_t = torch.cat((rep4next_.squeeze(1), rep4prev_.squeeze(1) , rep4self_.squeeze(1) ),0)
        self.train()
        
        # Get score (next-token)
        batch_rep_s = torch.cat((rep4next.squeeze(1), rep4prev.squeeze(1) , rep4self.squeeze(1) ),0)

        batch_msk_indices = torch.cat((next_pos_item_indices[loss_mask_next].flatten(),
                                       prev_pos_item_indices[loss_mask_prev].flatten(),
                                       self_pos_item_indices[loss_mask_self].flatten()),0)
        
        
        batch_tgt_item_indices = torch.cat((next_tgt_item_indices[loss_mask_next],
                                            prev_tgt_item_indices[loss_mask_prev],
                                            self_tgt_item_indices[loss_mask_self]),0)#[*, 1+N]
        mat = (batch_msk_indices.unsqueeze(1) == batch_msk_indices.unsqueeze(0)) #[*, *]
        #[*,1,L]
        seq_mat = (masked_item_seq_indices)
        comp_mat = ~mat
        mat = mat.float()
        comp_mat = comp_mat.float()
        coef_n = mat.sum(-1,keepdim=True)
        
        #Consensus Representation
        prob = nn.Softmax(dim=-1)
        
        tgt_pos_ebd = self.V(batch_msk_indices)# [**, D]
        ebd4att = tgt_pos_ebd.detach().clone()# [**, D]
        attn = torch.matmul(ebd4att, batch_rep_t.t()) ##
        attn_mat = prob(attn + comp_mat*(-100000.0))
        eye_mat = torch.eye(len(mat)).cuda()
        
        #Get Consensus Representation
        
        #rep4consen = torch.matmul(mat, batch_rep_t)/coef_n #mean pooling
        rep4consen = torch.matmul(attn_mat.detach().clone(), batch_rep_t) #[**,1, D] attn_mat.detach() 중요.
        rep4consen = rep4consen.unsqueeze(1)

        # KL-Divergence
        batch_rep_s= batch_rep_s.unsqueeze(1)
        tgt_ebd = torch.cat((tgt_next, tgt_prev, tgt_self),0)
        loss = nn.CrossEntropyLoss()
        
        logit_P = rep4consen.bmm(tgt_ebd.permute([0,2,1])).squeeze(1) #[**, 1+N]
        logit_Q = batch_rep_s.bmm(tgt_ebd.permute([0,2,1])).squeeze(1) #[**, 1+N]
        
        log_P = F.log_softmax(logit_P, dim = -1)
        log_Q = F.log_softmax(logit_Q, dim = -1)
        
        #logit_P = torch.matmul(rep4consen.squeeze(1), tgt_pos_ebd.t())
        #logit_Q = torch.matmul(batch_rep2.squeeze(1), tgt_pos_ebd.t())
        
        logit_IP = torch.matmul(tgt_pos_ebd, rep4consen.squeeze(1).t()) #+1000.0*(eye_mat-mat) #[**, **] 
        logit_IQ = torch.matmul(tgt_pos_ebd, batch_rep_s.squeeze(1).t())#+1000.0*(eye_mat-mat) #[**, **]
        
        log_IP = F.log_softmax(logit_IP, dim = -1)
        log_IQ = F.log_softmax(logit_IQ, dim = -1)
        
        #같은 user안의 시퀀스/아이템은 네거티브 샘플에서 제외시키기.(아직안함)
        #Ensemble의 개수가 2개이상인것만할까

        #cut = (mat.sum(1) >=3)

        #eye = torch.eye(len(mat)).cuda()
        #sup = torch.diagonal(-F.log_softmax(logit_IP + mat*(-10000.0) + eye*(10000.0),dim= -1) , 0)
        ln = len(rep4next) + len(rep4self)
        
        CE_consen = - log_Q*prob(logit_P).detach().clone() #*(~((mat-eye_mat).bool())) # Cross-Entropy.
        CE_item = -log_IQ*prob(logit_IP).detach().clone() #*(~((mat-eye_mat).bool()))# Cross-Entropy.
        
        
        # MIP & AMIP & CONSISTENCY & CONSENSUS
        return amip_loss.mean(), mip_loss.mean(), CE_consen.sum(-1).mean() +CE_item.sum(-1).mean()
