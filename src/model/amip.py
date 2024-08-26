import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pickle
import math
from .layers import TransformerEncoder as TE
from .layers import FeedForward as FF

from utils.inference_utils import map_index
from utils.inference_utils import filter_sequence
from utils.loader_utils import load_mapping_info
from collections import namedtuple

class AMIP(nn.Module):
    
    def __init__(self, args):
        
        super(AMIP, self).__init__()
        self.args = args
        self.mask_prob = self.args.mask_prob
        self.mtype = 'JOIN'
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        torch.cuda.set_device(self.device)
        
        #model parameters
        self.V = nn.Embedding(self.args.num_items+1, self.args.dims, padding_idx = 0)# +1 is mask token
        self.P = nn.Embedding(self.args.window_length+1, self.args.dims , padding_idx=0)
        self.P2= nn.Embedding(self.args.window_length+1, self.args.dims , padding_idx=0)
        self.P3= nn.Embedding(self.args.window_length+1, self.args.dims , padding_idx=0)
        
        self.T = nn.Embedding(4, self.args.dims)# Task-specific embedding
        self.T2 = nn.Embedding(4, self.args.dims)# Task-specific embedding
        
        
        self.task_norm = nn.ModuleList([nn.LayerNorm(self.args.dims),
                                        nn.LayerNorm(self.args.dims),
                                        nn.LayerNorm(self.args.dims),
                                        nn.LayerNorm(self.args.dims)])
        
        self.task_dropout = nn.ModuleList([nn.Dropout(self.args.dropout),
                                           nn.Dropout(self.args.dropout),
                                           nn.Dropout(self.args.dropout),
                                           nn.Dropout(self.args.dropout)])
        
        self.task_lin  = nn.ModuleList([nn.Linear(self.args.dims, self.args.dims),
                                           nn.Linear(self.args.dims, self.args.dims),
                                           nn.Linear(self.args.dims, self.args.dims)])
        
        self.seq_transformer_encoder = TE(n_layers = self.args.encoder_layers,
                                          n_heads = self.args.heads,
                                          hidden_size = self.args.dims,
                                          inner_size = self.args.dims*4,
                                          hidden_dropout_prob = self.args.dropout,
                                          attn_dropout_prob = 0.0,
                                          hidden_act = 'gelu',
                                          bidirectional=True)
 
        self.mask_index = torch.cuda.LongTensor([self.args.num_items])
        self.dropout_layer = nn.Dropout(self.args.dropout)
        
        self.act = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(self.args.dims)
        
        self.lin = nn.Linear(self.args.dims*2,self.args.dims)
        #etc
        self.pad_sequence = torch.zeros(self.args.window_length, dtype = int, device = self.device)
        self.all_items = torch.arange(self.args.num_items, device = self.device)
        
        self.apply(self._init_weights)
        self.V.weight.data[0] = 0.
        self.T.weight.data.normal_(1., 1./self.T.embedding_dim)
        self.T2.weight.data.normal_(0., 1./self.T2.embedding_dim)
    
    
    def dim_selection(self):
        
        var = torch.var(self.V.weight,dim=0)
        indices = torch.topk(-var,k=self.args.dims //2).indices
        selection = torch.zeros(self.args.dims, dtype=torch.int32, device = "cuda")
        selection[indices] = 2.
        
        return selection
    
    def gelu(self, x):
        """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
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
        attention_mask = (item_seq != 0) & (item_seq != self.mask_index)
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
        mask_condition = (mask==0)
        masked_seq = (~mask_condition)*seq + (mask_condition)*self.mask_index
        return (masked_seq, mask)
    
    def append_mask(self, seq, mask):
        
        B,L = seq.size()
        masked_seq = torch.cat((seq, self.mask_index.repeat(B,1)),1)
        mask = torch.cat(( mask, torch.zeros(B,1, dtype=torch.bool).cuda()),1)#[B,L+1]
        
        return (masked_seq, mask)
    
    def get_contextualized_rep(self, user_indices, item_seq_indices, aug_seq_indices = None, pred_opt = 'single' ):
        
        '''
         predict the score at mask [*, N]
        
        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (* X N) target_item_indices:loss mask로 마스크된 위치의 target들
        :param torch.BoolTensor (B x L) loss_mask: True if the element is masked
        '''
        B,L = item_seq_indices.size()
        device_ = item_seq_indices.device
        position_ids = torch.arange(L , dtype=torch.long, device = device_)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq_indices)
        position_ebd = self.P(position_ids)
        
        if pred_opt == "aug":
            seq_ebd = self.V(item_seq_indices) + position_ebd
            seq_ebd = self.dropout_layer(self.layer_norm(seq_ebd))
            position_ebd2 = self.P2(position_ids)
            position_ebd3 = self.P3(position_ids)
            
            aug_ebd = self.V(aug_seq_indices) + torch.cat((position_ebd, position_ebd),1)
            aug_ebd = self.dropout_layer(self.layer_norm(aug_ebd))
            
            seq_ebd = torch.cat((aug_ebd,seq_ebd),1)
            seq_rep = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask( torch.cat( (aug_seq_indices, item_seq_indices),1) ))[-1]
        
        else:
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
        
        if task_number != -1:
            ret = (task_layer*rep + task_bias)
        else:
            #ret = (self.task_norm[task_number](rep))
            ret = self.task_dropout[task_number](self.task_norm[task_number](rep))
        
        return ret

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
        position_ids = torch.arange(L , dtype=torch.long, device = item_seq_indices.device)
        position_ids = position_ids.unsqueeze(0).expand_as((item_seq_indices))
        position_ebd = self.P(position_ids)
        
        tgt_ebd = self.V(target_item_indices) #[N(1), 1, 64]
        
        seq_ebd = self.V(item_seq_indices)+position_ebd
        
        seq_ebd = self.layer_norm(seq_ebd)
        seq_rep = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask(item_seq_indices))[-1]
        
        # rec_heads: batch x dims (e.g., 500, 64)
        # tgt_ebd : all_Items x dims (e.g., 32980, 1, 64)
        rep4next = seq_rep[:,-2,:]
        rep4next = self.get_task_contextualized_rep(rep4next, task_number = 2)

        rep4tgt = seq_rep[:,-1,:]
        rep4tgt = self.get_task_contextualized_rep(rep4tgt, task_number = 1)
        
        rec_heads = self.get_task_contextualized_rep(rep4next + rep4tgt, task_number = -1)
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

    def sample_augmented_items(self,sparse_matrix):
        indices = sparse_matrix._indices()
        row_count = sparse_matrix.size(0)
    
        # Initialize a tensor to store the samples
        samples = torch.full((row_count,), -1, dtype=torch.long, device=sparse_matrix.device)
        
        # Get unique rows and their counts
        unique_rows, row_counts = torch.unique(indices[0], return_counts=True)
        
        # Create an array of start indices for each row's non-zero elements
        row_start_indices = torch.cat([torch.tensor([0], device=sparse_matrix.device), torch.cumsum(row_counts, dim=0)[:-1]])
        
        # Sample a random index for each row
        random_indices_within_rows = torch.floor(torch.rand(unique_rows.size(0), device=sparse_matrix.device) * row_counts).long()
        
        # Calculate the absolute positions of the sampled indices
        sampled_indices_positions = row_start_indices + random_indices_within_rows
        
        # Fetch the corresponding column indices
        sampled_column_indices = indices[1][sampled_indices_positions]
        
        # Assign the sampled column indices to the correct rows in the samples tensor
        samples[unique_rows] = sampled_column_indices

        return samples
    
    def sampled_ce_loss(self, scores):
        #scores = B,L+1,1
        
        numerator =  scores[:,0:1] # (*, 1)
        denominator = (scores[:,1:].exp().sum(-1,keepdims=True) + scores[:,0:1].exp() ).log() #[*, 1]
        loss = -((numerator - denominator))
        return loss
    

    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

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
        amip_loss2 = torch.cuda.DoubleTensor()
        aug_amip_loss = torch.cuda.DoubleTensor()
        mip_loss = torch.cuda.DoubleTensor()

        
        pad_filter = (item_seq_indices.sum(-1)!=0)
        item_seq_indices = item_seq_indices[pad_filter]
        pos_item_indices = pos_item_indices[pad_filter]
        neg_item_indices = neg_item_indices[pad_filter]
        
        selection = self.dim_selection().cuda()
        B = item_seq_indices.size(0)
        '''
        if (aug_item_indices > self.args.num_items).sum() > 0:
            breakpoint()
        if (aug_item_indices == -1).sum() >0 :
            breakpoint()
        '''
        
        S_u = torch.cat((item_seq_indices,pos_item_indices),1)# S_u
        
        masked_item_seq_indices, input_mask = self.mask_sequence(S_u, mask_prob)
        #masked_item_seq_indices, input_mask = self.mask_sequence(item_seq_indices, mask_prob)
        #masked_item_seq_indices, input_mask = self.append_mask(masked_item_seq_indices, input_mask)#맨마지막이 mask
        
        masked_items = (masked_item_seq_indices == self.mask_index)*S_u + (masked_item_seq_indices != self.mask_index)* 0
        
        # Target Item Constuction
        neg_item_indices = neg_item_indices.unsqueeze(1).expand(-1,item_seq_indices.size(1)+1,-1) # B,L+1,N   
        #Next
        next_pos_item_indices = torch.cat((item_seq_indices[:,1:], pos_item_indices, pos_item_indices),1).unsqueeze(2) #(B,L+1,1) #Next Item
        next_tgt_item_indices = torch.cat((next_pos_item_indices, neg_item_indices),-1)
        #Prev
        prev_pos_item_indices = torch.cat((pos_item_indices, item_seq_indices[:,:]),1).unsqueeze(2) #(B,L+1,1) #Prev Item
        prev_tgt_item_indices = torch.cat((prev_pos_item_indices, neg_item_indices),-1)
        
        # Loss Mask Construction
        false_mask = torch.zeros((input_mask.size(0),1) ,device = input_mask.device).bool()
        item_loss_mask = (masked_item_seq_indices != 0) & (masked_item_seq_indices != self.mask_index)
        loss_mask_next = (torch.cat(( (masked_item_seq_indices[:,1:] == self.mask_index), false_mask),-1)) & item_loss_mask #next,아이템 조건
        loss_mask_prev = (torch.cat((false_mask, (masked_item_seq_indices[:,:-1] == self.mask_index)),-1)) & item_loss_mask #prev,아이템 조건
        
        loss_mask_self = (masked_item_seq_indices == self.mask_index)
        transformer_rep = self.get_contextualized_rep(user_indices, masked_item_seq_indices, None, pred_opt = 'single')
        
        #Get score (next-token)
        rep4next = transformer_rep[loss_mask_next].unsqueeze(1) # [*,1,dims]
        rep4next = self.get_task_contextualized_rep(rep4next, task_number = 2)
        
        rep4next_tgt = transformer_rep[loss_mask_next.roll(1,1)].unsqueeze(1)
        rep4next_tgt = self.get_task_contextualized_rep(rep4next_tgt, task_number = 1)
        
        rep4next_f = self.get_task_contextualized_rep(rep4next + rep4next_tgt, task_number = -1)
        
        tgt_next = self.V(next_tgt_item_indices)[loss_mask_next]
        score_next = rep4next_f.bmm(tgt_next.permute([0,2,1])).squeeze(1) #[*,1+N]
        amip_loss = torch.cat((amip_loss, self.sampled_ce_loss(score_next)), 0)
        
        #Get score (prev-token)
        rep4prev = transformer_rep[loss_mask_prev].unsqueeze(1) # [*,1,dims]
        rep4prev = self.get_task_contextualized_rep(rep4prev, task_number = 0)
        rep4prev_tgt = transformer_rep[loss_mask_prev.roll(-1,1)].unsqueeze(1)
        rep4prev_tgt = self.get_task_contextualized_rep(rep4prev_tgt, task_number = 1)
        
        rep4prev_f = self.get_task_contextualized_rep(rep4prev + rep4prev_tgt, task_number = -1)
        tgt_prev = self.V(prev_tgt_item_indices)[loss_mask_prev]
        score_prev = rep4prev_f.bmm(tgt_prev.permute([0,2,1])).squeeze(1) #[*,1+N]          
        amip_loss2 = torch.cat((amip_loss2, self.sampled_ce_loss(score_prev)),0)
        

        return amip_loss.mean(), amip_loss2.mean()