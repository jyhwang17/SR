import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pickle
from .layers import TransformerEncoder as TE
from utils.inference_utils import map_index
from utils.inference_utils import filter_sequence
from utils.loader_utils import load_mapping_info
from collections import namedtuple

class BERT4REC(nn.Module):
    
    def __init__(self, args):
        super(BERT4REC, self).__init__()
        #BERT4REC wit relative position bias
        self.args = args
        self.mask_prob = self.args.mask_prob
        self.mtype = 'JOIN'
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        torch.cuda.set_device(self.device)
        
        #model parameters
        self.V = nn.Embedding(self.args.num_items+1, self.args.dims, padding_idx = 0)# +1 is mask token
        self.Vb= nn.Embedding(self.args.num_items+1, 1, padding_idx = 0)
        self.P = nn.Embedding(self.args.window_length+2, self.args.dims , padding_idx=0)
        
        self.V.weight.data.normal_(0., 1./self.V.embedding_dim)
        self.V.weight.data[0] = 0.

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
        self.projection = nn.Linear(self.args.dims, self.args.dims)
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

    def masked_prediction(self, user_indices, item_seq_indices, target_item_indices, loss_mask):
        
        '''
         predict the score at mask [*, N]
        
        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (* X N) target_item_indices:loss mask로 마스크된 위치의 target들
        :param torch.BoolTensor (B x L) loss_mask: True if the element is masked
        '''
        
        B,L = item_seq_indices.size()
        tgt_ebd = self.V(target_item_indices)
        tgt_bias = self.Vb(target_item_indices)
        
        seq_ebd = self.V(item_seq_indices)
        seq_ebd = self.layer_norm(seq_ebd)
        seq_ebd = self.dropout_layer(seq_ebd)
        seq_rep = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask(item_seq_indices))[-1]
        
        # [#of mask, dims]
        seq_rep = seq_rep[loss_mask] # [*, dims]
        seq_rep = self.projection(seq_rep)

        rec_heads = seq_rep.unsqueeze(1)#[*, 1, dims]
        rel_score = rec_heads.bmm(tgt_ebd.permute([0,2,1])).squeeze(1) + tgt_bias.squeeze(-1) #[*, N]
        
        return rel_score
    
    def next_item_prediction(self, user_indices, item_seq_indices, target_item_indices, target_domain):
        
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
        tgt_ebd = self.V(target_item_indices) #[N(1), 1, 64]
        tgt_bias = self.Vb(target_item_indices) #[N(1),1, 1]
        
        seq_ebd = self.V(item_seq_indices)
        seq_ebd = self.layer_norm(seq_ebd)
        seq_ebd = self.dropout_layer(seq_ebd)
        seq_rep = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask(item_seq_indices))[-1]
        seq_rep = self.projection(seq_rep)

        # rec_heads: batch x dims (e.g., 500, 64)
        # tgt_ebd : all_Items x dims (e.g., 32980, 1, 64)
        rec_heads = seq_rep[:,-1,:]
        rel_score = rec_heads.mm(tgt_ebd.squeeze(1).t()) + tgt_bias.squeeze(1).t() # batch x all_Items
        return rel_score
        
    def forward(self, user_indices, item_seq_indices, target_item_indices, target_domain, pred_opt = 'eval'):

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
            return self.next_item_prediction(user_indices, item_seq_indices, target_item_indices, target_domain)
    
    def loss(self, user_indices, sorted_item_seq_indices, item_seq_indices, pos_item_indices, neg_item_indices):
        
        '''
        compute loss

        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (B x 1) pos_item_indices:
        :param torch.LongTensor (B x N) neg_item_indices:
        :return: loss
        '''

        B,L = item_seq_indices.size() 
        mask_prob = self.mask_prob
        masked_item_seq_indices, input_mask = self.mask_sequence(item_seq_indices, mask_prob)
        masked_item_seq_indices, input_mask = self.append_mask(masked_item_seq_indices, input_mask)
        
        pos_tgt_item_indices = torch.cat((item_seq_indices, pos_item_indices),1).unsqueeze(2)#(B,L+1,1)
        pos_tgt_item_indices = pos_tgt_item_indices[(~input_mask)]  # [*, 1]

        #
        neg_item_indices = neg_item_indices.unsqueeze(1)#B,1,N
        neg_tgt_item_indices = neg_item_indices.expand(-1,item_seq_indices.size(1)+1,-1) # B,L+1,N
        neg_tgt_item_indices = neg_tgt_item_indices[(~input_mask)]  # [*, N]
        
        pos_score = self.masked_prediction(user_indices, masked_item_seq_indices,
                                           pos_tgt_item_indices, ~input_mask) #[*, 1]
        neg_score = self.masked_prediction(user_indices, masked_item_seq_indices,
                                           neg_tgt_item_indices, ~input_mask) #[*, N]

        numerator =  pos_score # (*, 1)
        denominator = (neg_score.exp().sum(-1,keepdims=True) + pos_score.exp()).log() #[*, 1]

        loss = -((numerator - denominator)) # [*,1]

        return loss.mean()