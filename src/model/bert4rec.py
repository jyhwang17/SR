import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pickle
import math
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
                                          n_heads = self.args.heads,
                                          hidden_size = self.args.dims,
                                          inner_size = self.args.dims*4,
                                          hidden_dropout_prob = self.args.dropout,
                                          attn_dropout_prob = 0.0,
                                          hidden_act = 'gelu',
                                          bidirectional=True)
 
        self.mask_index = torch.cuda.LongTensor([self.args.num_items])
        
        self.dropout_layer = nn.Dropout(self.args.dropout)
        self.layer_norm = nn.LayerNorm(self.args.dims)
        self.act = nn.GELU()
        self.projection = nn.Linear(self.args.dims, self.args.dims)
        
        #etc
        self.all_items = torch.arange(self.args.num_items, device = self.device)
        self.apply(self._init_weights)
        self.V.weight.data[0] = 0.
        
    def _init_weights(self, module):
        
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):

            module.weight.data.normal_(mean=0.0, std=1./self.args.dims )
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
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.0)
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
    
    
    def gelu(self, x):
        
        """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
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
        
        seq_rep = self.get_contextualized_rep( user_indices, item_seq_indices)
        rec_heads = seq_rep.unsqueeze(1)#[*, 1, dims]
        rel_score = rec_heads.bmm(tgt_ebd.permute([0,2,1])).squeeze(1) + tgt_bias.squeeze(-1) #[*, N]
        
        return rel_score
    
    def get_rep_list(self, user_indices, item_seq_indices, pos_item_indices):
        
        
        mask_prob = 0.0
        masked_item_seq_indices,m =self.append_mask(item_seq_indices,
                                                    torch.ones_like(item_seq_indices, dtype=torch.bool,
                                                                    device = self.device))
        
        B,L = masked_item_seq_indices.size() 
        tgt_ebd = self.V(pos_item_indices)#
        
        position_ids = torch.arange(L, dtype = torch.long, device = self.device )
        position_ids = position_ids.unsqueeze(0).expand_as(masked_item_seq_indices)
        position_ebd = self.P(position_ids)
        
        seq_ebd = self.V(masked_item_seq_indices) + position_ebd
        seq_ebd = self.layer_norm(seq_ebd)
        seq_ebd = self.dropout_layer(seq_ebd)
        
        seq_rep = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask(masked_item_seq_indices))
        #seq_rep.append(self.gelu(self.projection(seq_rep[-1])) )
        
        return seq_rep, seq_ebd, tgt_ebd.squeeze(1)
    
    def get_contextualized_rep(self, user_indices, item_seq_indices):
        
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
        
        seq_ebd = self.V(item_seq_indices) + position_ebd
        seq_ebd = self.layer_norm(seq_ebd)
        seq_ebd = self.dropout_layer(seq_ebd)
        #seq_rep, i_entropy, m_entropy = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask(item_seq_indices))
        
        seq_rep = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask(item_seq_indices))
        seq_rep = seq_rep[-1]
        #seq_rep = self.projection(seq_rep)
        #seq_rep = self.gelu(seq_rep)
        
        return self.dropout_layer(seq_rep)
    
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
        
        tgt_ebd = self.V(target_item_indices) #[N(1), 1, 64]
        tgt_bias = self.Vb(target_item_indices) #[N(1),1, 1]
        seq_rep = self.get_contextualized_rep(user_indices, item_seq_indices)
        #rec_heads: batch x dims (e.g., 500, 64)
        # tgt_ebd : all_Items x dims (e.g., 32980, 1, 64)
        
        rec_heads = seq_rep[:,-1,:]        
        rel_score = rec_heads.mm(tgt_ebd.squeeze(1).t()) + tgt_bias.squeeze(1).t() # batch x all_Items
        
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

    def get_entropy(self, user_indices, item_seq_indices, pos_item_indices, neg_item_indices):
        
        B,L = item_seq_indices.size()
        
        mask_prob= 0.0
        masked_item_seq_indices, input_mask = self.mask_sequence(item_seq_indices, mask_prob)
        masked_item_seq_indices, input_mask = self.append_mask(masked_item_seq_indices, input_mask)
        
        position_ids = torch.arange(L , dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq_indices)
        position_ebd = self.P(position_ids)
        
        seq_ebd = self.V(item_seq_indices) + position_ebd
        seq_ebd = self.layer_norm(seq_ebd)
        
        seq_rep, i_entropy, m_entropy = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask(item_seq_indices))

        return i_entropy, m_entropy
        
    def loss(self, user_indices, item_seq_indices, pos_item_indices, neg_item_indices):
        
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
        S_u = torch.cat((item_seq_indices,pos_item_indices),1)# S_u
        masked_item_seq_indices, input_mask = self.mask_sequence(S_u, mask_prob)
        
        pos_tgt_item_indices = torch.cat((item_seq_indices, pos_item_indices),1) #(B,L+1)
        all_item_indices = torch.arange(1,self.args.num_items).cuda()
        
        selection = (masked_item_seq_indices == self.mask_index)
        tgt_ebd = self.V(pos_tgt_item_indices)[selection] # (*,D)
        tgt_bias = self.Vb(pos_tgt_item_indices)[selection]#(*,1)
        all_ebd = self.V(all_item_indices) #(N,D)
        all_bias = self.Vb(all_item_indices)#(N,1)
        
        seq_rep = self.get_contextualized_rep(user_indices, masked_item_seq_indices)
        seq_rep = seq_rep[selection] #[*,D]

        #seq_rep : [*, D]
        #tgt_ebd : [*, D]
        #all_ebd : [N, D]
        #tgt_bias: [*,1]
        #all_bias: [N,1]
        
        numerator = (seq_rep * tgt_ebd).sum(-1,keepdims=True) + tgt_bias #(*,1) 
        denominator = (((seq_rep.unsqueeze(1)  * all_ebd.unsqueeze(0)).sum(-1) + all_bias.t()).exp().sum(-1, keepdims=True)).log() # (*,N)
        
        loss = -((numerator - denominator))
        
        return loss.mean()