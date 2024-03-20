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

class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.
    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer
    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer
    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

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
        #self.projection = nn.Linear(self.args.dims, self.args.dims)
        '''
        self.projection = FF(hidden_size = self.args.dims,
                             inner_size= self.args.dims*4,
                             hidden_dropout_prob=self.args.dropout,
                             hidden_act='gelu',
                             layer_norm_eps=1e-24)
        '''
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
        loss = torch.cuda.DoubleTensor()
        
        for i in range(5):
            masked_item_seq_indices, input_mask = self.mask_sequence(item_seq_indices, mask_prob)
            masked_item_seq_indices, input_mask = self.append_mask(masked_item_seq_indices, input_mask)
            
            pos_tgt_item_indices1 = torch.cat((item_seq_indices[:,1:],
                                               pos_item_indices, pos_item_indices),1).unsqueeze(2) #(B,L+1,1) #Next Item
            '''
            pos_tgt_item_indices2 = torch.cat((item_seq_indices[:,2:],
                                               pos_item_indices, pos_item_indices, pos_item_indices),1).unsqueeze(2) #(B,L+1,1)
            '''
            pos_tgt_item_indices2 = torch.cat((pos_item_indices, item_seq_indices[:,:]),1).unsqueeze(2) #(B,L+1,1) #Prev Item
            
            neg_tgt_item_indices = neg_item_indices.unsqueeze(1).expand(-1,item_seq_indices.size(1)+1,-1) # B,L+1,N
            
            tgt_item_indices1 = torch.cat((pos_tgt_item_indices1,neg_tgt_item_indices),-1)
            tgt_item_indices2 = torch.cat((pos_tgt_item_indices2,neg_tgt_item_indices),-1)
            
            false_mask = torch.zeros((input_mask.size(0),1) ,device = input_mask.device).bool()
            loss_mask1 = (torch.cat((~input_mask[:,1:],false_mask),-1))  # next 아이템이 mask인지 아닌지
            
            #loss_mask2 = ~(torch.cat((input_mask[:,2:], false_mask, false_mask),-1))  # next,next 아이템이 mask인지 아닌지
            #input_mask2 = loss_mask1 & loss_mask2 # 연속적으로 mask인지.
            loss_mask2 = (torch.cat((false_mask, ~input_mask[:,:-1]),-1)) # prev, 아이템이 mask인지 아닌지
            
            #Get representation
            transformer_rep = self.get_contextualized_rep(user_indices, masked_item_seq_indices)
            
            # Get score
            rep1  = transformer_rep[loss_mask1].unsqueeze(1) # [*,1,dims]
            tgt_ebd1 = self.V(tgt_item_indices1)[loss_mask1]
            score1 = rep1.bmm(tgt_ebd1.permute([0,2,1])).squeeze(1) #[*,1+N]
            
            rep2  = transformer_rep[loss_mask2].unsqueeze(1) # [*,1,dims]
            #rep2 = self.projection(rep2)
            
            tgt_ebd2 = self.V(tgt_item_indices2)[loss_mask2]
            score2 = rep2.bmm(tgt_ebd2.permute([0,2,1])).squeeze(1) #[*,1+N]
            
            numerator1 =  score1[:,[0]] # (*, 1)
            denominator1 = (score1[:,1:].exp().sum(-1,keepdims=True) + score1[:,[0]].exp() ).log() #[*, 1]
            
            numerator2 =  score2[:,[0]] # (*, 1)
            denominator2 = (score2[:,1:].exp().sum(-1,keepdims=True) + score2[:,[0]].exp() ).log() #[*, 1]

            loss1 = -((numerator1 - denominator1))
            loss2 = -((numerator2 - denominator2))
            
            loss = torch.cat((loss, loss1, loss2),0)
            
        return loss.mean()
    
    def loss2(self, user_indices, item_seq_indices, pos_item_indices, neg_item_indices):
        
        '''
        compute loss

        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (B x 1) pos_item_indices:
        :param torch.LongTensor (B x N) neg_item_indices:
        :return: loss
        '''
        B,L = item_seq_indices.size() 
        
        
        ###################################################################### Forward
        
        
        mask_prob = self.mask_prob
        loss = torch.cuda.DoubleTensor()
        for i in range(5):
            masked_item_seq_indices, input_mask = self.mask_sequence(item_seq_indices, mask_prob)
            masked_item_seq_indices, input_mask = self.append_mask(masked_item_seq_indices, input_mask)
        
            pos_tgt_item_indices = torch.cat((item_seq_indices[:,1:], pos_item_indices, pos_item_indices),1).unsqueeze(2) #(B,L+1,1)
        
            false_mask = torch.zeros((input_mask.size(0),1) ,device= input_mask.device).bool()
            forward_input_mask = torch.cat((input_mask[:,1:],false_mask),-1)  # next 아이템이 mask인지 아닌지

            neg_tgt_item_indices = neg_item_indices.unsqueeze(1).expand(-1,item_seq_indices.size(1)+1,-1) # B,L+1,N
        
            pos_score = self.masked_prediction(user_indices, masked_item_seq_indices,
                                               pos_tgt_item_indices[(~forward_input_mask)], ~forward_input_mask,
                                               pred_opt ='Forward') #[*, 1]
        
            neg_score = self.masked_prediction(user_indices, masked_item_seq_indices,
                                               neg_tgt_item_indices[(~forward_input_mask)], ~forward_input_mask,
                                               pred_opt ='Forward') #[*, N]

            numerator =  pos_score # (*, 1)
            denominator = (neg_score.exp().sum(-1,keepdims=True) + pos_score.exp()).log() #[*, 1]

            nce = -((numerator - denominator))
            loss = torch.cat((loss, nce),0)
        
        #loss = -((numerator - denominator)) # [*,1]
        ##################################################################### Backward
        '''
            pos_tgt_item_indices = torch.cat((pos_item_indices, item_seq_indices[:,:]),1).unsqueeze(2) #(B,L+1,1)
            backward_input_mask = torch.cat((false_mask, input_mask[:,:-1]),-1)  # prev아이템이 mask인지 아닌지
        
            pos_tgt_item_indices = pos_tgt_item_indices[(~backward_input_mask)]  # [*, 1]
        
            pos_score = self.masked_prediction(user_indices,masked_item_seq_indices,
                                               pos_tgt_item_indices, ~backward_input_mask,
                                               pred_opt ='Backward') #[*, 1]
            neg_score = self.masked_prediction(user_indices, masked_item_seq_indices,
                                               neg_tgt_item_indices[(~backward_input_mask)], ~backward_input_mask,
                                               pred_opt ='Backward') #[*, N]
        
            numerator =  pos_score # (*, 1)
            denominator = (neg_score.exp().sum(-1,keepdims=True) + pos_score.exp()).log() #[*, 1]
            bnce = -((numerator - denominator))
            loss = torch.cat((loss, bnce ),0)
        '''
        ###################################################################### Next-Item Prediction
        '''
        mask_prob = self.mask_prob
        masked_item_seq_indices, input_mask = self.mask_sequence(item_seq_indices, 0.0)
        
        masked_item_seq_indices, input_mask = self.append_mask(masked_item_seq_indices, input_mask)
        
        pos_tgt_item_indices = torch.cat((item_seq_indices[:,1:], pos_item_indices, pos_item_indices),1).unsqueeze(2) #(B,L+1,1)
        false_mask = torch.zeros((input_mask.size(0),1) ,device= input_mask.device).bool()
        
        forward_input_mask = torch.cat((input_mask[:,1:],false_mask),-1)  # next 아이템이 mask인지 아닌지
        #pos_tgt_item_indices = pos_tgt_item_indices[(~forward_input_mask)]  # [*, 1]
        
        neg_tgt_item_indices = neg_item_indices.unsqueeze(1).expand(-1,item_seq_indices.size(1)+1,-1) # B,L+1,N
        
        pos_score = self.masked_prediction(user_indices, masked_item_seq_indices,
                                           pos_tgt_item_indices[(~forward_input_mask)], ~forward_input_mask,
                                           pred_opt ='Forward') #[*, 1]
        neg_score = self.masked_prediction(user_indices, masked_item_seq_indices,
                                           neg_tgt_item_indices[(~forward_input_mask)], ~forward_input_mask,
                                           pred_opt ='Forward') #[*, N]

        numerator =  pos_score # (*, 1)
        denominator = (neg_score.exp().sum(-1,keepdims=True) + pos_score.exp()).log() #[*, 1]
        loss = torch.cat((loss, -((numerator - denominator)) ),0)
        '''
        return loss.mean()