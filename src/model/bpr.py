import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .layers import TransformerEncoder as TE
from utils.sample_utils import NEGSampler

random_seed=0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic=True
np.random.seed(random_seed)

class BPR(nn.Module):
    
    def __init__(self, args):
        super(BPR, self).__init__()
        
        #
        # Base Model : SASREC
        # Loss : NCE + NCE
        
        self.args = args
        self.mtype = 'SR'
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        self.V = nn.Embedding(self.args.num_items, self.args.dims, padding_idx = 0)
        self.V.weight.data.normal_(0., 1./self.V.embedding_dim)
        
        self.U = nn.Embedding(self.args.num_seqs, self.args.dims, padding_idx = 0)
        self.U.weight.data.normal_(0., 1./self.U.embedding_dim)
        
        self.item2dom = torch.LongTensor(args.item2dom).cuda()
        self.num_dom_items = torch.Tensor(args.num_dom_items).cuda()
        
        self.pad = torch.LongTensor([0]).cuda()
        self.sigmoid = torch.nn.Sigmoid()
        self.neg_sampler = NEGSampler(args)
        self.criterion = nn.BCELoss(reduction='none')
        self.ones = torch.ones((self.args.num_seqs+self.args.num_items, 1), dtype=torch.float32).cuda()
        self.ui_mat_tensor = args.ui_mat_tensor
        
    def graph_convolution(self,user_indices, item_indices, target_domain):

        ebd = torch.cat((self.U.weight,self.V.weight),0)
        g_ebd = ebd
        '''
        mat = self.ui_mat_tensor[target_domain]
        row_sums = mat.matmul(self.ones)
        for l in range(self.args.graph_layers):
            temp_ebd = (1./(row_sums + 0.000001))*torch.sparse.mm(mat, g_ebd)
            g_ebd = g_ebd+temp_ebd
        g_ebd = g_ebd/(1.0+self.args.graph_layers)
        '''
        return g_ebd[user_indices], g_ebd[self.args.num_seqs+item_indices]

    def get_aggregated_representation(self, user_indices, item_indices, target_domain):
        
        user_ebd, item_ebd = self.graph_convolution(user_indices, item_indices, target_domain) # B,L+1,D
        return user_ebd, item_ebd
    
    def forward(self, user_indices, item_seq_indices, target_item_indices, target_domain, pred_opt = ''):
        
        #evaluation
        B,L = item_seq_indices.size() # N,1
        user_ebd, tgt_ebd = self.get_aggregated_representation(user_indices, target_item_indices, 0)

        score = user_ebd.mm(tgt_ebd.squeeze(1).t())
        
        return score
    
    def masked_prediction(self, user_indices, item_seq_indices, target_item_indices, loss_mask):
        
        '''
         predict the score at mask [*, N]
         
        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (* X N) target_item_indices:loss mask로 마스크된 위치의 target들
        :param torch.BoolTensor (B x L) loss_mask: True if the element is masked
        '''
        
        B,L = item_seq_indices.size()
        user_ebd, tgt_ebd = self.get_aggregated_representation(user_indices, target_item_indices, target_domain = 0)
        user_ebd = user_ebd.unsqueeze(1).expand(-1,item_seq_indices.size(1),-1)
        
        user_ebd = user_ebd[loss_mask].unsqueeze(1)
        
        #[*,1,D]~[*,N,D]
        score = user_ebd.bmm(tgt_ebd.permute([0,2,1])).squeeze(1) #[*,N]

        return score
        
    def mask_sequence(self, seq, mask_prob):
        
        # false is the mask
        B,L = seq.size()
        mask = torch.rand((B,L), device = self.device ).ge(mask_prob).cuda()
        mask = (seq == 0) | mask # pad에는 mask를 씌우지 않음.
        masked_seq = mask*seq + (mask==0)*self.pad
        return (masked_seq, mask)
    
    def append_mask(self, seq, mask):
        

        B,L = seq.size()
        masked_seq = torch.cat((seq, self.pad.repeat(B,1)),1)
        mask = torch.cat(( mask, torch.zeros(B,1, dtype=torch.bool).cuda()),1)#[B,L+1]
        
        return (masked_seq, mask)


    def loss_bce(self, user_indices, sorted_item_seq_indices, item_seq_indices, pos_item_indices, neg_item_indices, mask_prob ):
        
        '''
        compute bce loss
        
        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (B x 1) pos_item_indices:
        :param torch.LongTensor (B x 1 x N) neg_item_indices:
        '''
        
        B,L = item_seq_indices.size()
        
        masked_item_seq_indices, input_mask = self.mask_sequence(item_seq_indices, mask_prob)#[B,L]
        masked_item_seq_indices, input_mask = self.append_mask(masked_item_seq_indices, input_mask)#[B,L+1] #input_mask가 false면 mask된 위치.

        masked_item_seq_indices = torch.cat((item_seq_indices, pos_item_indices),1)

        pos_tgt_item_indices = torch.cat((item_seq_indices, pos_item_indices),1).unsqueeze(2)#(B,L+1,1)
        pos_tgt_item_indices = pos_tgt_item_indices[(~input_mask)] # [ # of mask, 1]

        neg_item_indices = neg_item_indices.unsqueeze(1)#B,1,N
        neg_tgt_item_indices = neg_item_indices.expand(-1,L+1,-1)#(B,L+1,N)
        neg_tgt_item_indices = neg_tgt_item_indices[(~input_mask)] # [ # of mask, N]
        
        pos_score = self.masked_prediction(user_indices, masked_item_seq_indices,
                                           pos_tgt_item_indices, ~input_mask) #[ # of mask, 1]
        
        neg_score = self.masked_prediction(user_indices, masked_item_seq_indices,
                                           neg_tgt_item_indices, ~input_mask) #[ # oF mask, N]
        
        
        pos_target = torch.ones_like(pos_score).cuda()
        neg_target = torch.zeros_like(neg_score).cuda()
        ploss = self.criterion(self.sigmoid(pos_score),pos_target)
        nloss = self.criterion(self.sigmoid(neg_score),neg_target)
        loss = torch.cat((ploss,nloss),1) # [# of mask , 1 + N ]
        
        return loss.mean()

    def loss_bpr(self, user_indices, sorted_item_seq_indices, item_seq_indices, pos_item_indices, neg_item_indices, mask_prob ):
        
        '''
        compute bce loss
        
        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (B x 1) pos_item_indices:
        :param torch.LongTensor (B x 1 x N) neg_item_indices:
        
        '''
        B,L = item_seq_indices.size() 

        
        masked_item_seq_indices, input_mask = self.mask_sequence(item_seq_indices, mask_prob)#[B,L]
        masked_item_seq_indices, input_mask = self.append_mask(masked_item_seq_indices, input_mask)#[B,L+1] #input_mask가 false면 mask된 위치.

        masked_item_seq_indices = torch.cat((item_seq_indices, pos_item_indices),1)

        pos_tgt_item_indices = torch.cat((item_seq_indices, pos_item_indices),1).unsqueeze(2)#(B,L+1,1)
        pos_tgt_item_indices = pos_tgt_item_indices[(~input_mask)] # [ # of mask, 1]

        neg_item_indices = neg_item_indices.unsqueeze(1)#B,1,N
        neg_tgt_item_indices = neg_item_indices.expand(-1,L+1,-1)#(B,L+1,N)
        neg_tgt_item_indices = neg_tgt_item_indices[(~input_mask)] # [ # of mask, N]
        
        pos_score = self.masked_prediction(user_indices, masked_item_seq_indices,
                                           pos_tgt_item_indices, ~input_mask) #[ # of mask, 1]
        
        neg_score = self.masked_prediction(user_indices, masked_item_seq_indices,
                                           neg_tgt_item_indices, ~input_mask) #[ # oF mask, N]
        
        m = nn.LogSigmoid()
        loss = -m(pos_score-neg_score)
        
        return loss.mean()
    
    def loss(self, user_indices, sorted_item_seq_indices, item_seq_indices,  pos_item_indices, neg_item_indices, mask_prob , loss_type = 'nce'):
        
        '''
        compute loss

        :param torch.LongTensor (B) user_indices:
        :param torch.LongTensor (B x L) item_seq_indices:
        :param torch.LongTensor (B x 1) pos_item_indices:
        :param torch.LongTensor (B x N) neg_item_indices:
        
        :return: loss
        '''
        
        return self.loss_bpr(user_indices, sorted_item_seq_indices, item_seq_indices, pos_item_indices, neg_item_indices, mask_prob)
    