import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .layers import TransformerEncoder as TE
import math
import random
class CL4SREC(nn.Module):
    
    def __init__(self, args):
        super(CL4SREC, self).__init__()
        
        #CL4SREC
        self.args = args
        self.mtype = 'JOIN'
        
        self.V = nn.Embedding(self.args.num_items+1, self.args.dims, padding_idx = 0)
        self.V.weight.data.normal_(0., 1./self.V.embedding_dim)
        
        
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
                
        self.augmentation_rule = self.args.augmentation_rule
        
        self.criterion = nn.BCELoss(reduction='none')
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.args.dropout)
        self.layer_norm = nn.LayerNorm(self.args.dims)
         
        self.apply(self._init_weights)
        self.V.weight.data[0] = 0.
        
        self.mask_index = torch.cuda.LongTensor([self.args.num_items])
        
    def _init_weights(self, module):
        
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):

            module.weight.data.normal_(mean=0.0, std=1./self.args.dims )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def get_attention_mask(self, item_seq, bidirectional = False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def get_contextualized_rep(self, user_indices, item_seq_indices):
        
        
        B,L = item_seq_indices.size()
        device_ = item_seq_indices.device
        position_ids = torch.arange(L , dtype=torch.long, device = device_)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq_indices)
        position_ebd = self.P(position_ids)
        
        seq_ebd = self.V(item_seq_indices) + position_ebd
        seq_ebd = self.layer_norm(seq_ebd)
        seq_ebd = self.dropout_layer(seq_ebd)
        seq_rep = self.seq_transformer_encoder(seq_ebd, self.get_attention_mask(item_seq_indices))[-1]      
        
        return seq_rep
    
    def sampled_ce_loss(self, scores):
        #scores = B,L+1,1
        
        numerator =  scores[:,0:1] # (*, 1)
        denominator = (scores[:,1:].exp().sum(-1,keepdims=True) + scores[:,0:1].exp() ).log() #[*, 1]
        loss = -((numerator - denominator))
        
        return loss
    
    def item_crop(self, item_seq, item_seq_len, eta = 0.5):
        
        num_left = math.floor(item_seq_len * eta)
        crop_begin = random.randint(0, item_seq_len - num_left)
        croped_item_seq = np.zeros(item_seq.shape[0])
        
        if crop_begin + num_left < item_seq.shape[0]:
            croped_item_seq[item_seq.shape[0]-num_left:] = item_seq.cpu().detach().numpy()[crop_begin:crop_begin + num_left]
        else:
            croped_item_seq[item_seq.shape[0]-num_left:] = item_seq.cpu().detach().numpy()[crop_begin:]
        return torch.tensor(croped_item_seq, dtype=torch.long, device=item_seq.device),torch.tensor(num_left, dtype=torch.long, device=item_seq.device)

    def item_mask(self, item_seq, item_seq_len, gamma=0.5):
        num_mask = math.floor(item_seq_len * gamma)
        mask_index = random.sample(range(item_seq_len), k=num_mask)
        masked_item_seq = item_seq.cpu().detach().numpy().copy()
        masked_item_seq[mask_index] = self.args.num_items  # token 0 has been used for semantic masking
        return torch.tensor(masked_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len

    def item_reorder(self, item_seq, item_seq_len, beta = 0.5):
        
        num_reorder = math.floor(item_seq_len * beta)
        reorder_begin = random.randint(0, item_seq_len - num_reorder)
        reordered_item_seq = item_seq.cpu().detach().numpy().copy()
        shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
        random.shuffle(shuffle_index)
        reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
        return torch.tensor(reordered_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len

    def augment(self, item_seq, item_seq_len):
    
        switch = torch.randint(0, 3, (item_seq.size(0), 2))
        
        if self.augmentation_rule == 0:
            candidates = torch.tensor([0])
        elif self.augmentation_rule == 1:
            candidates = torch.LongTensor([1])
        elif self.augmentation_rule == 2:
            candidates = torch.LongTensor([2])
        elif self.augmentation_rule == 3:
            candidates = torch.LongTensor([0,1])
        elif self.augmentation_rule == 4:
            candidates = torch.LongTensor([0,2])
        elif self.augmentation_rule == 5:
            candidates = torch.LongTensor([1,2])
        elif self.augmentation_rule == 6:
            candidates = torch.LongTensor([0,1,2])
        
        # Define the batch size
        batch_size = item_seq.size(0)
        # Generate random indices from 0 to the number of candidates (exclusive)
        random_indices = torch.randint(0, len(candidates), (batch_size, 2))
        # Map indices to the actual candidates
        switch = candidates[random_indices]
        
        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []

        for idx, (seq, length) in enumerate(zip(item_seq, item_seq_len)):

            aug_idx_1 = int(switch[idx,0])
            aug_idx_2 = int(switch[idx,1])

            if aug_idx_1 == 0:
                aug_seq, aug_len = self.item_crop(seq, length)
            elif aug_idx_1 == 1:
                aug_seq, aug_len = self.item_mask(seq, length)
            elif aug_idx_1 == 2:
                aug_seq, aug_len = self.item_reorder(seq, length)
            
            aug_seq1.append(aug_seq)
            aug_len1.append(aug_len)
            
            if aug_idx_2 == 0:
                aug_seq, aug_len = self.item_crop(seq, length)
            elif aug_idx_2 == 1:
                aug_seq, aug_len = self.item_mask(seq, length)
            elif aug_idx_2 == 2:
                aug_seq, aug_len = self.item_reorder(seq, length)

            aug_seq2.append(aug_seq)
            aug_len2.append(aug_len)
        
        return torch.stack(aug_seq1), torch.stack(aug_seq2) #, switch

    def forward(self, user_indices, item_seq_indices, target_item_indices, pred_opt = ''):
        
        B,L = item_seq_indices.size()

        tgt_ebd = self.V(target_item_indices)#[B,L,N,D] or #[B,1,D]
        seq_rep = self.get_contextualized_rep(user_indices, item_seq_indices)
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
        
        score = self.forward(user_indices, total_sequence[:,:-1], tgt_item_indices, pred_opt='training') #=>[B,L,1+N]
        
        pos_score = score[:,:,[0]]
        neg_score = score[:,:,1:]
        
        numerator =  pos_score # (*, 1)
        denominator = (neg_score.exp().sum(-1,keepdims=True) + pos_score.exp()).log() #[*, 1]
        
        loss = -((numerator - denominator)) # [*,1]


        seq_len = (total_sequence != 0).sum(-1, keepdims=True)
        aug_total_sequence1, aug_total_sequence2 =  self.augment(total_sequence, seq_len)
        
        aug_pad_filter = (aug_total_sequence1.sum(-1) != 0) & (aug_total_sequence2.sum(-1) != 0)
        aug_msk_filter = (aug_total_sequence1[:,-1] != self.args.num_items) & (aug_total_sequence2[:,-1] != self.args.num_items)
        aug_filter = aug_pad_filter & aug_msk_filter

        aug_total_sequence1 = aug_total_sequence1[aug_filter]
        aug_total_sequence2 = aug_total_sequence2[aug_filter]

        aug_seq_rep1 = self.get_contextualized_rep(user_indices, aug_total_sequence1[:,1:])[:,-1,:]
        aug_seq_rep2= self.get_contextualized_rep(user_indices, aug_total_sequence2[:,1:])[:,-1,:]

        logits1 = aug_seq_rep1.mm(aug_seq_rep2.t()) / math.sqrt(self.args.dims) # B x B
        
        numerator = torch.diagonal(logits1, 0)
        denominator = (logits1.exp().sum(dim=1)).log()
        contrastive_loss1 = -(numerator - denominator)
        
        
        logits2= aug_seq_rep2.mm(aug_seq_rep1.t()) / math.sqrt(self.args.dims)
        
        numerator = torch.diagonal(logits2, 0)
        denominator = (logits2.exp().sum(dim=1)).log()
        contrastive_loss2 = -(numerator - denominator)
        
        
        return loss.mean(), contrastive_loss1.mean() + contrastive_loss2.mean()