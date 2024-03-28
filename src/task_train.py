import os
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data
import argparse
import copy
from argparse import Namespace
from tqdm import tqdm
from model.sasrec import SASREC
from model.bert4rec import BERT4REC
from model.hgn import HGN
from model.fmlp import FMLP
from model.nip import NIP
######################################
from model.mip import MIP
from model.nmip import NMIP
from model.pmip import PMIP
from model.amip import AMIP
######################################
from model.asmip import ASMIP

import sys
from utils.loader_utils import SEQDataset
from utils.eval_utils import evaluate_topk
from utils.sample_utils import NEGSampler
import time
from time import sleep
from collections import defaultdict
parser = argparse.ArgumentParser()
#Env setup
parser.add_argument("--gpu",type=str,default='0',help="gpu number")
#Data setup
parser.add_argument("--data_path",type=str,default="/data1/jyhwang/SR/", help="data_path")
parser.add_argument("--dataset",type=str,default="Books",help="dataset")
parser.add_argument("--save",choices=[True, False],default=True)
#Experiment setup
parser.add_argument("--max_epoch",type=int,default=200,help="training epoch")
parser.add_argument("--lr",type=float,default=0.0005,help="learning_rate")
parser.add_argument("--decay",type=float,default=0.0,help="weight decay")
parser.add_argument("--batch_size",type=int,default=256,help="batch size")
parser.add_argument("--loss_type",choices=['bce','nce'],default='nce')
parser.add_argument("--negs",type=int,default=200,help="# of neg samples")
parser.add_argument("--mode",choices=['develop','tune'],default='develop')
parser.add_argument("--seed",type=int,default=0,help="seed")

#Model setup
parser.add_argument("--model",choices=['nip','nmip','pmip','amip','asmip'],default='asmip')
parser.add_argument("--shots",type = int, default = 1, help = "shots")
parser.add_argument("--alpha",type = float, default = 0.1, help = "")
parser.add_argument("--beta",type = float, default = 0.1, help = "")
parser.add_argument("--dropout",type = float, default = 0.1,help="dropout")
parser.add_argument("--dims",type=int,default=128,help="embedding size")
parser.add_argument("--encoder_layers", type=int, default=2, help="# of encoder layers")
parser.add_argument("--graph_layers", type=int, default=1, help="# of graph layers")
parser.add_argument("--window_length",type=int,default=50,help="window length")
parser.add_argument("--target_length",type=int,default=1,help="target length")
parser.add_argument("--mask_prob",type=float,default=0.4,help="masking probability")

args = parser.parse_args()
args.model_img_path = "./knowledge/"+args.dataset+"/"

random_seed=args.seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic=True
np.random.seed(random_seed)

#Data loading
dataset = SEQDataset(args)
args.num_seqs = args.num_users = dataset.num_users
args.num_items = dataset.num_items

if args.model =='nip':
    model = NIP(args).cuda()
elif args.model == 'mip':
    model = MIP(args).cuda()
elif args.model == 'nmip':
    model = NMIP(args).cuda()
elif args.model == 'pmip':
    model = PMIP(args).cuda()
elif args.model == 'amip':
    model = AMIP(args).cuda()
elif args.model == 'asmip':
    model = ASMIP(args).cuda()

train_loader = data.DataLoader(dataset, batch_size = args.batch_size, shuffle=True)
optimizer= torch.optim.Adam([v for v in model.parameters()], lr=args.lr, weight_decay = args.decay)

best_valid = defaultdict(int)
best= defaultdict(int)

model_state = defaultdict(int)
best_epoch = defaultdict(int)
neg_sampler = NEGSampler(args)
stop_cnt = 0

entropy_result=[]
pop_result=[]
topk_pop_result=[]
hits_list = []

for epoch in range(1,args.max_epoch+1):

    model.train()

    P = [1,args.window_length+1,args.window_length+2]
    for it, batch_instance in enumerate(tqdm(train_loader, desc="Training", position=0, disable = (args.mode =='tune') )):
        
        users = batch_instance[:,:P[0]].squeeze().cuda()
        sequence = batch_instance[:,P[0]:P[1]].cuda()
        positive = batch_instance[:,P[1]:P[2]].cuda()
        sorted_sequence = batch_instance[:,P[2]:].cuda()

        negative = neg_sampler.sample_negative_items_online(sorted_sequence, args.negs)
        batch_loss=[]
        
        if args.model == 'nip':
            nip_loss = model.loss(users,
                                  sequence,
                                  positive,
                                  negative)
            batch_loss = nip_loss.mean()
        elif args.model == 'mip':
            mip_loss = model.loss(users,
                                  sequence,
                                  positive,
                                  negative)
            batch_loss = mip_loss.mean()
        elif args.model == 'nmip':
            nmip_loss = model.loss(users,
                                   sequence,
                                   positive,
                                   negative)
            batch_loss = nmip_loss.mean()
        elif args.model == 'pmip':
            pmip_loss = model.loss(users,
                                   sequence,
                                   positive,
                                   negative)
            batch_loss = pmip_loss.mean()
        elif args.model == 'amip':
            amip_loss = model.loss(users,
                                   sequence,
                                   positive,
                                   negative)
            batch_loss = amip_loss.mean()
        elif args.model =='asmip':
            amip_loss, mip_loss = model.loss(users,
                                            sequence,
                                            positive, #B,1
                                            negative) #B,N

            batch_loss = amip_loss.mean() + args.alpha*mip_loss.mean()
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()#B,L,L

    model.eval()
    if epoch % 2 == 1:
        
        validation_mask = (torch.LongTensor(dataset.valid_last_subseqs).sum(1) > 0)
        users = torch.arange(dataset.num_seqs)
        result20 = evaluate_topk(model, users[validation_mask], dataset, 'valid', 20)
        entropy_result.append(float(result20['entropy']))
        pop_result.append(float(result20['hits_pop']))
        topk_pop_result.append(float(result20['topk_pop']))
        
        if args.mode == 'develop':
            print("Validation[%s/%s]"%(epoch,args.max_epoch))
            print("[RECALL ]@20 :: %.4lf"%(result20['recall']))
            print("[NDCG   ]@20 :: %.4lf"%(result20['ndcg']))
            print("[MRR    ]@20 :: %.4lf"%(result20['mrr']))
            print("[Entropy]    :: %.4lf"%(result20['entropy']))
            print("[Avg.HitsPop]:: %.4lf"%(result20['hits_pop']))
            print("[Avg.topkPop]:: %.4lf"%(result20['topk_pop']))
            
            
        if best_valid['recall@20'] <= result20['recall']:
            best_valid['recall@20'] = result20['recall']
            best_valid['ndcg@20'] = result20['ndcg']
            best_valid['mrr@20'] = result20['mrr']
            
            hits_list = result20['hits_list']
            
            '''
            if args.mode == 'tune':
                result10 = evaluate_topk(model, users[validation_mask], dataset, 'valid', 20)
                result50 = evaluate_topk(model, users[validation_mask], dataset, 'valid', 20)
                result100 = evaluate_topk(model, users[validation_mask], dataset, 'valid', 20)
                best_valid['recall@10'] = result10['recall']
                best_valid['recall@20'] = result20['recall']
                best_valid['recall@50'] = result50['recall']
                best_valid['recall@100'] = result100['recall']
                
                best_valid['ndcg@10'] = result10['ndcg']
                best_valid['ndcg@20'] = result20['ndcg']
                best_valid['ndcg@50'] = result50['ndcg']
                best_valid['ndcg@100'] = result100['ndcg']
            '''    
            stop_cnt=0
        else:
            stop_cnt= stop_cnt + 1
    
    if stop_cnt >=20:break

'''
print("Entropy:")
print(entropy_result)
print("Hits Popularity:")
print(pop_result)
print("topk Popularity:")
print(topk_pop_result)

#print("Hits_List:")
#print(list(hits_list.flatten()))
print("Validation Performance")
print("[RECALL ]@20:: %.4lf"%(best_valid["recall@20"]))
print("[NDCG   ]@20:: %.4lf"%(best_valid["ndcg@20"]))
'''

if args.mode == 'tune':
    print(args)

    objs = {"hyperparams": args, "entropy": entropy_result}
    
    with open('./knowledge/'+'%s'%(args.dataset)+'/hyperparams/'+'%s_%s-%s_%.4lf'%(args.model, args.window_length, args.dims, float(best["ndcg@20"]))+'.pkl','wb') as f:
        pickle.dump(objs, f)
    
    print("Validation Performance")
    
    print("[RECALL ]@20:: %.4lf"%(best_valid["recall@20"]))
    print("[NDCG   ]@20:: %.4lf"%(best_valid["ndcg@20"]))
    print("[MRR    ]@50:: %.4lf"%(best_valid["ndcg@20"]))
    