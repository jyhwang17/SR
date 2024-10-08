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
from model.cl4srec import CL4SREC
from model.fmlp import FMLP

from model.mip import MIP
from model.nip import NIP

from model.proposed import PROPOSED
from model.proposed2 import PROPOSED2
from model.catproposed import CATPROPOSED
from model.mulproposed import MULPROPOSED
from model.addproposed import ADDPROPOSED

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
parser.add_argument("--dataset",type=str,default="CDs",help="dataset")
parser.add_argument("--save",choices=[True, False],default=True)

#Experiment setup
parser.add_argument("--max_epoch",type=int,default=500,help="training epoch")
parser.add_argument("--lr",type=float,default=0.0005,help="learning_rate")
parser.add_argument("--decay",type=float,default=0.0,help="weight decay")
parser.add_argument("--batch_size",type=int,default=256,help="batch size")
parser.add_argument("--loss_type",choices=['bce','nce'],default='nce')
parser.add_argument("--negs",type=int,default=200,help="# of neg samples")
parser.add_argument("--mode",choices=['develop','tune'],default='develop')
parser.add_argument("--seed",type=int,default=0,help="seed")

#Model setup
parser.add_argument("--model",choices=['sasrec','hgn','bert4rec','fmlp',
                                       'ct4rec','cbit','ct4rec',
                                       'cl4srec','proposed','proposed2',
                                       'catproposed','mulproposed','addproposed',
                                       ],default='bert4rec')

parser.add_argument("--shots",type=int, default=1)
parser.add_argument("--alpha",type=float,default=0.2, help= " loss weight")
parser.add_argument("--beta", type=float,default=0.5, help= "loss weight")
parser.add_argument("--gamma",type=float,default=1.0, help= " loss weight")
parser.add_argument("--lambda",type=float,default=1.0, help= " loss weight")
parser.add_argument("--num_experts",type=int,default=8, help = "num_experts")

parser.add_argument("--dropout",type=float,default=0.3,help="dropout")
parser.add_argument("--dims",type=int,default=128,help="embedding size")
parser.add_argument("--heads",type=int, default = 2, help ="# of heads")
parser.add_argument("--encoder_layers", type=int, default = 2, help="# of encoder layers")

parser.add_argument("--window_length",type=int,default=50,help="window length")
parser.add_argument("--target_length",type=int,default=1,help="target length")
parser.add_argument("--mask_prob",type=float,default=0.3,help="masking probability")
parser.add_argument("--aug_cnt",type=int,default=1,help="query augmentation count")
parser.add_argument("--augmentation_rule",type=float,default=1,help="augmentation rule")

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

if args.model == 'sasrec':
    model = SASREC(args).cuda()
elif args.model == 'bert4rec':
    model = BERT4REC(args).cuda()
elif args.model == 'hgn':
    model = HGN(args).cuda()
elif args.model == 'cl4srec':
    model = CL4SREC(args).cuda()
elif args.model == 'fmlp':
    model = FMLP(args).cuda()
elif args.model == 'proposed':
    model = PROPOSED(args).cuda()
elif args.model == 'proposed2':
    model = PROPOSED2(args).cuda()
elif args.model == 'catproposed':
    model = CATPROPOSED(args).cuda()
elif args.model == 'mulproposed':
    model = MULPROPOSED(args).cuda()
elif args.model == 'addproposed':
    model = ADDPROPOSED(args).cuda()
    
train_loader = data.DataLoader(dataset, batch_size = args.batch_size, shuffle=True)
optimizer= torch.optim.Adam([v for v in model.parameters()], lr=args.lr, weight_decay = args.decay)
best_valid = {'ndcg':0.0, 'recall':0.0, 'mrr':0.0 }
best= defaultdict(int)

model_state = defaultdict(int)
best_epoch = defaultdict(int)
neg_sampler = NEGSampler(args)
stop_cnt = 0

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
        consistency = []
        
        if args.model == 'proposed':
            amip_loss1, amip_loss2, amip_loss3, amip_loss4 =  model.loss(users, sequence, positive, negative)
            batch_loss = amip_loss1 + args.alpha*amip_loss2 + args.beta*amip_loss3 + args.gamma*amip_loss4
        elif args.model == 'proposed2':
            
            amip_loss1, amip_loss2, amip_loss3, amip_loss4, mip_loss =  model.loss(users, sequence, positive, negative)
            batch_loss = amip_loss1 + args.alpha*amip_loss3 + args.beta*amip_loss2 + args.gamma*mip_loss
            
        elif args.model == 'catproposed' or args.model == 'mulproposed' or args.model == 'addproposed':
            
            #model.gate.update_T( max(0.2, 3.0*(0.9**(epoch)) ) )
            amip_loss1, amip_loss2, amip_loss3, amip_loss4 = model.loss(users, sequence, positive, negative)
            
            batch_loss = amip_loss1 + args.alpha*amip_loss3 + args.beta*amip_loss2
            
        elif args.model == 'sasrec' or args.model =='nip':
            basic_loss = model.loss(users,sequence,positive,negative) #B,N
            batch_loss = basic_loss
            
        elif args.model == 'bert4rec' or args.model == 'mip':
            basic_loss = model.loss(users,sequence,positive,negative)
            batch_loss = basic_loss
            
        elif args.model == 'hgn':
            batch_loss = model.loss(users, sequence, positive, negative)
            
        elif args.model == 'cl4srec':
            basic_loss, cl_loss = model.loss(users, sequence, positive, negative)
            batch_loss = basic_loss + args.alpha*cl_loss
            
        elif args.model == 'fmlp':
            basic_loss = model.loss(users, sequence, positive, negative)
            batch_loss = basic_loss

        optimizer.zero_grad()
        batch_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()#B,L,L
    
    model.eval()

    if epoch %2==0:
        validation_mask = (torch.LongTensor(dataset.valid_last_subseqs).sum(1) > 0)
        users = torch.arange(dataset.num_seqs)
        result20 = evaluate_topk(model, users[validation_mask], dataset, 'valid', 20)
        
        if args.mode == 'develop':
            
            print("Validation[%s/%s]"%(epoch,args.max_epoch))
            print("[RECALL ]@20:: %.4lf"%(result20['recall']))
            print("[NDCG   ]@20:: %.4lf"%(result20['ndcg']))
            print("[MRR    ]@20:: %.4lf"%(result20['mrr']))
            
        if best_valid['ndcg'] <= result20['ndcg']:
            best_valid = result20
            model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            
            #Testing
            test_mask = (torch.LongTensor(dataset.test_last_subseqs).sum(1) > 0)
            users = torch.arange(dataset.num_seqs)
        
            result10 = evaluate_topk(model, users[test_mask], dataset, 'test', 10)
            result20 = evaluate_topk(model, users[test_mask], dataset, 'test', 20)
            result50 = evaluate_topk(model, users[test_mask], dataset, 'test', 50)
            
            best["recall@10"] = result10['recall']
            best["recall@20"] = result20['recall']
            best["recall@50"] = result50['recall']
            
            best["ndcg@10"] = result10['ndcg']
            best["ndcg@20"] = result20['ndcg']
            best["ndcg@50"] = result50['ndcg']
            
            best["mrr@10"] = result10['mrr']
            best["mrr@20"] = result20['mrr']
            best["mrr@50"] = result50['mrr']
            
            stop_cnt=0
        else:
            stop_cnt= stop_cnt + 1
            
    if stop_cnt >=50: break

if args.mode == 'develop':
    print("-------------------------")
    print("[Best VALID EPOCH]:: %s"%(int(best_epoch)))
    print("[RECALL ]@20:: %.4lf"%(float(best_valid["recall"])))
    print("[NDCG   ]@20:: %.4lf"%(float(best_valid["ndcg"])))
    print("[MRR    ]@20:: %.4lf"%(float(best_valid["mrr"])))
    
    print("-------------------------")
    print("[RECALL ]@10:: %.4lf"%(best["recall@10"]))
    print("[RECALL ]@20:: %.4lf"%(best["recall@20"]))
    print("[RECALL ]@50:: %.4lf"%(best["recall@50"]))
        
    print("[NDCG   ]@10:: %.4lf"%(best["ndcg@10"]))
    print("[NDCG   ]@20:: %.4lf"%(best["ndcg@20"]))
    print("[NDCG   ]@50:: %.4lf"%(best["ndcg@50"]))
        
    print("[MRR    ]@10:: %.4lf"%(best["mrr@10"]))
    print("[MRR    ]@20:: %.4lf"%(best["mrr@20"]))
    print("[MRR    ]@50:: %.4lf"%(best["mrr@50"]))

if args.mode == 'tune':
    objs = {"hyperparams": args, "valid_results": best_valid }
    with open('./knowledge/%s/hyperparams/%s_%s-%s_%.4lf_%.4lf'%(args.dataset,
                                                                 args.model,args.window_length,args.dims,
                                                                 float(best_valid["ndcg"]), float(best["ndcg@20"]))+'.pkl','wb') as f:
        pickle.dump(objs, f)
    
        with open('./log/%s/%s_%s_%s.txt'%(args.dataset,args.model,args.window_length,args.dims) ,'a') as f:
            f.write(str(args))
            f.write("\n")
            f.write("[RECALL ]@10:: %.4lf\n"%(best["recall@10"]))
            f.write("[RECALL ]@20:: %.4lf\n"%(best["recall@20"]))
            f.write("[RECALL ]@50:: %.4lf\n"%(best["recall@50"]))

            f.write("[NDCG   ]@10:: %.4lf\n"%(best["ndcg@10"]))
            f.write("[NDCG   ]@20:: %.4lf\n"%(best["ndcg@20"]))
            f.write("[NDCG   ]@50:: %.4lf\n"%(best["ndcg@50"]))

            f.write("[MRR    ]@10:: %.4lf\n"%(best["mrr@10"]))
            f.write("[MRR    ]@20:: %.4lf\n"%(best["mrr@20"]))
            f.write("[MRR    ]@50:: %.4lf\n"%(best["mrr@50"]))
            f.close()
    
    torch.save(model_state, './knowledge/%s/%s_%s-%s_%.4lf_%.4lf'%(args.dataset,
                                                                   args.model, args.window_length, args.dims,
                                                                   float(best_valid["ndcg"]), float(best["ndcg@20"])))