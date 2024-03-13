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
import sys
from utils.loader_utils import SEQDataset
from utils.eval_utils import evaluate_topk
from utils.eval_utils import evaluate_domain_topk
from utils.sample_utils import NEGSampler
import time
from time import sleep
from collections import defaultdict
parser = argparse.ArgumentParser()
#Env setup
parser.add_argument("--gpu",type=str,default='0',help="gpu number")
#Data setup
parser.add_argument("--data_path",type=str,default="/data1/jyhwang/SCDR/", help="data_path")
parser.add_argument("--dataset",type=str,default="105multi101",help="dataset")
parser.add_argument("--save",choices=[True, False],default=True)
#Experiment setup
parser.add_argument("--max_epoch",type=int,default=150,help="training epoch")
parser.add_argument("--lr",type=float,default=0.0005,help="learning_rate")
parser.add_argument("--decay",type=float,default=0.0,help="weight decay")
parser.add_argument("--batch_size",type=int,default=256,help="batch size")
parser.add_argument("--loss_type",choices=['bce','nce'],default='nce')
parser.add_argument("--negs",type=int,default=200,help="# of neg samples")
parser.add_argument("--mode",choices=['develop','tune'],default='develop')
parser.add_argument("--seed",type=int,default=0,help="seed")

#Model setup
parser.add_argument("--model",choices=['sasrec','hgn','bert4rec','fmlp'],default='bert4rec')
parser.add_argument("--dropout",type=float,default=0.1,help="dropout")
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
args.num_domains = len(dataset.num_dom_items)
args.num_seqs = dataset.num_seqs
args.num_items = dataset.num_items
args.num_dom_items = dataset.num_dom_items
args.ii_mat_tensor = dataset.ii_mat_tensor
args.item2dom = dataset.item2dom
item2dom = torch.LongTensor(dataset.item2dom).cuda()

if args.model == 'sasrec':
    model = SASREC(args).cuda()
elif args.model == 'bert4rec':
    model = BERT4REC(args).cuda()
elif args.model == 'fmlp':
    model = FMLP(args).cuda()
elif args.model == 'hgn':
    model = HGN(args).cuda()

train_loader = data.DataLoader(dataset, batch_size = args.batch_size, shuffle=True)
optimizer= torch.optim.Adam([v for v in model.parameters()], lr=args.lr, weight_decay = args.decay)

best_valid_ndcg = defaultdict(int)
best_valid_mrr = defaultdict(int)
best_valid_recall = defaultdict(int)

best_test_ndcg=defaultdict(int)
best_test_mrr =defaultdict(int)
best_test_recall =defaultdict(int)

model_state = defaultdict(int)
best_epoch = defaultdict(int)
neg_sampler = NEGSampler(args)

for epoch in range(1,args.max_epoch+1):

    model.train()

    P = [1,args.window_length+1,args.window_length+2]
    for it, batch_instance in enumerate(tqdm(train_loader, desc="Training", position=0, disable = (args.mode =='tune') )):
        users = batch_instance[:,:P[0]].squeeze().cuda()
        sequence = batch_instance[:,P[0]:P[1]].cuda()
        positive = batch_instance[:,P[1]:P[2]].cuda()
        sorted_sequence = batch_instance[:,P[2]:].cuda()
        if args.model == 'sasrec'or args.model=='hgn' or args.model == 'fmlp' :
            negative = neg_sampler.sample_negative_items_online(sorted_sequence, args.window_length) # B,L
        else:
            negative = neg_sampler.sample_negative_items_online(sorted_sequence, args.negs)
        batch_loss=[]

        task_loss = model.loss(users,
                               sorted_sequence,
                               sequence,
                               positive, #B,1
                               negative) #B,N
                               
        batch_loss = task_loss.mean()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()#B,L,L

    model.eval()
    if epoch %2==0 or True:
        # Validation
        
        '''
        for target_domain in range(1,args.num_domains):
            
            valid_seqs = torch.arange(dataset.num_seqs)
            valid_domains = dataset.valid_domains
            valid_seqs = valid_seqs[(valid_domains == target_domain ).flatten()]           
            result = evaluate_domain_topk(model, valid_seqs, dataset, target_domain, 'valid', 'eval', 10)

            valid_seqs = torch.arange(dataset.num_seqs)
            valid_domains = dataset.valid_domains
            valid_type = dataset.cd_type
            
            if args.mode == 'develop':
                print("Validation")
                print("Domain:: %s"%(target_domain))
                print("[%s/%s][recall]@10:: %.4lf"%(epoch,args.max_epoch,result["recall"]))
                print("[%s/%s][ndcg  ]@10:: %.4lf"%(epoch,args.max_epoch,result["ndcg"]))
                print("[%s/%s][mrr   ]@10:: %.4lf"%(epoch,args.max_epoch,result["mrr"]))

            if result["ndcg"] >= best_valid_ndcg[target_domain]:

        if args.mode == 'develop': #and epoch % 5 ==0:
        '''
if args.mode == 'tune':
    print(args)
    objs = {"hyperparams": args}
    # 하이퍼파라미터 저장
    # 모델저장
    # Valid Printing
    # Test Printing