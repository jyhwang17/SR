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
parser.add_argument("--model",choices=['c2dsr','sasrec','sasrecs','hgn','dream','bert4rec','fmlp','unicdrseq','bert4recm2','sasrecm','sasrecm2'],default='bert4rec')
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
        if args.model == 'sasrec' or args.model == 'sasrecm' or args.model == 'sasrecs' or args.model == 'sasrecm2' or args.model=='hgn' or args.model == 'fmlp' or args.model == 'unicdrseq' :
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
        args.num_domains
        for target_domain in range(1,args.num_domains):
            
            valid_seqs = torch.arange(dataset.num_seqs)
            valid_domains = dataset.valid_domains
            valid_seqs = valid_seqs[(valid_domains == target_domain ).flatten()]           
            result = evaluate_domain_topk(model, valid_seqs, dataset, target_domain, 'valid', 'eval', 10)

            valid_seqs = torch.arange(dataset.num_seqs)
            valid_domains = dataset.valid_domains
            valid_type = dataset.cd_type

            cross_valid_seqs = valid_seqs[(valid_domains == target_domain ).flatten() & (valid_type==2).flatten()]
            intra_valid_seqs = valid_seqs[(valid_domains == target_domain ).flatten() & (valid_type==1).flatten()]
            cross_result = evaluate_domain_topk(model, cross_valid_seqs, dataset, target_domain, 'valid', 'eval', 10)
            intra_result = evaluate_domain_topk(model, intra_valid_seqs, dataset, target_domain, 'valid', 'eval', 10)
            
            if args.mode == 'develop':
                print("Validation")
                print("Domain:: %s"%(target_domain))
                print("[%s/%s][recall]@10:: %.4lf"%(epoch,args.max_epoch,result["recall"]))
                print("[%s/%s][ndcg  ]@10:: %.4lf"%(epoch,args.max_epoch,result["ndcg"]))
                print("[%s/%s][mrr   ]@10:: %.4lf"%(epoch,args.max_epoch,result["mrr"]))

                print("--cross--")
                print("[%s/%s][recall]@10:: %.4lf"%(epoch,args.max_epoch,cross_result["recall"]))
                print("[%s/%s][ndcg  ]@10:: %.4lf"%(epoch,args.max_epoch,cross_result["ndcg"]))
                print("[%s/%s][mrr   ]@10:: %.4lf"%(epoch,args.max_epoch,cross_result["mrr"]))
                
                print("--intra--")
                print("[%s/%s][recall]@10:: %.4lf"%(epoch,args.max_epoch,intra_result["recall"]))
                print("[%s/%s][ndcg  ]@10:: %.4lf"%(epoch,args.max_epoch,intra_result["ndcg"]))
                print("[%s/%s][mrr   ]@10:: %.4lf"%(epoch,args.max_epoch,intra_result["mrr"]))

            if result["ndcg"] >= best_valid_ndcg[target_domain]:
                best_valid_ndcg[target_domain] = result["ndcg"]
                best_valid_recall[target_domain] = result["recall"]
                best_valid_mrr[target_domain] = result["mrr"]

                best_epoch[target_domain] = epoch
                model_state[target_domain] = copy.deepcopy(model.state_dict())
                
                #Test
                #config
                test_seqs = torch.arange(dataset.num_seqs)
                test_domains = dataset.test_domains
                test_seqs = test_seqs[(test_domains == target_domain ).flatten()]           
                test_type = dataset.cd_type
                
                result50 = evaluate_domain_topk(model, test_seqs, dataset, target_domain, 'test', 'eval', 50)
                result20 = evaluate_domain_topk(model, test_seqs, dataset, target_domain, 'test', 'eval', 20)
                result10 = evaluate_domain_topk(model, test_seqs, dataset, target_domain, 'test', 'eval', 10)
                result5 = evaluate_domain_topk(model, test_seqs, dataset, target_domain, 'test', 'eval', 5)
                result1 = evaluate_domain_topk(model, test_seqs, dataset, target_domain, 'test', 'eval', 1)

                test_seqs = torch.arange(dataset.num_seqs)
                cross_test_seqs = test_seqs[(test_domains == target_domain).flatten() & (test_type==2).flatten()]
                intra_test_seqs = test_seqs[(test_domains == target_domain).flatten() & (test_type==1).flatten()]
                cross_result50 = evaluate_domain_topk(model, cross_test_seqs, dataset, target_domain, 'test', 'eval', 50)
                cross_result20 = evaluate_domain_topk(model, cross_test_seqs, dataset, target_domain, 'test', 'eval', 20)
                cross_result10 = evaluate_domain_topk(model, cross_test_seqs, dataset, target_domain, 'test', 'eval', 10)
                cross_result5 = evaluate_domain_topk(model, cross_test_seqs, dataset, target_domain, 'test', 'eval', 5)
                cross_result1 = evaluate_domain_topk(model, cross_test_seqs, dataset, target_domain, 'test', 'eval', 1)
                intra_result50 = evaluate_domain_topk(model, intra_test_seqs, dataset, target_domain, 'test', 'eval', 50)
                intra_result20 = evaluate_domain_topk(model, intra_test_seqs, dataset, target_domain, 'test', 'eval', 20)
                intra_result10 = evaluate_domain_topk(model, intra_test_seqs, dataset, target_domain, 'test', 'eval', 10)
                intra_result5 = evaluate_domain_topk(model, intra_test_seqs, dataset, target_domain, 'test', 'eval', 5)
                intra_result1 = evaluate_domain_topk(model, intra_test_seqs, dataset, target_domain, 'test', 'eval', 1)
                
                
                #Test according to the position
                test_seqs = torch.arange(dataset.num_seqs)
                test_domains = dataset.test_domains         
                test_type = dataset.test_position_type 
                
                test_pos_seqs1 = test_seqs[(test_domains == target_domain).flatten() &
                                           ((test_type!=-1)&(test_type<=1)).flatten()]
                test_pos_seqs3 = test_seqs[(test_domains == target_domain).flatten() &
                                           ((test_type!=-1)&(test_type<=3)).flatten()]
                test_pos_seqs5 = test_seqs[(test_domains == target_domain).flatten() &
                                           ((test_type!=-1)&(test_type<=5)).flatten()]
                test_pos_seqs10 = test_seqs[(test_domains == target_domain).flatten() &
                                           ((test_type!=-1)&(test_type<=10)).flatten()]
                test_pos_seqs20 = test_seqs[(test_domains == target_domain).flatten() &
                                           ((test_type!=-1)&(test_type<=20)).flatten()]
                
                pos_result1 = evaluate_domain_topk(model, test_pos_seqs1, dataset, target_domain, 'test', 'eval', 20)
                pos_result3 = evaluate_domain_topk(model, test_pos_seqs3, dataset, target_domain, 'test', 'eval', 20)
                pos_result5 = evaluate_domain_topk(model, test_pos_seqs5, dataset, target_domain, 'test', 'eval', 20)
                pos_result10 = evaluate_domain_topk(model, test_pos_seqs10, dataset, target_domain, 'test', 'eval', 20)
                pos_result20 = evaluate_domain_topk(model, test_pos_seqs20, dataset, target_domain, 'test', 'eval', 20)
                #Test according to the sparseness
                test_seqs = torch.arange(dataset.num_seqs)
                test_domains = dataset.test_domains          
                test_type = dataset.test_sparse_type
                
                test_sparse_seqs1 = test_seqs[(test_domains == target_domain).flatten() &
                                           ((test_type!=-1)&(test_type < 1)).flatten()]
                test_sparse_seqs3 = test_seqs[(test_domains == target_domain).flatten() &
                                           ((test_type!=-1)&(test_type < 3)).flatten()]
                test_sparse_seqs5 = test_seqs[(test_domains == target_domain).flatten() &
                                           ((test_type!=-1)&(test_type < 5)).flatten()]
                test_sparse_seqs10 = test_seqs[(test_domains == target_domain).flatten() &
                                           ((test_type!=-1)&(test_type < 10)).flatten()]
                test_sparse_seqs20 = test_seqs[(test_domains == target_domain).flatten() &
                                           ((test_type!=-1)&(test_type < 20)).flatten()]
                
                sparse_result1 = evaluate_domain_topk(model, test_sparse_seqs1, dataset, target_domain, 'test', 'eval', 20)
                sparse_result3 = evaluate_domain_topk(model, test_sparse_seqs3, dataset, target_domain, 'test', 'eval', 20)
                sparse_result5 = evaluate_domain_topk(model, test_sparse_seqs5, dataset, target_domain, 'test', 'eval', 20)
                sparse_result10 = evaluate_domain_topk(model, test_sparse_seqs10, dataset, target_domain, 'test', 'eval', 20)
                sparse_result20 = evaluate_domain_topk(model, test_sparse_seqs20, dataset, target_domain, 'test', 'eval', 20)
                
                
                best_test_recall[str(target_domain)+"total"+"50"] = result50["recall"]
                best_test_recall[str(target_domain)+"total"+"20"] = result20["recall"]
                best_test_recall[str(target_domain)+"total"+"10"] = result10["recall"]
                best_test_recall[str(target_domain)+"total"+"5"] = result5["recall"]
                best_test_recall[str(target_domain)+"total"+"1"] = result1["recall"]

                best_test_ndcg[str(target_domain)+"total"+"50"] = result50["ndcg"]
                best_test_ndcg[str(target_domain)+"total"+"20"] = result20["ndcg"]
                best_test_ndcg[str(target_domain)+"total"+"10"] = result10["ndcg"]
                best_test_ndcg[str(target_domain)+"total"+"5"] = result5["ndcg"]
                best_test_ndcg[str(target_domain)+"total"+"1"] = result1["ndcg"]

                best_test_mrr[str(target_domain)+"total"+"50"] = result50["mrr"]
                best_test_mrr[str(target_domain)+"total"+"20"] = result20["mrr"]
                best_test_mrr[str(target_domain)+"total"+"10"] = result10["mrr"]
                best_test_mrr[str(target_domain)+"total"+"5"] = result5["mrr"]
                best_test_mrr[str(target_domain)+"total"+"1"] = result1["mrr"]
                
                #cross-domain-transition
                best_test_recall[str(target_domain)+"cross"+"50"] = cross_result50["recall"]
                best_test_recall[str(target_domain)+"cross"+"20"] = cross_result20["recall"]
                best_test_recall[str(target_domain)+"cross"+"10"] = cross_result10["recall"]
                best_test_recall[str(target_domain)+"cross"+"5"] = cross_result5["recall"]
                best_test_recall[str(target_domain)+"cross"+"1"] = cross_result1["recall"]
                best_test_ndcg[str(target_domain)+"cross"+"50"] = cross_result50["ndcg"]
                best_test_ndcg[str(target_domain)+"cross"+"20"] = cross_result20["ndcg"]
                best_test_ndcg[str(target_domain)+"cross"+"10"] = cross_result10["ndcg"]
                best_test_ndcg[str(target_domain)+"cross"+"5"] = cross_result5["ndcg"]
                best_test_ndcg[str(target_domain)+"cross"+"1"] = cross_result1["ndcg"]
                best_test_mrr[str(target_domain)+"cross"+"50"] = cross_result50["mrr"]
                best_test_mrr[str(target_domain)+"cross"+"20"] = cross_result20["mrr"]
                best_test_mrr[str(target_domain)+"cross"+"10"] = cross_result10["mrr"]
                best_test_mrr[str(target_domain)+"cross"+"5"] = cross_result5["mrr"]
                best_test_mrr[str(target_domain)+"cross"+"1"] = cross_result1["mrr"]
                #intra-transition
                best_test_recall[str(target_domain)+"intra"+"50"] = intra_result50["recall"]
                best_test_recall[str(target_domain)+"intra"+"20"] = intra_result20["recall"]
                best_test_recall[str(target_domain)+"intra"+"10"] = intra_result10["recall"]
                best_test_recall[str(target_domain)+"intra"+"5"] = intra_result5["recall"]
                best_test_recall[str(target_domain)+"intra"+"1"] = intra_result1["recall"]
                
                best_test_ndcg[str(target_domain)+"intra"+"50"] = intra_result50["ndcg"]
                best_test_ndcg[str(target_domain)+"intra"+"20"] = intra_result20["ndcg"]
                best_test_ndcg[str(target_domain)+"intra"+"10"] = intra_result10["ndcg"]
                best_test_ndcg[str(target_domain)+"intra"+"5"] = intra_result5["ndcg"]
                best_test_ndcg[str(target_domain)+"intra"+"1"] = intra_result1["ndcg"]

                best_test_mrr[str(target_domain)+"intra"+"50"] = intra_result50["mrr"]
                best_test_mrr[str(target_domain)+"intra"+"20"] = intra_result20["mrr"]
                best_test_mrr[str(target_domain)+"intra"+"10"] = intra_result10["mrr"]
                best_test_mrr[str(target_domain)+"intra"+"5"] = intra_result5["mrr"]
                best_test_mrr[str(target_domain)+"intra"+"1"] = intra_result1["mrr"]
                
                
                #Position -aware evaluation
                best_test_ndcg[str(target_domain)+"pos"+"1"] = pos_result1["ndcg"]
                best_test_ndcg[str(target_domain)+"pos"+"3"] = pos_result3["ndcg"]
                best_test_ndcg[str(target_domain)+"pos"+"5"] = pos_result5["ndcg"]
                best_test_ndcg[str(target_domain)+"pos"+"10"] = pos_result10["ndcg"]
                best_test_ndcg[str(target_domain)+"pos"+"20"] = pos_result20["ndcg"]
                
                best_test_mrr[str(target_domain)+"pos"+"1"] = pos_result1["mrr"]
                best_test_mrr[str(target_domain)+"pos"+"3"] = pos_result3["mrr"]
                best_test_mrr[str(target_domain)+"pos"+"5"] = pos_result5["mrr"]
                best_test_mrr[str(target_domain)+"pos"+"10"] = pos_result10["mrr"]
                best_test_mrr[str(target_domain)+"pos"+"20"] = pos_result20["mrr"]
                
                #sparse -aware evaluation
                best_test_ndcg[str(target_domain)+"sparse"+"1"] = sparse_result1["ndcg"]
                best_test_ndcg[str(target_domain)+"sparse"+"3"] = sparse_result3["ndcg"]
                best_test_ndcg[str(target_domain)+"sparse"+"5"] = sparse_result5["ndcg"]
                best_test_ndcg[str(target_domain)+"sparse"+"10"] = sparse_result10["ndcg"]
                best_test_ndcg[str(target_domain)+"sparse"+"20"] = sparse_result20["ndcg"]
                
                best_test_mrr[str(target_domain)+"sparse"+"1"] = sparse_result1["mrr"]
                best_test_mrr[str(target_domain)+"sparse"+"3"] = sparse_result3["mrr"]
                best_test_mrr[str(target_domain)+"sparse"+"5"] = sparse_result5["mrr"]
                best_test_mrr[str(target_domain)+"sparse"+"10"] = sparse_result10["mrr"]
                best_test_mrr[str(target_domain)+"sparse"+"20"] = sparse_result20["mrr"]
                
        if args.mode == 'develop': #and epoch % 5 ==0:
            for target_domain in range(1,args.num_domains):
                print("Domain:%s"%(target_domain))
                print("Best_epoch:%s"%(best_epoch[target_domain]))
                print("[valid][recall]@10:: %.4lf"%(best_valid_recall[target_domain]))
                print("[valid][ndcg  ]@10:: %.4lf"%(best_valid_ndcg[target_domain]))
                print("[valid][mrr   ]@10:: %.4lf"%(best_valid_mrr[target_domain]))
                print("Domain:%s"%(target_domain))
                print("Best_epoch:%s"%(best_epoch[target_domain]))
                print("[test][recall]@50:: [%.4lf,%.4lf,%.4lf]"%(best_test_recall[str(target_domain)+"total"+"50"],best_test_recall[str(target_domain)+"cross"+"50"],best_test_recall[str(target_domain)+"intra"+"50"]))
                print("[test][ndcg  ]@50:: [%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"total"+"50"],best_test_ndcg[str(target_domain)+"cross"+"50"],best_test_ndcg[str(target_domain)+"intra"+"50"]))
                print("[test][mrr   ]@50:: [%.4lf,%.4lf,%.4lf]"%(best_test_mrr[str(target_domain)+"total"+"50"],best_test_mrr[str(target_domain)+"cross"+"50"],best_test_mrr[str(target_domain)+"intra"+"50"]))

                print("[test][recall]@20:: [%.4lf,%.4lf,%.4lf]"%(best_test_recall[str(target_domain)+"total"+"20"],best_test_recall[str(target_domain)+"cross"+"20"],best_test_recall[str(target_domain)+"intra"+"20"]))
                print("[test][ndcg  ]@20:: [%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"total"+"20"],best_test_ndcg[str(target_domain)+"cross"+"20"],best_test_ndcg[str(target_domain)+"intra"+"20"]))
                print("[test][mrr   ]@20:: [%.4lf,%.4lf,%.4lf]"%(best_test_mrr[str(target_domain)+"total"+"20"],best_test_mrr[str(target_domain)+"cross"+"20"],best_test_mrr[str(target_domain)+"intra"+"20"]))

                print("[test][recall]@10:: [%.4lf,%.4lf,%.4lf]"%(best_test_recall[str(target_domain)+"total"+"10"],best_test_recall[str(target_domain)+"cross"+"10"],best_test_recall[str(target_domain)+"intra"+"10"]))
                print("[test][ndcg  ]@10:: [%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"total"+"10"],best_test_ndcg[str(target_domain)+"cross"+"10"],best_test_ndcg[str(target_domain)+"intra"+"10"]))
                print("[test][mrr   ]@10:: [%.4lf,%.4lf,%.4lf]"%(best_test_mrr[str(target_domain)+"total"+"10"],best_test_mrr[str(target_domain)+"cross"+"10"],best_test_mrr[str(target_domain)+"intra"+"10"]))

                print("[test][recall]@5:: [%.4lf,%.4lf,%.4lf]"%(best_test_recall[str(target_domain)+"total"+"5"],best_test_recall[str(target_domain)+"cross"+"5"],best_test_recall[str(target_domain)+"intra"+"5"]))
                print("[test][ndcg  ]@5:: [%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"total"+"5"],best_test_ndcg[str(target_domain)+"cross"+"5"],best_test_ndcg[str(target_domain)+"intra"+"5"]))
                print("[test][mrr   ]@5:: [%.4lf,%.4lf,%.4lf]"%(best_test_mrr[str(target_domain)+"total"+"5"],best_test_mrr[str(target_domain)+"cross"+"5"],best_test_mrr[str(target_domain)+"intra"+"5"]))

                print("[test][recall]@1:: [%.4lf,%.4lf,%.4lf]"%(best_test_recall[str(target_domain)+"total"+"1"],best_test_recall[str(target_domain)+"cross"+"1"],best_test_recall[str(target_domain)+"intra"+"1"]))
                print("[test][ndcg  ]@1:: [%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"total"+"1"],best_test_ndcg[str(target_domain)+"cross"+"1"],best_test_ndcg[str(target_domain)+"intra"+"1"]))
                print("[test][mrr   ]@1:: [%.4lf,%.4lf,%.4lf]"%(best_test_mrr[str(target_domain)+"total"+"1"],best_test_mrr[str(target_domain)+"cross"+"1"],best_test_mrr[str(target_domain)+"intra"+"1"]))
                print("[test][position-ndcg]@20:: [%.4lf,%.4lf,%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"pos"+"1"],best_test_ndcg[str(target_domain)+"pos"+"3"],best_test_ndcg[str(target_domain)+"pos"+"5"],best_test_ndcg[str(target_domain)+"pos"+"10"],best_test_ndcg[str(target_domain)+"pos"+"20"]))
                print("[test][sparse-ndcg]@20:: [%.4lf,%.4lf,%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"sparse"+"1"],best_test_ndcg[str(target_domain)+"sparse"+"3"],best_test_ndcg[str(target_domain)+"sparse"+"5"],best_test_ndcg[str(target_domain)+"sparse"+"10"],best_test_ndcg[str(target_domain)+"sparse"+"20"]))
                
if args.mode == 'tune':
    print(args)
    objs = {"hyperparams": args}
    for target_domain in range(1,args.num_domains):
        with open('./knowledge/'+'%s'%(args.dataset)+'/'+'%s'%(target_domain)+'/hyperparams/'+'%s_%s-%s_%.4lf'%(args.model, args.window_length, args.dims, float(best_valid_ndcg[target_domain]))+'.pkl','wb') as f:
            pickle.dump(objs, f)
            print("Domain:%s"%(target_domain))
            print("Best_epoch:%s"%(best_epoch[target_domain]))
            print("[valid][recall]@10:: %.4lf"%(best_valid_recall[target_domain]))
            print("[valid][ndcg  ]@10:: %.4lf"%(best_valid_ndcg[target_domain]))
            print("[valid][mrr   ]@10:: %.4lf"%(best_valid_mrr[target_domain]))
            print("Domain:%s"%(target_domain))
            print("Best_epoch:%s"%(best_epoch[target_domain]))
            print("[test][recall]@50:: [%.4lf,%.4lf,%.4lf]"%(best_test_recall[str(target_domain)+"total"+"50"],best_test_recall[str(target_domain)+"cross"+"50"],best_test_recall[str(target_domain)+"intra"+"50"]))
            print("[test][ndcg  ]@50:: [%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"total"+"50"],best_test_ndcg[str(target_domain)+"cross"+"50"],best_test_ndcg[str(target_domain)+"intra"+"50"]))
            print("[test][mrr   ]@50:: [%.4lf,%.4lf,%.4lf]"%(best_test_mrr[str(target_domain)+"total"+"50"],best_test_mrr[str(target_domain)+"cross"+"50"],best_test_mrr[str(target_domain)+"intra"+"50"]))

            print("[test][recall]@20:: [%.4lf,%.4lf,%.4lf]"%(best_test_recall[str(target_domain)+"total"+"20"],best_test_recall[str(target_domain)+"cross"+"20"],best_test_recall[str(target_domain)+"intra"+"20"]))
            print("[test][ndcg  ]@20:: [%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"total"+"20"],best_test_ndcg[str(target_domain)+"cross"+"20"],best_test_ndcg[str(target_domain)+"intra"+"20"]))
            print("[test][mrr   ]@20:: [%.4lf,%.4lf,%.4lf]"%(best_test_mrr[str(target_domain)+"total"+"20"],best_test_mrr[str(target_domain)+"cross"+"20"],best_test_mrr[str(target_domain)+"intra"+"20"]))
            
            print("[test][recall]@10:: [%.4lf,%.4lf,%.4lf]"%(best_test_recall[str(target_domain)+"total"+"10"],best_test_recall[str(target_domain)+"cross"+"10"],best_test_recall[str(target_domain)+"intra"+"10"]))
            print("[test][ndcg  ]@10:: [%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"total"+"10"],best_test_ndcg[str(target_domain)+"cross"+"10"],best_test_ndcg[str(target_domain)+"intra"+"10"]))
            print("[test][mrr   ]@10:: [%.4lf,%.4lf,%.4lf]"%(best_test_mrr[str(target_domain)+"total"+"10"],best_test_mrr[str(target_domain)+"cross"+"10"],best_test_mrr[str(target_domain)+"intra"+"10"]))

            print("[test][recall]@5:: [%.4lf,%.4lf,%.4lf]"%(best_test_recall[str(target_domain)+"total"+"5"],best_test_recall[str(target_domain)+"cross"+"5"],best_test_recall[str(target_domain)+"intra"+"5"]))
            print("[test][ndcg  ]@5:: [%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"total"+"5"],best_test_ndcg[str(target_domain)+"cross"+"5"],best_test_ndcg[str(target_domain)+"intra"+"5"]))
            print("[test][mrr   ]@5:: [%.4lf,%.4lf,%.4lf]"%(best_test_mrr[str(target_domain)+"total"+"5"],best_test_mrr[str(target_domain)+"cross"+"5"],best_test_mrr[str(target_domain)+"intra"+"5"]))

            print("[test][recall]@1:: [%.4lf,%.4lf,%.4lf]"%(best_test_recall[str(target_domain)+"total"+"1"],best_test_recall[str(target_domain)+"cross"+"1"],best_test_recall[str(target_domain)+"intra"+"1"]))
            print("[test][ndcg  ]@1:: [%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"total"+"1"],best_test_ndcg[str(target_domain)+"cross"+"1"],best_test_ndcg[str(target_domain)+"intra"+"1"]))
            print("[test][mrr   ]@1:: [%.4lf,%.4lf,%.4lf]"%(best_test_mrr[str(target_domain)+"total"+"1"],best_test_mrr[str(target_domain)+"cross"+"1"],best_test_mrr[str(target_domain)+"intra"+"1"]))
            print("[test][position-ndcg]@20:: [%.4lf,%.4lf,%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"pos"+"1"],best_test_ndcg[str(target_domain)+"pos"+"3"],best_test_ndcg[str(target_domain)+"pos"+"5"],best_test_ndcg[str(target_domain)+"pos"+"10"],best_test_ndcg[str(target_domain)+"pos"+"20"]))
            print("[test][position-mrr]@20:: [%.4lf,%.4lf,%.4lf,%.4lf,%.4lf]"%(best_test_mrr[str(target_domain)+"pos"+"1"],best_test_mrr[str(target_domain)+"pos"+"3"],best_test_mrr[str(target_domain)+"pos"+"5"],best_test_mrr[str(target_domain)+"pos"+"10"],best_test_mrr[str(target_domain)+"pos"+"20"]))
            print("[test][sparse-ndcg]@20:: [%.4lf,%.4lf,%.4lf,%.4lf,%.4lf]"%(best_test_ndcg[str(target_domain)+"sparse"+"1"],best_test_ndcg[str(target_domain)+"sparse"+"3"],best_test_ndcg[str(target_domain)+"sparse"+"5"],best_test_ndcg[str(target_domain)+"sparse"+"10"],best_test_ndcg[str(target_domain)+"sparse"+"20"]))
            print("[test][sparse-mrr]@20:: [%.4lf,%.4lf,%.4lf,%.4lf,%.4lf]"%(best_test_mrr[str(target_domain)+"sparse"+"1"],best_test_mrr[str(target_domain)+"sparse"+"3"],best_test_mrr[str(target_domain)+"sparse"+"5"],best_test_mrr[str(target_domain)+"sparse"+"10"],best_test_mrr[str(target_domain)+"sparse"+"20"]))
            
            torch.save(model_state[target_domain], './knowledge/%s/%s/%s_%s-%s_%.4lf'%(args.dataset, target_domain, args.model, args.window_length, args.dims, best_valid_ndcg[target_domain]))
