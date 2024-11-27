import os
import re
import sys
from tqdm import tqdm
from global_config import *
import numpy as np
import argparse
import json
from datetime import datetime

def clean_string(string):
    clean_str = re.sub(r"[^0-9,]","",string)
    return clean_str

import numpy as np
import pickle as pkl

def compute_mrr_score(ground_truth, predictions):
    if len(ground_truth) == len(predictions) == 0:
        return 1
    reciprocal_ranks = []
    for i, prediction in enumerate(predictions):
        if prediction in ground_truth:
            reciprocal_rank = 1 / (i + 1)
            reciprocal_ranks.append(reciprocal_rank)
    if len(reciprocal_ranks) == 0: return 0
    mrr = sum(reciprocal_ranks)/len(reciprocal_ranks)
    return mrr


def compute_ndcg_score(ground_truth, predictions, k=5):
    if len(ground_truth) == len(predictions) == 0:
        return 1
    relevance_scores = []
    length = min(len(ground_truth),len(predictions))
    k = min(length,k)
    ground_truth = ground_truth[:k]
    predictions = predictions[:k]
    for i in range(k):
        prediction = predictions[i]
        relevance_score = 1 if prediction in ground_truth else 0
        relevance_scores.append(relevance_score)
    dcg_k = np.sum(relevance_scores / np.log2(np.arange(2, k+2)))
    sorted_ground_truth = sorted(ground_truth, reverse=True)
    idcg_k = np.sum([1 if sample in ground_truth else 0 for sample in sorted_ground_truth[:k]] / np.log2(np.arange(2, k+2)))
    ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0
    return ndcg_k

def compute_hits_score(ground_truth, predictions, k=1):
    if len(ground_truth) == len(predictions) == 0:
        return 1
    hits = len(set(predictions[:k]).intersection(set(ground_truth)))
    l = len(predictions[:k])
    if l == 0: l = 1
    return hits/l

def compute_score(qtype, mode, log_path, args, info): 
    # log_score_filename = os.path.join(log_path, args.score_file)
    cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_score_filename = os.path.join(log_path, f"{cur_time}_{args.score_file}")
    if os.path.exists(log_score_filename):
        os.remove(log_score_filename)


    scores = {"hits@1":0,"hits@3":0,"hits@10":0,
            "ndcg@1":0,"ndcg@3":0,"ndcg@10":0,
            "mrr":0}
    
    idx_list = json.load(open(os.path.join(args.prefix_path, args.random_path, f"{qtype}.json"), "r"))
    for idx in idx_list:
        gt_filename = os.path.join(args.prefix_path, args.ground_truth_path, f"{qtype}_{idx}_answer.txt")
        pred_filename = os.path.join(args.prefix_path, args.prediction_path, f"{qtype}_{idx}_predicted_answer.txt")
        
        with open(gt_filename) as gt_f:
            cleaned_gt = clean_string(gt_f.read()).split(",")
            gt = [int(x) for x in cleaned_gt if x.isdigit()]

        with open(pred_filename) as pred_f:
            cleaned_pred = clean_string(pred_f.read()).split(",")
            pred = [int(x) for x in cleaned_pred if x.isdigit()]
        gt = list(dict.fromkeys(gt))
        pred = list(dict.fromkeys(pred))
        scores["hits@1"] += compute_hits_score(gt, pred, k=1)
        scores["hits@3"] += compute_hits_score(gt, pred, k=3)
        scores["hits@10"] += compute_hits_score(gt, pred, k=10)
        scores["ndcg@1"] += compute_ndcg_score(gt, pred, k=1)
        scores["ndcg@3"] += compute_ndcg_score(gt, pred, k=3)
        scores["ndcg@10"] += compute_ndcg_score(gt, pred, k=10)
        scores["mrr"] += compute_mrr_score(gt, pred)

    print("MRR:",scores["mrr"]/len(idx_list))
    with open(log_score_filename, mode) as score_file:
        print(qtype, file=score_file)
        for arg, val in vars(args).items():
            print(f"{arg}: {val}", file=score_file)
        print(info, file=score_file)
        print("HITS@1:",scores["hits@1"]/len(idx_list), file=score_file)
        print("HITS@3:",scores["hits@3"]/len(idx_list), file=score_file)
        print("HITS@10:",scores["hits@10"]/len(idx_list), file=score_file)
        print("NDCG@1:",scores["ndcg@1"]/len(idx_list), file=score_file)
        print("NDCG@3:",scores["ndcg@3"]/len(idx_list), file=score_file)
        print("NDCG@10:",scores["ndcg@10"]/len(idx_list), file=score_file)
        print("MRR:",scores["mrr"]/len(idx_list), file=score_file)

def compute_score_main(args, info):
    log_path = os.path.join(args.prefix_path, args.log_score_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if args.qtype == "all":
        for i, qtype in enumerate(QUERY_STRUCTS.keys()):
            compute_score(qtype, "a", log_path, args, info) # "all1.txt", "all2.txt"...
    else:
        compute_score(args.qtype, "w", log_path, args, info) #"2p.txt"




            

