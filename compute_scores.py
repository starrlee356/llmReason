import os
import re
import sys
from tqdm import tqdm
from global_config import QUERY_STRUCTS
import numpy as np
import argparse
import json

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

def compute_score(qtype, mode, args): 
    log_path = os.path.join(args.output_path, args.log_score_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_score_filename = os.path.join(log_path, args.score_file)
    if os.path.exists(log_score_filename):
        os.remove(log_score_filename)

    if args.random_size > 0:
        idx_list = json.load(open(os.path.join(args.output_path, "random_list", f"{qtype}_random_list.json"), "r"))
    else:
        idx_list = [i for i in range(args.whole_size)]

    scores = {"hits@1":0,"hits@3":0,"hits@10":0,
            "ndcg@1":0,"ndcg@3":0,"ndcg@10":0,
            "mrr":0}
    
    for idx in idx_list:
        gt_filename = os.path.join(args.output_path, args.ground_truth_path, f"{qtype}_{idx}_answer.txt")
        pred_filename = os.path.join(args.output_path, args.prediction_path, f"{qtype}_{idx}_predicted_answer.txt")
        
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

        
    with open(log_score_filename, mode) as score_file:
        print(qtype, file=score_file)
        print("HITS@1:",scores["hits@1"]/len(idx_list), file=score_file)
        print("HITS@3:",scores["hits@3"]/len(idx_list), file=score_file)
        print("HITS@10:",scores["hits@10"]/len(idx_list), file=score_file)
        print("NDCG@1:",scores["ndcg@1"]/len(idx_list), file=score_file)
        print("NDCG@3:",scores["ndcg@3"]/len(idx_list), file=score_file)
        print("NDCG@10:",scores["ndcg@10"]/len(idx_list), file=score_file)
        print("MRR:",scores["mrr"]/len(idx_list), file=score_file)

def compute_score_main(args):
    if args.qtype == "all":
        for i, qtype in enumerate(QUERY_STRUCTS.keys()):
            compute_score(qtype, "a", args) # "all1.txt", "all2.txt"...
    else:
        compute_score(args.qtype, "w", args) #"2p.txt"
                
"""
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default="../data/NELL-betae/processed", help="Path to ground truth data.")
    parser.add_argument('--ground_truth_path', type=str, default="answers", help="Path to ground truth data.")
    parser.add_argument('--prediction_path', type=str, default="preds", help="Path to the prediction files.")
    parser.add_argument('--log_score_path', type=str, default="scores", help="Path to log scores")
    parser.add_argument('--score_file', type=str, default="2p.txt", help="file name to log scores")
    parser.add_argument('--qtype', type=str, default="2p")
    parser.add_argument('--random_size', type=int, default=50)
    parser.add_argument('--whole_size', type=int, default=0) #由infer_ans_main设置
    args = parser.parse_args()
    """



            

