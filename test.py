import os
import csv
from tqdm import tqdm
import logging
import pickle as pkl
import argparse
import multiprocessing as mp
import random
import itertools
from Infer_Answer1 import infer_and_answer
from compute_scores import compute_score_main



def main():    

    
    pred_path = os.path.join(args.output_path, args.prediction_path)
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    #logical_queries = queries[('e', ('r',))]
    """
    size = len(logical_queries)
    idx_to_queries = {i:q for i, q in enumerate(logical_queries)}
    idx_list = random.sample(range(size), 10)
    random_path = os.path.join(args.output_path, "random_list")
    if not os.path.exists(random_path):
        os.makedirs(random_path)

    random_file = os.path.join(random_path, f"1p_random_list_tmp.pkl")
    with open(random_file, "wb") as f:
        pkl.dump(idx_list, f)

    for idx in idx_list:
        with open(os.path.join(random_path, f"1p_q.txt"), "a") as qf:
            print(f"question idx={idx}: {idx_to_queries[idx]}", file=qf)

    idx_as = pkl.load(open(random_file, "rb"))
    for idx in idx_as:
        with open(os.path.join(random_path, f"1p_a.txt"), "a") as af:
            print(f"question idx={idx}: {idx_to_queries[idx]}", file=af)
    """  
    id2q = pkl.load(open("../data/NELL-betae/processed/idx2query.pkl", "rb"))
    ans = pkl.load(open(os.path.join(args.data_path, "answers.pkl"), "rb"))

    for i in range(5):
        q = id2q["1p"][i]
        a = ans[q]
        print(f"idx{i}")
        print(a)
        with open(f"../data/NELL-betae/processed/sorted_answers/1p_{i}_answer.txt", "r") as f:
            cont = f.read()
            print(cont)
    

    


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="../data/NELL-betae", help="Path to raw data.")
    parser.add_argument('--output_path', type=str, default="../data/NELL-betae/processed", help="Path to output the processed files.")
    parser.add_argument('--rel_width', type=int, default=50, help="Ask LLM to retrieve top rel_width relations each time.")
    parser.add_argument('--ent_width', type=int, default=50, help="Retrieve top ent_width entities from fuzzy vector.")
    parser.add_argument('--prune', type=float, default=0.9, help="if entities score add up to {prune} then omit the rest.")
    parser.add_argument('--fuzzy_rule', type=str, default="min_max", help="min_max/ prod/ lukas.")
    parser.add_argument('--llm_name', type=str, default="llama3:8b")
    parser.add_argument('--qtype', type=str, default="2p")
    parser.add_argument('--score_rule', type=str, default="max", help="the rule when rel proj. max/sum.") #origin: sum
    parser.add_argument('--normalize_rule', type=str, default="standard_norm", help="choose a normalize function. min_max_norm/standard_norm/l2_norm/sigmoid/softmax.")
    parser.add_argument("--random_size", type=int, default=50, help="randomly select k queries for each qtype. if size=0 then use the whole dataset.")
    parser.add_argument("--whole_size", type=int, default=0, help="randomly select k queries for each qtype. if size=0 then use the whole dataset.")
    
    parser.add_argument('--ground_truth_path', type=str, default="answers", help="Path to ground truth data.")
    parser.add_argument('--prediction_path', type=str, default="preds", help="Path to the prediction files.")
    parser.add_argument('--log_score_path', type=str, default="scores", help="Path to log scores")
    parser.add_argument('--score_file', type=str, default="2p.txt", help="file name to log scores")

    args = parser.parse_args()
    main()


