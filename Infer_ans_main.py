import os
import csv
from tqdm import tqdm
import logging
import pickle as pkl
import argparse
import multiprocessing as mp
import random
import json
from Infer_Answer1 import infer_and_answer
from compute_scores import compute_score_main
import time


logging.basicConfig(level=logging.INFO)

def mergeDictionary(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = value.union(dict_1[key])
   return dict_3

def main():    

    entity_triplets_file = os.path.join(args.output_path, "entity_triplets.pkl")
    id2ent_file = os.path.join(args.data_path, "id2ent.pkl")
    id2rel_file = os.path.join(args.data_path, "id2rel.pkl")
    stats_file = os.path.join(args.data_path, "stats.txt")
    queries = pkl.load(open(os.path.join(args.data_path, "queries.pkl"), "rb"))
    pred_path = os.path.join(args.output_path, args.prediction_path)
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    
    # llm = LLM(model_name=args.llm_name)
    model = infer_and_answer(entity_triplets_file, id2ent_file, id2rel_file, stats_file, args.rel_width, 
                             args.ent_width, args.fuzzy_rule, args.model_name, args.prune, args.score_rule, 
                             args.normalize_rule)
    idx_to_queries = pkl.load(open(os.path.join(args.output_path, "idx2query.pkl"), "rb"))
    
    def answer(qtype, qpattern):
        logical_queries = queries[qpattern] #set
        size = len(logical_queries)
        
        if args.random_size > 0:
            idx_list = random.sample(range(size), args.random_size)
            des = f"random {args.random_size}"
            random_path = os.path.join(args.output_path, "random_list")
            if not os.path.exists(random_path):
                os.makedirs(random_path)
            with open(os.path.join(random_path, f"{qtype}_random_list.json"), "w") as f:
                json.dump(idx_list, f)
        else:
            args.whole_size = size
            idx_list = [i for i in range(size)]
            des = f"total {size}"
        for idx in tqdm(idx_list, desc=f"{qtype} {qpattern} {des}"):
            query = idx_to_queries[qtype][idx]
            model.answer_query(logical_query=query, query_type=qtype, idx=idx, output_path=pred_path)
        print(f"avg prompt len = {model.prompt_length / model.llm_cnt}")
        print(f"avg rel prompt len = {model.rel_prompt_length / model.llm_cnt}")
        print(f"avg rel set size = {model.rel_set_size / model.llm_cnt}")

    if args.qtype == "all":
        for qtype, qpattern in model.q_structs.items():
            answer(qtype=qtype, qpattern=qpattern)
    else:
        for qtype, qpattern in model.q_structs.items():
            if qtype == args.qtype:
                answer(qtype=qtype, qpattern=qpattern)

    print(f"there are {model.empty_cnt} queries to which LLM fails to generate answers.")
    print(f"llm time = {model.llm_time} s.")  
    
    

if __name__=="__main__":
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="../data/NELL-betae", help="Path to raw data.")
    parser.add_argument('--output_path', type=str, default="../data/NELL-betae/processed", help="Path to output the processed files.")
    parser.add_argument('--rel_width', type=int, default=50, help="Ask LLM to retrieve top rel_width relations each time.")
    parser.add_argument('--ent_width', type=int, default=25, help="Retrieve top ent_width entities from fuzzy vector.")
    parser.add_argument('--prune', type=float, default=0.9, help="if entities score add up to {prune} then omit the rest.")
    parser.add_argument('--fuzzy_rule', type=str, default="min_max", help="min_max/ prod/ lukas.")
    parser.add_argument('--model_name', type=str, default="Meta-Llama-3-8B-Instruct")
    parser.add_argument('--qtype', type=str, default="2p")
    parser.add_argument('--score_rule', type=str, default="max", help="the rule when rel proj. max/sum.") #origin: sum
    parser.add_argument('--normalize_rule', type=str, default="standard_norm", help="choose a normalize function. min_max_norm/standard_norm/l2_norm/sigmoid/softmax.")
    parser.add_argument("--random_size", type=int, default=50, help="randomly select k queries for each qtype. if size=0 then use the whole dataset.")
    parser.add_argument("--whole_size", type=int, default=0, help="randomly select k queries for each qtype. if size=0 then use the whole dataset.")
    
    parser.add_argument('--ground_truth_path', type=str, default="sorted_answers", help="Path to ground truth data.")
    parser.add_argument('--prediction_path', type=str, default="preds", help="Path to the prediction files.")
    parser.add_argument('--log_score_path', type=str, default="scores", help="Path to log scores")
    parser.add_argument('--score_file', type=str, default="2p.txt", help="file name to log scores")

    args = parser.parse_args()
    main()

    compute_score_main(args)

    end = time.time()
    print(f"total time = {end-start} s.")


