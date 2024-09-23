import os
import csv
from tqdm import tqdm
import logging
import pickle as pkl
import argparse
import multiprocessing as mp
from Infer_Answer import infer_and_answer
from llm import LLM

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
    pred_path = os.path.join(args.output_path, "preds")
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    
    # llm = LLM(model_name=args.llm_name)
    model = infer_and_answer(entity_triplets_file, id2ent_file, id2rel_file, stats_file, args.rel_width, args.ent_width, args.fuzzy_rule, args.llm_name)
    
    for qtype, qpattern in model.q_structs.items():
        if qpattern == ('e', ('r', 'r')):
            logical_queries = queries[qpattern]
            print(f"============== {qtype}: {qpattern} ==============")
            for idx, query in tqdm(enumerate(logical_queries)):
                model.answer_query(logical_query=query, query_type=qtype, idx=idx, output_path=pred_path)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/NELL-betae", help="Path to raw data.")
    parser.add_argument('--output_path', type=str, default="data/NELL-betae/processed", help="Path to output the processed files.")
    parser.add_argument('--rel_width', type=int, default=50, help="Ask LLM to retrieve top rel_width relations each time.")
    parser.add_argument('--ent_width', type=int, default=50, help="Retrieve top ent_width entities from fuzzy vector.")
    parser.add_argument('--fuzzy_rule', type=str, default="min_max", help="min_max/ prod/ lukas.")
    parser.add_argument('--llm_name', type=str, default="llama2:70b")
    args = parser.parse_args()
    main()
