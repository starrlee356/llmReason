import os
import pickle as pkl
import argparse
import csv
from global_config import *
from tqdm import tqdm

def mergeDictionary(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = value.union(dict_1[key])
   return dict_3

def merge_queries_and_save():
    save_file = os.path.join(args.data_path, "queries.pkl")
    if args.over_write or not os.path.exists(save_file):   
        train_queries_path = os.path.join(f"{args.data_path}","train-queries.pkl")
        test_queries_path = os.path.join(f"{args.data_path}","test-queries.pkl")
        valid_queries_path = os.path.join(f"{args.data_path}","valid-queries.pkl") 
        with open(train_queries_path,"rb") as trainq_file:
            train_queries = pkl.load(trainq_file)
        with open(test_queries_path,"rb") as testq_file:
            test_queries = pkl.load(testq_file)
        with open(valid_queries_path,"rb") as validq_file:
            valid_queries = pkl.load(validq_file)
        temp_queries = mergeDictionary(train_queries, test_queries)
        queries = mergeDictionary(temp_queries,valid_queries) # qtype:[queries] queries:train,test,val
        del(train_queries, test_queries, valid_queries, temp_queries)
        
        with open(save_file, "wb") as f:
            pkl.dump(queries, f)

def merge_answer_and_save():
    save_file = os.path.join(args.data_path, "answers.pkl")
    if args.over_write or not os.path.exists(save_file):  
        train_answers_path = os.path.join(f"{args.data_path}","train-answers.pkl")
        test_easy_answers_path = os.path.join(f"{args.data_path}","test-easy-answers.pkl")
        valid_easy_answers_path = os.path.join(f"{args.data_path}","valid-easy-answers.pkl")
        test_hard_answers_path = os.path.join(f"{args.data_path}","test-hard-answers.pkl")
        valid_hard_answers_path = os.path.join(f"{args.data_path}","valid-hard-answers.pkl")
        with open(train_answers_path,"rb") as traina_file:
            train_a = pkl.load(traina_file)
        with open(test_easy_answers_path,"rb") as testea_file:
            test_easy_a = pkl.load(testea_file)
        with open(valid_easy_answers_path,"rb") as validea_file:
            valid_easy_a = pkl.load(validea_file)
        with open(test_hard_answers_path,"rb") as testha_file:
            test_hard_a = pkl.load(testha_file)
        with open(valid_hard_answers_path,"rb") as validha_file:
            valid_hard_a = pkl.load(validha_file)
        test_answers = mergeDictionary(test_easy_a, test_hard_a)
        valid_answers = mergeDictionary(valid_easy_a, valid_hard_a)
        temp_answers = mergeDictionary(train_a,valid_answers)
        answers = mergeDictionary(temp_answers, test_answers)
        del(train_a, test_easy_a, valid_easy_a, test_hard_a, valid_hard_a, test_answers, valid_answers, temp_answers)
        with open(save_file, "wb") as f:
            pkl.dump(answers, f)

def gen_triplets_and_save():
    if not os.path.exists(f"{args.output_path}"):
        os.makedirs(f"{args.output_path}")
    save_file = os.path.join(f"{args.output_path}","entity_triplets.pkl")

    if args.over_write or not os.path.exists(save_file):  
        entity_triplets = {}
        triplet_files = [os.path.join(f"{args.data_path}","train.txt"), 
                        os.path.join(f"{args.data_path}","valid.txt"), 
                        os.path.join(f"{args.data_path}","test.txt")]
        for triplet_file in triplet_files:
            with open(triplet_file,"r") as kg_data_file:
                kg_tsv_file = csv.reader(kg_data_file, delimiter="\t")
                for line in kg_tsv_file:
                    e1, r, e2 = map(int,line)
                    triplet = (e1, r, e2)
                    if e1 in entity_triplets: entity_triplets[e1].add(triplet)
                    else: entity_triplets[e1] = set([triplet])
                    if e2 in entity_triplets: entity_triplets[e2].add(triplet)
                    else: entity_triplets[e2] = set([triplet])
        with open(save_file,"wb") as entity_triplets_file:
            pkl.dump(entity_triplets, entity_triplets_file)

def gen_ans_txt():
    directory = os.path.join(args.output_path, "answers")
    if not os.path.exists(directory):
        os.makedirs(directory)
    if args.over_write:
        with open(os.path.join(args.data_path, "queries.pkl"), "rb") as q_f:
            queries = pkl.load(q_f)
        with open(os.path.join(args.data_path, "answers.pkl"), "rb") as a_f:
            answers = pkl.load(a_f)

        for q_type in Q_types:
            q = queries[QUERY_STRUCTS[q_type]] # "1p" -> ('e', ('r',))
            for idx, query in enumerate(q):
                file_name = os.path.join(directory, f"{q_type}_{idx}_answer.txt")
                ans_text = ", ".join(map(str, answers[query]))
                with open(file_name, "w") as f:
                    print(ans_text, file=f)

from collections import defaultdict
def gen_sorted_ans():
    directory = os.path.join(args.output_path, "sorted_answers")
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(args.data_path, "queries.pkl"), "rb") as q_f:
        queries = pkl.load(q_f)
    with open(os.path.join(args.data_path, "answers.pkl"), "rb") as a_f:
        answers = pkl.load(a_f)

    idx2query = defaultdict()
    for qtype, qpattern in QUERY_STRUCTS.items():
        logical_queries = queries[qpattern] #set of queries of this qtype
        idx2query[qtype] = {i:q for i,q in enumerate(tqdm(logical_queries))}
        for i, q in tqdm(idx2query[qtype].items()):
            ans_file = os.path.join(directory, f"{qtype}_{i}_answer.txt")
            ans_text = ", ".join(map(str, answers[q]))
            with open(ans_file, "w") as f:
                print(ans_text, file=f)

    save_dict_path = os.path.join(args.output_path, "idx2query.pkl")
    with open(save_dict_path, "wb") as file:
        pkl.dump(idx2query, file)
    
"""
import itertools
from tqdm import tqdm
def gen_QandA():
    directory = os.path.join(args.output_path, "QandA")
    if not os.path.exists(directory):
        os.makedirs(directory)
    if args.over_write:
        with open(os.path.join(args.data_path, "queries.pkl"), "rb") as q_f:
            queries = pkl.load(q_f)
        with open(os.path.join(args.data_path, "answers.pkl"), "rb") as a_f:
            answers = pkl.load(a_f)

        for q_type in Q_types:
            q = queries[QUERY_STRUCTS[q_type]] # "1p" -> ('e', ('r',))
            for idx, query in enumerate(tqdm(itertools.islice(q, 100))):
                file_name = os.path.join(directory, f"{q_type}_{idx}_answer.txt")
                ans_text = str(query) + "\n" + ", ".join(map(str, answers[query]))
                with open(file_name, "w") as f:
                    print(ans_text, file=f)
"""

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/NELL-betae", help="Path to raw data.")
    parser.add_argument('--output_path', type=str, default="../data/NELL-betae/processed", help="Path to output the processed files.")
    parser.add_argument('--over_write', action="store_true", help="default is false: if file exist, will not overwrite. pass --over_write in script to over write.")
    args = parser.parse_args()
    #merge_answer_and_save()
    #merge_queries_and_save()
    #gen_triplets_and_save()
    #gen_ans_txt()
    #gen_QandA()
    gen_sorted_ans()
