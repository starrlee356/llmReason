import os
from global_config import *
import pickle as pkl
from collections import defaultdict
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import random 
import time
from llm import LLM_vllm
import re
from _rp_utils import *
import argparse
import csv
import json

class RelProj:
    def __init__(self, answers, queries, train_G, id2rel, id2q, pos_num, neg_num, id2ent):
        self.answers = answers
        self.queries = queries
        self.train_G = train_G
        self.id2rel = id2rel
        self.id2q = id2q
        self.pos_num = pos_num
        self.neg_num = neg_num
        self.id2ent = id2ent

        

    def search_per_level(self): # for each level
        cur_heads = deepcopy(self.head_ents) #self.head_ents:tail of cur level i.e. head for next level. int ent:set{tuple prepath}
        self.head_ents = {} #add tail of this level
        for ent_id, prepaths in cur_heads.items():
            for triplet in self.train_G[ent_id]:
                _, r, t = triplet
                for prepath in prepaths:
                    path = prepath + (r,)
                    self.paths_from_head.add(path) #set{tuple(rid)}, set of paths from head ent in entities_with_score

                    if path in self.path2tail: 
                        self.path2tail[path].add(t)
                    else:
                        self.path2tail[path] = {t}
                    
                    if t in self.head_ents: #update head_ents
                        self.head_ents[t].add(path)
                    else:
                        self.head_ents[t] = {path}


    def path_tuple2str(self, path_tuple):
        r_names = [self.id2rel[rid] for rid in path_tuple]
        path_str = "(" + ", ".join(r_names) + ")"
        return path_str
    
    def make_dataset(self, ids, mode, bound):
        save_list_q1 = []
        save_list_q2 = []
        top_score = 0
        for i in tqdm(ids, desc=f"making {mode} dataset"):
            q = self.id2q["1p"][i]
            (h, (r,)) = q
            ans_set = self.answers[q]

            self.head_ents = {h: {()}} #head of cur level : pre path(tuple)
            self.paths_from_head = set() #all paths starting from h
            self.path2tail = {} #tail id: set of paths from h to tail
            
            for depth in range(3):
                self.search_per_level()

            all_path_score = {}
            for path in self.paths_from_head:  
                tails = self.path2tail[path]
                hits = ans_set & tails
                score = len(hits)/len(ans_set) * len(hits)/len(tails)
                all_path_score[path] = score
            sorted_all_path_score = sorted(all_path_score.items(), key=lambda x:x[1], reverse=True)
            # top_score += sorted_all_path_score[0][1]
        # top_score /= len(ids)
        # print(top_score)

            r_name = self.id2rel[r]
            h_name = self.id2ent[h]  
            nl_q1 = f"which entities are connected to head {h_name} by relation {r_name}?" 
            nl_q2 = f"Head entity={h_name}, relation={r_name}."
            pos_path_score = []
            neg_path_score = []
            # top_path,top_score = sorted_all_path_score[0]
            for i, (path, score) in enumerate(sorted_all_path_score[1:]):
                if score > 1/3:
                    pos_path_score.append((path, score))
                else:
                    neg = sorted_all_path_score[i:]
                    neg_path_score = random.sample(neg, min(len(neg), len(pos_path_score)+1))
                    break
            num1 = min(len(pos_path_score), 4)
            num2 = min(len(neg_path_score), 5)
            path_score = [sorted_all_path_score[0]] + random.sample(pos_path_score, num1) + random.sample(neg_path_score, num2)

            for path, score in path_score:
                path = self.path_tuple2str(path)
                save_list_q1.append({"rel": nl_q1, "path": path, "score": score})
                save_list_q2.append({"rel": nl_q2, "path": path, "score": score})
            if len(save_list_q1) > bound:
                break

        save_dir = "../path_score/data_validQ"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        with open(os.path.join(save_dir, f"{mode}_data_q1_balanced2.json"), "w") as f:
            json.dump(save_list_q1, f)
        
        with open(os.path.join(save_dir, f"{mode}_data_q2_balanced2.json"), "w") as f:
            json.dump(save_list_q2, f)

 

    def main(self):
        total_size = len(self.id2q["1p"])
        print(total_size)
        all_ids = random.sample(range(total_size), min(8000, total_size))
        # all_ids = random.sample(range(total_size), min(20, total_size)) # for dbg
        val_ids = all_ids[:(len(all_ids)//4)]
        train_ids = all_ids[(len(all_ids)//4):]
        self.make_dataset(train_ids, "train", 15000)
        self.make_dataset(val_ids, "val", 5000)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_path", type=str, default="../data/NELL0/processed")
    parser.add_argument("--data_path", type=str, default="../data/NELL0")
    parser.add_argument("--trainG_file", type=str, default="llmR/train-headent-triplets.pkl")
    # parser.add_argument("--id2q_file", type=str, default="test-id2q.pkl")
    parser.add_argument("--id2q_file", type=str, default="valid-id2q.pkl")

    parser.add_argument("--pos_num", type=int, default=5)
    parser.add_argument("--neg_num", type=int, default=5)
    args = parser.parse_args()

    start = time.time()


    # answers = pkl.load(open(os.path.join(args.data_path, "test-answers.pkl"), "rb"))
    # queries = pkl.load(open(os.path.join(args.data_path, "test-queries.pkl"), "rb"))
    answers = pkl.load(open(os.path.join(args.data_path, "valid-answers.pkl"), "rb"))
    queries = pkl.load(open(os.path.join(args.data_path, "valid-queries.pkl"), "rb"))
    trainG = pkl.load(open(os.path.join(args.prefix_path, args.trainG_file), "rb"))
    id2rel = pkl.load(open(os.path.join(args.data_path, "id2rel.pkl"), "rb"))
    id2ent = pkl.load(open(os.path.join(args.data_path, "id2ent.pkl"), "rb"))
    id2q = pkl.load(open(os.path.join(args.prefix_path, args.id2q_file), "rb"))
    RP = RelProj(answers, queries, trainG, id2rel, id2q, args.pos_num, args.neg_num, id2ent)
    RP.main()

    end = time.time()
    print(f"time = {end-start} s")


