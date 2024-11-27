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

    def get_paths_from_head(self): # for each level
        cur_heads = deepcopy(self.head_ents)
        self.head_ents = {} #add tail of this level. i.e. head for next level.
        for ent_id, prepaths in cur_heads.items():
            for triplet in self.train_G[ent_id]:
                h, r, t = triplet
                for prepath in prepaths:
                    path = prepath + (r,)

                    if t in self.tail2path:
                        self.tail2path[t].add(path)
                    else:
                        self.tail2path[t] = {path}

                    if t in self.head_ents:
                        self.head_ents[t].add(path)
                    else:
                        self.head_ents[t] = {path}

                    self.all_paths_from_head.add(path)

    def make_dataset(self, ids, mode):
        save_list = []

        for i in tqdm(ids, desc=f"making {mode} dataset"):
            q = self.id2q["1p"][i]
            (h, (r,)) = q
            ans_set = self.answers[q]

            self.head_ents = {h: {()}} #head of cur level : pre path(tuple)
            self.all_paths_from_head = set() #all paths starting from h
            self.tail2path = {} #tail id: set of paths from h to tail

            for depth in range(3):
                self.get_paths_from_head()

            gt_paths = set()
            miss_path_cnt = 0

            for t in ans_set:
                try:
                    h_t_paths = self.tail2path[t]
                    gt_paths = gt_paths.union(h_t_paths)
                except KeyError:
                    miss_path_cnt += 1
            
            false_paths = self.all_paths_from_head - gt_paths

            neg_examples = random.sample(false_paths, min(len(false_paths), self.neg_num)) #list of tuples of rid
            pos_examples = random.sample(gt_paths, min(len(gt_paths), self.pos_num))

            relation = f"({id2rel[r]},)"
            neg_paths = []
            pos_paths = []
            for path_tuple in neg_examples:
                rname_list = [self.id2rel[rid] for rid in path_tuple]
                path_str = "(" + ", ".join(rname_list) + ")"
                neg_paths.append(path_str)
            for path_tuple in pos_examples:
                rname_list = [self.id2rel[rid] for rid in path_tuple]
                path_str = "(" + ", ".join(rname_list) + ")"
                pos_paths.append(path_str)
            
            save_list.append({"relation":relation, "pos_paths": pos_paths, "neg_paths": neg_paths})
        
        with open(os.path.join(args.prefix_path, "llmR", f"encoder_{mode}_data.json"), "w") as f:
            json.dump(save_list, f)


    def main(self):
        total_size = len(self.id2q["1p"])
        all_ids = random.sample(range(total_size), min(15000, total_size))
        #all_ids = random.sample(range(total_size), min(15, total_size))
        val_ids = all_ids[:(len(all_ids)//3)]
        train_ids = all_ids[(len(all_ids)//3):]
        self.make_dataset(train_ids, "train")
        self.make_dataset(val_ids, "val")

        
def gen_headent_triplets():#head ent id: set of triplets
    entity_triplets = {}
    triplet_file = os.path.join(f"{args.data_path}", "train.txt")

    with open(triplet_file,"r") as kg_data_file:
        kg_tsv_file = csv.reader(kg_data_file, delimiter="\t")
        for line in kg_tsv_file:
            e1, r, e2 = map(int,line)
            triplet = (e1, r, e2)
            if e1 in entity_triplets: entity_triplets[e1].add(triplet)
            else: entity_triplets[e1] = set([triplet])

    with open(os.path.join(args.prefix_path, args.trainG_file) , "wb") as entity_triplets_file:
        pkl.dump(entity_triplets, entity_triplets_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_path", type=str, default="../data/NELL0/processed")
    parser.add_argument("--data_path", type=str, default="../data/NELL0")
    parser.add_argument("--trainG_file", type=str, default="llmR/train-headent-triplets.pkl")
    parser.add_argument("--id2q_file", type=str, default="test-id2q.pkl")

    parser.add_argument("--pos_num", type=int, default=5)
    parser.add_argument("--neg_num", type=int, default=5)
    args = parser.parse_args()

    start = time.time()

    if not os.path.exists(os.path.join(args.prefix_path, args.trainG_file)):
        gen_headent_triplets()

    answers = pkl.load(open(os.path.join(args.data_path, "test-answers.pkl"), "rb"))
    queries = pkl.load(open(os.path.join(args.data_path, "test-queries.pkl"), "rb"))
    trainG = pkl.load(open(os.path.join(args.prefix_path, args.trainG_file), "rb"))
    id2rel = pkl.load(open(os.path.join(args.data_path, "id2rel.pkl"), "rb"))
    id2ent = pkl.load(open(os.path.join(args.data_path, "id2ent.pkl"), "rb"))
    id2q = pkl.load(open(os.path.join(args.prefix_path, args.id2q_file), "rb"))
    RP = RelProj(answers, queries, trainG, id2rel, id2q, args.pos_num, args.neg_num, id2ent)
    RP.main()

    end = time.time()
    print(f"time = {end-start} s")


