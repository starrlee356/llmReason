from tqdm import tqdm
import argparse
import random
import pickle as pkl
import os
from collections import defaultdict
import re
import numpy as np
import time
from copy import deepcopy
#from _rp_utils import *
from _rp_utils import normalize, _rp_prompt, path_tail_prompt

# only use DR path score & intersection of tail set to get ans & cal MRR

class RelProj:
    def __init__(self, args, LLM, entities_with_score, r_name, ent_num, ent_triplets, id2rel, DR, ans, id2ent, question):
        self.args = args
        self.LLM = LLM
        self.entities_with_score = entities_with_score
        self.ent_triplets = ent_triplets
        self.id2rel = id2rel
        self.r_name = r_name
        self.ent_num = ent_num 
        self.DR = DR
        self.ans = set(ans) # list of ans id of this q. only for 1p
        self.id2ent = id2ent
        self.question = question

        self.llm_time = 0
        self.empty_cnt = 0
        
        self.search_time = 0
        self.DR_time = 0
        
        self.path2tail = {} #tuple path: set of tail id. for all paths from all ents in entities_with_score
        
        #for debug
        self.hit_rate = 0
        self.hit_rate_wo_prune = 0
        self.hit_rate_prune_tail = 0
        
        self.mrr_wo_llm = 0
        self.empty_inter_id = 0
        self.all_path_top_score = 0 # fuse score
        self.DR_path_top_score = 0 # fuse score

    def search_per_level(self): # for each level
        cur_heads = deepcopy(self.head_ents) #self.head_ents:tail of cur level i.e. head for next level. int ent:set{tuple prepath}
        self.head_ents = {} #add tail of this level
        for ent_id, prepaths in cur_heads.items():
            for triplet in self.ent_triplets[ent_id]:
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
    
    def get_cand_paths(self):
        self.paths_with_score = {} #*paths from all head in entities_with_score* : *score from head_score & DR_path_score*. prepare for paths prune(to LLM prompt).
        for head, head_score in self.entities_with_score.items():
            self.head_ents = {head: {()}}
            self.paths_from_head = set()
            start_search = time.time()
            for depth in range(self.args.depth):
                self.search_per_level()
            end_search = time.time()
            self.search_time += end_search - start_search
            paths_str = []
            paths_tuple = list(self.paths_from_head) # set -> list
            for path_tuple in paths_tuple:
                paths_str.append(self.path_tuple2str(path_tuple))
            
            start_DR = time.time()
            path_scores = self.DR.cal_scores([f"({self.r_name},)"] + paths_str) #tensor on cpu, shape=[len(all_paths)]. norm: F.sigmoid.
            path_scores = path_scores.tolist()
            for path_id, path_tuple in enumerate(paths_tuple):
                DR_path_score = path_scores[path_id] * head_score
                if path_tuple in self.paths_with_score:
                    pre_score = self.paths_with_score[path_tuple]
                    self.paths_with_score[path_tuple] = max(pre_score, DR_path_score)
                else:
                    self.paths_with_score[path_tuple] = DR_path_score
            end_DR = time.time()
            self.DR_time += end_DR - start_DR

        K = min(len(self.paths_with_score), self.args.path_width)
        all_items = sorted(self.paths_with_score.items(), key=lambda x:x[1], reverse=True) #list of tuple(path,score)
        topK_items = all_items[:K] 
        topK_paths = [path for path, _ in topK_items]
        
        # for debug
        all_ans = set()
        all_ans_wo_prune = set()
        for tails in self.path2tail.values():
            all_ans_wo_prune = all_ans_wo_prune.union(tails)
        hit_ans_wo_prune = all_ans_wo_prune & self.ans
        self.hit_rate_wo_prune = len(hit_ans_wo_prune) / len(self.ans) #1
        
        for path in topK_paths:
            all_ans = all_ans.union(self.path2tail[path])
        hit_ans = all_ans & self.ans
        self.hit_rate = len(hit_ans) / len(self.ans) 
      
        return topK_paths, topK_items



    def run(self):        
        top_paths, top_paths_with_score = self.get_cand_paths() #list of path_tuple; list of (path_tuple, score)
        tails_with_score = {}

        # whether path from trainG is pure
        all_path_info = {}
        DR_path_info = {}

        for path, score in self.paths_with_score.items():
            tail_set = self.path2tail[path]
            hit_set = tail_set & self.ans
            hits_tails = len(hit_set) / len(tail_set)
            hits_ans = len(hit_set) / len(self.ans)
            info = f"len ans = {len(self.ans)}, len tails = {len(tail_set)}, len hits = {len(hit_set)}, origin path score = {score}"
            all_path_info[path] = {"hits_tails":hits_tails, "hits_ans":hits_ans, "fuse":hits_ans*hits_tails, "info":info}
        sorted_all_paths = sorted(all_path_info.items(), key=lambda x:x[1]["fuse"], reverse=True)
        self.all_path_top_score = sorted_all_paths[0][1]["fuse"]

        for path, score in top_paths_with_score:
            tail_set = self.path2tail[path]
            hit_set = tail_set & self.ans
            hits_tails = len(hit_set) / len(tail_set)
            hits_ans = len(hit_set) / len(self.ans)
            info = f"len ans = {len(self.ans)}, len tails = {len(tail_set)}, len hits = {len(hit_set)}, origin path score = {score}"
            DR_path_info[path] = {"hits_tails":hits_tails, "hits_ans":hits_ans, "fuse":hits_ans*hits_tails, "info":info}

        sorted_DR_paths = sorted(DR_path_info.items(), key=lambda x:x[1]["fuse"], reverse=True)
        self.DR_path_top_score = sorted_DR_paths[0][1]["fuse"]
        
        #print("cur q fin")

        """
        # check intersection
        common_tails = self.path2tail[top_paths[0]]
        for path_id in range(1, len(top_paths)):
            print(f"size common tail = {len(common_tails)}")
            common_tails = common_tails & self.path2tail[top_paths[path_id]]
            if len(common_tails) < 1:
                self.empty_inter_id += path_id
                print(f"common tail = empty at path{path_id}")
                break
      
        # rank scores. 
        for i, (path, score) in enumerate(top_paths_with_score):
            path_rank = i + 1
            tails = self.path2tail[path]
            for tail in tails:
                if tail in tails_with_score:
                    tails_with_score[tail] += score / path_rank
                else:
                    tails_with_score[tail] = score / path_rank
        
        # accumulate scores. (random 100q) time=540s. MRR=32
        for _, (path, score) in enumerate(top_paths_with_score):
            tails = self.path2tail[path]
            for tail in tails:
                if tail in tails_with_score:
                    tails_with_score[tail] += score 
                else:
                    tails_with_score[tail] = score 
        

        # max
        for _, (path, score) in enumerate(top_paths_with_score):
            tails = self.path2tail[path]
            for tail in tails:
                if tail in tails_with_score:
                    tails_with_score[tail] = max(score, tails_with_score[tail])
                else:
                    tails_with_score[tail] = score 

        # check ans hit in top_paths order 
        for id, path in enumerate(top_paths):
            if not self.ans:
                break
            hit_set = self.ans & self.path2tail[path]
            if not hit_set:
                continue
            for hit in hit_set:
                self.ans.remove(hit)
            print(f"path{id}, hit {len(hit_set)} / {len(self.path2tail[path])}, left ans = {len(self.ans)}")
        """
        preds_with_score = sorted(tails_with_score.items(), key=lambda x:x[1], reverse=True) #list of (tail id, score)
        preds = [tailid for tailid, _ in preds_with_score]
        mrr = compute_mrr_score(self.ans, preds)
        #self.mrr_wo_llm += mrr


        
        
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


