from tqdm import tqdm
import argparse
import random
import torch
import pickle as pkl
import os
from collections import defaultdict
import re
import numpy as np
import time
from copy import deepcopy
#from _rp_utils import *
from _rp_utils import normalize, _rp_prompt, path_tail_prompt
from path_score.train import scorerDataset
from torch.utils.data import DataLoader

# test path scorer + LLM
class RelProj:
    def __init__(self, args, LLM, entities_with_score, r_name, ent_num, ent_triplets, id2rel, DR, ans, id2ent, question):
        self.args = args
        self.LLM = LLM
        self.entities_with_score = entities_with_score
        self.ent_triplets = ent_triplets
        self.id2rel = id2rel
        self.r_name = r_name
        self.h_name = self.get_head_name(question)
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
        self.tails_size = 0
        
        self.empty_inter_id = 0
        self.all_path_top_score = 0 # fuse score
        self.DR_path_top_score = 0 # fuse score
        self.DR_loader_len = 0
        
        self.mrr_wo_llm = 0
        self.mrr_w_llm = 0
        self.topK_path_hit_ans_rate = 0
        self.llm_res_hit_ans_rate = 0
            
        self.fscore1 = 0
        self.fscore2 = 0
        self.fscore5 = 0

    def get_head_name(self, question):
        pa = r"Which entities are connected to (.+) by relation"
        match = re.findall(pa, question)
        for h in match:
            return h
        
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
            DR_data = []
            paths_tuple = list(self.paths_from_head) # set -> list
            for path_tuple in paths_tuple:
        
                DR_data.append({"path": self.path_tuple2str(path_tuple), "rel": self.question})
            
            start_DR = time.time()
            DR_dataset = scorerDataset(DR_data)
            DR_loader = DataLoader(DR_dataset, batch_size=self.args.DR_bsz)
            self.DR_loader_len += len(DR_loader)
            scores = []
            for batch in DR_loader:
                scores.append(self.DR.infer(batch)) #tensor on cpu, shape=[bsz]-> put in list
            scores = torch.cat(scores) # list of batches to tensor
            scores = scores.view(-1) #[len_paths, 1] -> [len_paths]

            end_DR = time.time()
            self.DR_time += end_DR - start_DR

            top_scores, top_ids = torch.topk(scores, min(len(scores), self.args.path_width))
            top_scores = top_scores.tolist()
            top_ids = top_ids.tolist()
            for id, path_id in enumerate(top_ids):
                path_score = top_scores[id]
                path = paths_tuple[path_id] #tuple
                if path in self.paths_with_score:
                    pre_score = self.paths_with_score[path]
                    self.paths_with_score[path] = max(pre_score, path_score*head_score)
                else:
                    self.paths_with_score[path] = path_score*head_score

        K = min(len(self.paths_with_score), self.args.path_width)
        all_items = sorted(self.paths_with_score.items(), key=lambda x:x[1], reverse=True) #list of tuple(path,score)
        topK_items = all_items[:K] 
        topK_paths = [path for path, _ in topK_items]
        
      
        return topK_paths, topK_items


    def format_llm_res(self, res:str):
        if res.startswith("Error"):
            print(res)

        pattern = r"(\d+): (\d*\.?\d*)"
        llm_path_score = {}
        path_ids = []
        scores = []
        matches = re.findall(pattern, res) # [(tail id, score),...] e.g. [('1', '1.0'), ('2', '0.1'), ('3', '1')]
        for (path_id, score) in matches:
            try:
                path_id = int(path_id)
                score = float(score)
                path_ids.append(path_id)
                scores.append(score)
            except ValueError:
                print(f"ValueError. tail id = {path_id}, score = {score}")
        
        if not matches or len(scores) < 1: 
            self.empty_cnt += 1
            return False, "No score from LLM"
        
        arr_score = np.array(scores)
        arr_score = normalize(arr_score, self.args.normalize_rule)
        for i, path_id in enumerate(path_ids):
            llm_path_score[path_id] = arr_score[i]

        return True, llm_path_score

    def run(self):   
            
        top_paths, top_paths_with_score = self.get_cand_paths() #list of path_tuple; list of (path_tuple, score)

        # preds = list(self.path2tail[top_paths[0]])
        top_tails = []
        for path in top_paths:
            tails = self.path2tail[path]
            for tail in tails:
                if tail not in top_tails:
                    top_tails.append(tail)
        
        prune_preds = top_tails[:50]
      
        self.get_cand_paths()
        all_path_score = {}
        for path in self.paths_from_head:  
            tails = self.path2tail[path]
            hits = self.ans & tails
            score = len(hits)/len(self.ans) * len(hits)/len(tails)
            all_path_score[path] = score
        sorted_all_path_score = sorted(all_path_score.items(), key=lambda x:x[1], reverse=True)
        self.all_path_top_score += sorted_all_path_score[0][1]   
        GT_tails = self.path2tail[sorted_all_path_score[0][0]]
        self.hit_rate_prune_tail += len(GT_tails & self.ans) / len(self.ans)

        # prune_preds = list(self.path2tail[sorted_all_path_score[0][0]])
        # self.tails_size += len(prune_preds[:50])
        # if len(prune_preds)>50:
        #     print(len(prune_preds))
        #     prune_preds = prune_preds[:50]
            
        mrr = compute_mrr_score(self.ans, prune_preds)
        self.mrr_wo_llm += mrr
        self.topK_path_hit_ans_rate += len(set(prune_preds) & self.ans) / len(self.ans)
        # self.tails_size += len(preds)
        
        llm_preds = [] 
        cand_str = ""
        for i, tail in enumerate(prune_preds):
            cand_str += f"{tail}. {self.id2ent[tail]}; "        
        prompt = """
Head entity = concept_mediatype_media_outlets, relation = concept:proxyfor, what is the tail entity?
Unordered candidate tails = 1. concept_lake_new; 2. concept_city_carbondale; 3. concept_city_belleville; 4. concept_city_williston; 5. concept_book_new;
Please score each candidate on a scale from 0 to 1 based on its likelihood to be the correct tail.
Think: The relation concept:proxyfor implies a connection where the head entity could be a "proxy" or stand-in for something else, and the tail entity should be something that the head entity "represents" or is associated with in some way.
1 is a lake, generally wouldn't be a direct proxy for a media outlet. 2~4 are cities, which might have media outlets associated with it, but cities themselves aren’t typically considered proxies for media outlets.
5 is a book, which is a type of media, it's very likely to be the answer.
Answer: 5: 1.0; 2: 0.3; 3: 0.3; 4: 0.3; 1: 0.01;

Head entity = %s, relation = %s, what is the tail entity?
Unordered candidate tails = %s
Please score each candidate on a scale from 0 to 1 based on its likelihood to be the correct tail.
Return only ID and score with on other text("ID: score; ID: score; ...")
"""

        prompt = prompt % (self.h_name, self.r_name, cand_str)
        start = time.time()
        res = self.LLM.run(prompt)
        f, tails_scores = self.format_llm_res(res)
        end = time.time()
        self.llm_time += end - start
        if f:
            tails_scores = sorted(tails_scores.items(), key=lambda x:x[1], reverse=True) # list of (tail id, score)
            llm_preds = [tail for tail, _ in tails_scores]
            self.mrr_w_llm += compute_mrr_score(self.ans, llm_preds)
            self.llm_res_hit_ans_rate += len(set(llm_preds) & self.ans) / len(self.ans)
            




def compute_mrr_score(ans_list, pred_list):
    rank_list = []
    for i, pred in enumerate(pred_list):
        if pred in ans_list:
            rank_list.append((i))
    if not rank_list:
        return 0
    rank_list = sorted(rank_list)
    cur_rank = []
    for i, rank in enumerate(rank_list):
        cur_rank.append(1./(rank-i+1))
    mrr = sum(cur_rank) / len(ans_list)
    if mrr > 1 or len(rank_list) > len(ans_list):
        print("bk")
    return mrr
