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


    def path_tuple2str(self, path_tuple):
        r_names = [self.id2rel[rid] for rid in path_tuple]
        path_str = "(" + ", ".join(r_names) + ")"
        return path_str

    def get_prompt(self, top_paths):
        cands_str = ""
        for id, path in enumerate(top_paths):
            tails = self.path2tail[path]
            if len(tails) > self.args.tail_width:
                tails = random.sample(tails, self.args.tail_width)
            cands_str += f"path{id}: {self.path_tuple2str(path)};\n"
        p = f"""
        """

    def format_llm_res(self, res:str): 
        pattern = r"path(\d+)-tail(\d+): (\d*\.?\d*)" 
        match = re.findall(pattern, res)
        formated_res = []
        if match:
            for (pid, tid, score) in match:
                try:
                    formated_res.append((int(pid), int(tid), float(score)))
                except ValueError:
                    continue
            if formated_res:
                return True, formated_res
        
        self.ent_num += 1
        return False, []

    def run(self):        
        top_paths, top_paths_with_score = self.get_cand_paths() #list of path_tuple; list of (path_tuple, score)
       
        aligned_tails = [] #aligned_tails[i] is list of tails to top_paths[i].
        dbg_tails = set() # for debug
        cand_str = ""
        for path_id, path in enumerate(top_paths):
            path_str = self.path_tuple2str(path)
            tails = self.path2tail[path]
            tails = random.sample(tails, min(self.args.tail_width, len(tails)))
            dbg_tails = dbg_tails.union(set(tails))
            aligned_tails.append(tails)
            tails_str = ""
            for tail_id, tail in enumerate(tails):
                tails_str += f"tail{tail_id}: ({self.id2ent[tail]}); "

            cand_str += f"path{path_id}: {path_str}; {tails_str}\n"
        prompt = path_tail_prompt(self.question, cand_str, self.args.prompt_width)
        # for debug
        hit_ans_prune_tail = dbg_tails & self.ans
        self.hit_rate_prune_tail = len(hit_ans_prune_tail) / len(self.ans)
        
        start = time.time()
        res = self.LLM.run(prompt)
        end = time.time()
        self.llm_time += end-start

        flag, llm_path_score = self.format_llm_res(res) #int ->score
        tail_vec = np.zeros((self.ent_num))

        if flag:
            for pid, tid, score in llm_path_score:
                try:
                    tail = aligned_tails[pid][tid]
                    tail_vec[tail] = score
                except IndexError:
                    continue
            tail_vec = normalize(tail_vec, self.args.normalize_rule)
            return tail_vec
        
        return None


