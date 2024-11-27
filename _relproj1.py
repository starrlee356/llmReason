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
from _rp_utils import *

# intersection of path from head entity and path from rid
class RelProj:
    def __init__(self, args, LLM, entities_with_score, r_name, rid, ent_num, ent_triplets, id2rel, rel2path):
        self.args = args
        self.LLM = LLM
        self.entities_with_score = entities_with_score
        self.ent_triplets = ent_triplets
        self.id2rel = id2rel
        
        self.paths_with_score = {} #str"rel id, ..": int score. score is from head ent, for prompt pruning.
        self.path2tail = {} # "rel id, ..": set(tail id, ...)  
        self.total_ents = {} # ent id: (int score, str prepath), cand tail ents.
        self.head_ents = self.from_input(entities_with_score) # enties of this level. ent id: (int score, str prepath)
        self.r_name = r_name
        self.rid = rid

        self.llm_cnt = 0
        self.llm_time = 0
        self.empty_cnt = 0
        self.ent_num = ent_num

        self.rel2path = rel2path
    

    def from_input(self, entities_with_score):
        cur_entities = {}
        for ent_id, score in entities_with_score.items():
            cur_entities[ent_id] = (score, "")
        return cur_entities

    def search_KG(self): # for each level
        cur_heads = deepcopy(self.head_ents)
        self.head_ents = {} #add tail of this level. i.e. head for next level.
        for ent_id, (score, prepath) in cur_heads.items():
            for triplet in self.ent_triplets[ent_id]:
                h, r, t = triplet
                if ent_id == h:
                    if prepath:
                        path = prepath + f", {r}"
                    else:
                        path = f"{r}"

                    try:
                        (pre_score, _) = self.total_ents[t]
                        if pre_score < score:
                            self.total_ents[t] = (score, path)
                            self.head_ents[t] = (score, path) 
                    except KeyError:
                        self.total_ents[t] = (score, path)
                        self.head_ents[t] = (score, path)

                    try:
                        self.path2tail[path].add(t)
                    except KeyError:
                        self.path2tail[path] = set([t])

                    path_tuple =  tuple(map(int, path.split(", ")))
                    try:
                        pre_score = self.paths_with_score[path_tuple]
                        self.paths_with_score[path_tuple] = max(pre_score, score)
                    except KeyError:
                        self.paths_with_score[path_tuple] = score

    def search(self, h, path):
        heads = {h}
        for rel in path:
            tmp_heads = set()
            for h in heads:
                triplets = self.ent_triplets[h]
                for (_, r, t) in triplets:
                    if r == rel:
                        tmp_heads.add(t)
            heads = tmp_heads
        return heads

    def get_prompt(self, top_paths):
        prompt = f"If you start from head entity and follow the right path(a list of relations), you can reach the true tail entity. Therefore it's essential to find the right path."
        prompt += f"Now you are given the true path ({self.r_name}), please score the below {len(top_paths)} candidate paths based on their relevance to ({self.r_name})."
        prompt += f"Your goal is to find those paths that best represent the true path ({self.r_name})."
        prompt += f"Each score is in [0, 1], the sum of all scores is 1.\n"
        for id, path in enumerate(top_paths):
            rel_name = []
            for rel_id in path:
                rel_name.append(self.id2rel[int(rel_id)])
            prompt += f"path{id}: (" + ", ".join(rel_name) + ");\n"
        prompt += "Example: if input is 'path1: (r1, r2, ...);\npath2: (r3, r4, ...);\npath3...', then your answer should be like: 'path1: 0.8\npath2: 0.1\n...'"
        prompt += f"Your answer should contain no more than {min(len(top_paths), self.args.prompt_width)} paths. Answer path with high score first."
        prompt += f"Only answer path id and its score with no other texst. Your answer is:"
        return prompt
    

    def format_llm_res(self, res:str, bound):
        if res.startswith("Error"):
            print(res)

        pattern = r"path(\d+): (\d*\.?\d*)"
        llm_path_score = {}
        path_ids = []
        scores = []
        matches = re.findall(pattern, res) # [(path id, score),...] e.g. [('1', '1.0'), ('2', '0.1'), ('3', '1')]
        for (path_id, score) in matches:
            try:
                path_id = int(path_id)
                score = float(score)
                if path_id < bound and path_id >= 0 and score > 0:
                    path_ids.append(path_id)
                    scores.append(score)
            except ValueError:
                print(f"ValueError. path id = {path_id}, score = {score}")
        
        if not matches or len(scores) < 1: 
            self.empty_cnt += 1
            return False, "No path score from LLM"
        
        arr_score = np.array(scores)
        arr_score = normalize(arr_score, self.args.normalize_rule)
        for i, path_id in enumerate(path_ids):
            llm_path_score[path_id] = arr_score[i]

        return True, llm_path_score



    def run(self):
        for _ in range(self.args.depth):
            self.search_KG()
        
        tail_ids = []
        tail_scores = []
        tails_with_scores = {}
        for tail_id, (tail_score, _) in self.total_ents.items():#all possible tail ent. ent id: (int score, str prepath)
            tail_ids.append(tail_id)
            tail_scores.append(tail_score)
        arr_score = np.array(tail_scores)
        arr_score = normalize(arr_score, self.args.normalize_rule)
        for i, tail_id in enumerate(tail_ids):
            tails_with_scores[tail_id] = arr_score[i]
        
        paths_from_head = set(self.paths_with_score.keys()) #再筛选高分path
        paths_from_rel = self.rel2path[self.rid]
        paths_from_rel.add((self.rid,))
        cand_paths = paths_from_rel & paths_from_head

        if len(cand_paths) < self.args.prompt_width:
            k = self.args.prompt_width - len(cand_paths)
            if k < len(paths_from_head):
                cand_paths = list(cand_paths) + random.sample(paths_from_head, k)
            else:
                cand_paths = list(cand_paths) + list(paths_from_head)
        elif len(cand_paths) > self.args.path_width:
            cand_paths = random.sample(cand_paths, self.args.path_width)
        else:
            cand_paths = list(cand_paths)

        #print(len(cand_paths), len(paths_from_head), len(paths_from_rel))
        prompt = self.get_prompt(cand_paths)

        start = time.time()
        res = self.LLM.run(prompt)
        end = time.time()
        self.llm_time += end-start
        self.llm_cnt += 1

        flag, llm_path_score = self.format_llm_res(res, len(cand_paths)) #int ->score

        tail_vec = np.zeros((self.ent_num))
        if flag:
            for path_id, path_score in llm_path_score.items():
                path = cand_paths[path_id] # tuple
                for head, head_score in self.entities_with_score.items():
                    tail_ids = self.search(head, path)
                    if not tail_ids:
                        print("search -> none")
                    for tail_id in tail_ids:
                        tail_vec[tail_id] = max(tail_vec[tail_id], path_score * head_score)
            tail_vec = normalize(tail_vec, self.args.normalize_rule)
            return tail_vec
        else:
            return None

