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

rel2path = pkl.load(open("../data/NELL0/processed/llmR/rel2path.pkl", "rb"))
#train_a = pkl.load(open("../data/NELL0/train-answers.pkl", "rb"))
#train_q = pkl.load(open("../data/NELL0/train-queries.pkl", "rb"))
train_a = pkl.load(open("../data/NELL0/test-answers.pkl", "rb"))
train_q = pkl.load(open("../data/NELL0/test-queries.pkl", "rb"))
train_G = pkl.load(open("../data/NELL0/processed/llmR/train-headent-triplets.pkl", "rb"))
id2rel = pkl.load(open("../data/NELL0/id2rel.pkl", "rb"))

class RelProj:
    def __init__(self, rel2path, train_a, train_q, train_G):
        self.rel2path = rel2path
        self.train_a = train_a
        self.train_q = train_q
        self.train_G = train_G
        self.empty_cnt = 0

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
        arr_score = normalize(arr_score, "softmax")
        for i, path_id in enumerate(path_ids):
            llm_path_score[path_id] = arr_score[i]

        return True, llm_path_score


    def main(self):
        llm = LLM_vllm("llama3:8b")
        queries = self.train_q[('e', ('r',))]
        queries = random.sample(queries, 500)
        miss_path_rate = 0
        miss_rel2path_rate = 0
        gt_all_paths_rate = 0
        for q in tqdm(queries):
            (h, (r,)) = q
            ans_set = self.train_a[q]
            paths_from_rel = self.rel2path[r] #set of tuples
            paths_from_rel.add((r,))
            self.head_ents = {h: {()}} #head of cur level : pre path(tuple)
            self.all_paths_from_head = set() #all paths starting from h
            self.tail2path = {} #tail id: set of paths from h to tail

            for depth in range(3):
                self.get_paths_from_head()

            gt_paths = set()
            ans_len = len(ans_set)
            miss_path_cnt = 0
            miss_rel2path_cnt = 0

            for t in ans_set:
                try:
                    h_t_paths = self.tail2path[t]
                    gt_paths = gt_paths.union(h_t_paths)
                    if len(h_t_paths & paths_from_rel) < 1: #paths_from_rel not help cur tail
                        miss_rel2path_cnt += 1 
                except KeyError:
                    miss_path_cnt += 1
            
            false_paths = self.all_paths_from_head - gt_paths
            false_ex = random.sample(false_paths, min(len(false_paths), 5)) #list
            gt_ex = random.sample(gt_paths, min(len(gt_paths), 10))
            ex_list = []
            for f in false_ex:
                ex_list.append((f, 0))
            for gt in gt_ex:
                ex_list.append((gt, 1))
            random.shuffle(ex_list)
            r_name = id2rel[r]
            prompt = f"""If you start from head entity and follow the right path(a list of relations), you can reach the true tail entity. Therefore it's essential to find the right path.
            Now you are given the true path ({r_name},), please score the below 15 candidate paths based on their relevance to ({r_name},).
            Your goal is to find those paths that best represent the true path ({r_name},).Each score is in [0, 1], the sum of all scores is 1.\n"""
            gt_score = {} #path id: score
            for i, ex in enumerate(ex_list):
                path = ex[0]
                gt_score[i] = ex[1]
                path_str = []
                for r in path:
                    path_str.append(id2rel[r])
                path_str = "(" + ", ".join(path_str) + ")"
                prompt += f"path{i}: {path_str}\n"
            prompt += f"""Example: if input is 'path1: (r1, r2, ...);\npath2: (r3, r4, ...);\npath3...', then your answer should be like: 'path1: 0.8\npath2: 0.1\n...'"
            Your answer should contain no more than {min(15, 10)} paths. Answer path with high score first. Only answer path id and its score with no other text. Your answer is:"""
            res = llm.run(prompt)
            flag, pred_score = self.format_llm_res(res, 15)
            
            miss_path_rate += miss_path_cnt / ans_len
            miss_rel2path_rate += miss_rel2path_cnt / ans_len

            if len(gt_paths) > len(self.all_paths_from_head):
                print(len(gt_paths),len(self.all_paths_from_head))
            gt_all_paths_rate += len(gt_paths) / len(self.all_paths_from_head)

            """
            set1 = paths_from_rel & paths_from_head
            input = all_paths_from_head & paths_from_rel
            gt = paths_from_head
            size1 = len(gt & input) 
            size2 = len(gt)
            print(size1, size2)
            """
        miss_path_rate /= len(queries)
        miss_rel2path_rate /= len(queries)
        gt_all_paths_rate /= len(queries)
        print(f"miss_path_rate = {miss_path_rate}. miss_rel2path_rate = {miss_rel2path_rate}. gt_all_paths_rate = {gt_all_paths_rate}")
          

if __name__ == "__main__":
    start = time.time()
    RP = RelProj(rel2path, train_a, train_q, train_G)
    RP.main()
    end = time.time()
    print(f"time = {end-start} s")
