from prompt_list import *
from global_config import *
import numpy as np
import pickle as pkl
from collections import defaultdict
import os
import ollama
import re
import torch
from llm import LLM
import time

class infer_and_answer:
    def __init__(self, entity_triplets_file, id2ent_file, id2rel_file, stats_file, rel_width, ent_width,
                  fuzzy_rule, model_name, prune, score_rule, normalize_rule, batch_size):
        self.entity_triplets = pkl.load(open(entity_triplets_file,"rb"))
        self.id2ent = pkl.load(open(id2ent_file,"rb"))
        self.id2rel = pkl.load(open(id2rel_file,"rb"))
        with open(stats_file, "r") as f:
            for line in f:
                if line.startswith("numentity:"):
                    self.ent_num = int(line.split(":")[1].strip())
                if line.startswith("numrel"):
                    self.rel_num = int(line.split(":")[1].strip())
        self.rel_width = rel_width
        self.ent_width = ent_width
        self.rule = fuzzy_rule
        self.q_structs = QUERY_STRUCTS
        self.model = model_name
        self.empty_cnt = 0
        self.prune = prune
        self.score_rule = score_rule
        self.normalize_rule = normalize_rule
        self.prompt_length = 0
        self.rel_prompt_length = 0
        self.rel_set_size = 0
        self.llm_cnt = 0 
        self.llm_time = 0
        self.batch_size = batch_size

    def search_KG(self, entities_with_scores): # {ent id: score}
        rel2answer = defaultdict(set) # dict. rel name: {(ent_id, ent_score, answer_id)}
        relations = set() # set of rel name.
        for entity, score in entities_with_scores.items():
            for triple in self.entity_triplets[entity]:
                h, r, t = triple
                if entity != h: # 只看从ent出来的边
                    continue
                rel_name = self.id2rel[r]
                relations.add(rel_name)
                rel2answer[rel_name].add((entity, score, t))
        return relations, rel2answer
    
    def run_llm(self, prompt):
        """
        content = ollama.generate(model=self.model, prompt=prompt)
        response = content["response"]
        return response
        """
        llm = LLM(self.model)
        return llm.run(prompt)
    
    def format_llm_res(self, res:str):
        if res.startswith("Error"):
            print(res)
            # 其他错误处理代码

        pattern = r'\d+\.\s*(\S+)\s*\(Score:\s*([0-1](?:\.\d+)?)\)'
        matches = re.findall(pattern, res)
        #relation_dict = {relation: float(score) for relation, score in matches if score > 0}

        relation_dict = {}
        score_arr = np.array([score for _, score in matches]).astype(np.float32) #dtype: "<U3" -> np.float32

        if not matches or score_arr.sum()==0: 
            return False, "No relations found"
        
        score_arr = self.normalize(score_arr)
        for i in range(score_arr.size):
            if score_arr[i] > 0:
                relation_dict[matches[i][0]] = score_arr[i] # relation: normalized score
        
        return True, relation_dict
        
        
    def normalize(self, arr):

        def average_norm(arr):
            total = np.sum(arr)
            if total == 0:
                return arr  
            return arr / total
        
        if self.normalize_rule == "average_norm":
            return average_norm(arr)
        
        def min_max_norm(arr):
            if arr.max() - arr.min() > 0:
                return (arr - arr.min()) / (arr.max() - arr.min())
            else:
                return arr - arr.min()
        
        if self.normalize_rule == "min_max_norm":
            return min_max_norm(arr)
        
        def standard_norm(arr):
            if np.std(arr) == 0:
                 return arr - np.mean(arr)
            return (arr - np.mean(arr)) / np.std(arr)
        
        if self.normalize_rule == "standard_norm":
            return standard_norm(arr)
        
        def sigmoid(arr):
            return 1 / (1 + np.exp(-arr))
        
        if self.normalize_rule == "sigmoid":
            return average_norm(min_max_norm(sigmoid(arr)))
        
        def softmax(arr):
            x = np.exp(arr - np.max(arr))
            if x.sum() > 0:
                return x / x.sum()
            return x
        
        if self.normalize_rule == "softmax":
            return min_max_norm(softmax(arr))
        
        def l2_norm(arr):
            norm = np.linalg.norm(arr, ord=2)
            if norm == 0:
                return arr
            return arr / norm
        
        if self.normalize_rule == "l2_norm":
            return l2_norm(arr)

    def rel_proj(self, vectors, sub_questions): # 接受一个variable的fuzzy vec; 返回它经过rel proj 后的variable的fuzzy vec
        rel2answer_list = []
        prompt_list = []
        ans_vec_list = []

        for i in range(len(vectors)):
            if vectors[i] is None:
                self.empty_cnt += 1
                ans_vec_list.append(None)

            entities_with_scores = self.fuzzyVector_to_entities(vectors[i]) #{ent id: score}
            relations, rel2answer = self.search_KG(entities_with_scores) # dict. rel name: {(ent_id, ent_score, answer_id)}
            rel2answer_list.append(rel2answer)
            prompt = extract_relation_prompt1 % (self.rel_width, sub_questions[i], '; '.join(relations))
            prompt_list.append(prompt)
            
            rel_set_size = len(relations)
            prompt_length = len(prompt)
            rel_prompt_length = len('; '.join(relations))
            #print(f"rel_set_size={rel_set_size}, prompt_len={prompt_length}, rel_prompt_len={rel_prompt_length}")
            #print(prompt)
            self.rel_set_size += rel_set_size
            self.prompt_length += prompt_length
            self.rel_prompt_length += rel_prompt_length
        self.llm_cnt += 1

        start = time.time()
        results = self.run_llm(prompt_list) 
        end = time.time()
        self.llm_time += (end-start)
        
        
        for j in range(len(results)):    
            flag, res = self.format_llm_res(results[j]) # dict. rel name: rel_score
            if not flag:
                print(res)
                ans_vec_list.append(None)
            else:
                relations_with_scores = res
                ans_vec = np.zeros((self.ent_num))
                for rel in relations_with_scores:
                    rel_score = relations_with_scores[rel]
                    for answer in rel2answer_list[j][rel]:
                        _, ent_score, ans_id = answer
                        if self.score_rule == "sum":
                            ans_vec[ans_id] += ent_score * rel_score 
                        if self.score_rule == "max":
                            ans_vec[ans_id] = max(ans_vec[ans_id], ent_score * rel_score)    
                ans_vec = self.normalize(ans_vec)
                ans_vec_list.append(ans_vec)

        return ans_vec_list
    
    def union(self, v1, v2, v3 = None): 
        if v1 is None or v2 is None:
            self.empty_cnt += 1
            return None
        if v3 is None:
            if self.rule == "min_max":
                return self.normalize(np.maximum(v1, v2))
            if self.rule == "prod":
                return self.normalize(v1 + v2 - v1 * v2)
            if self.rule == "lukas":
                return self.normalize(np.minimum(1, v1 + v2))
        else:
            if self.rule == "min_max":
                return self.normalize(np.maximum(np.maximum(v1, v2), v3))
            if self.rule == "prod":
                res = v1 + v2 + v3 - v1 * v2 - v1 * v3 - v2 * v3 + v1 * v2 * v3
                return self.normalize(res)
            if self.rule == "lukas":
                return self.normalize(np.minimum(1, v1 + v2 + v3))
            
    def intersection(self, v1, v2, v3 = None):
        if v1 is None or v2 is None:
            self.empty_cnt += 1
            return None 
        if v3 is None:
            if self.rule == "min_max":
                return self.normalize(np.minimum(v1, v2))
            if self.rule == "prod":
                return self.normalize(v1 * v2)
            if self.rule == "lukas":
                return self.normalize(np.maximum(0, v1 + v2 - 1))
        else:
            if self.rule == "min_max":
                return self.normalize(np.minimum(np.minimum(v1, v2), v3))
            if self.rule == "prod":
                return self.normalize(v1 * v2 * v3)
            if self.rule == "lukas":
                return self.normalize(np.maximum(0, v1 + v2 + v3 - 2))
            
    def negation(self, v):
        if v is None:
            self.empty_cnt += 1
            return None
        return self.normalize(1-v)

    def fuzzyVector_to_entities_wo_prune(self, vector): # from fuzzy vec to answer entity
        if vector is None:
            return None
        
        else:
            scores = []
            sorted_indices = np.argsort(vector)[::-1]
            for i in range(self.ent_width):
                if vector[sorted_indices[i]] <= 0:
                    break
                scores.append(vector[sorted_indices[i]])
            score_sum = sum(scores)

            if score_sum > 0:
                result = {sorted_indices[i]: scores[i]/score_sum for i in range(len(scores))} # ent id: normalized score
            else:
                result = {sorted_indices[i]: scores[i] for i in range(len(scores))} 
        
            return result
        
    def fuzzyVector_to_entities(self, vector): # use prune
        if vector is None:
            return None
        
        else:
            scores = []
            score_sum = 0
            sorted_indices = np.argsort(vector)[::-1]
            for i in range(len(sorted_indices)):
                if score_sum >= self.prune * vector.sum() or i > self.ent_width:
                    break
                scores.append(vector[sorted_indices[i]])
                score_sum += vector[sorted_indices[i]]

            if score_sum > 0:
                #result = {sorted_indices[i]: scores[i]/score_sum for i in range(len(scores))} # ent id: normalized score
                result = {sorted_indices[i]: scores[i] for i in range(len(scores))}
            else:
                result = {} #empty
        
            return result
        
    def ansVector_to_ansFile_wo_prune(self, vector, output_file):
        if vector is not None:
            sorted_indices = np.argsort(vector)[::-1] #全体实体,分数从高到低
            preds = []
            for id in sorted_indices:
                if vector[id] <= 0:
                    break
                else:
                    preds.append(id)
            pred = ", ".join(map(str, preds))
        else:
            pred = ""
        with open(output_file, "w") as prediction_file:
                print(pred, file=prediction_file)

    def ansVector_to_ansFile(self, vectors, batch_idx, output_file):
        pred_list = []
        output_file_list = []
        for i in range(len(vectors)):
            output_file_list.append(output_file % batch_idx[i])
            vector = vectors[i]
            if vector is None:
                pred_list.append("")
            else:
                sorted_indices = np.argsort(vector)[::-1] #全体实体,分数从高到低
                pred_ents = []
                score_sum = 0
                for _, id in enumerate(sorted_indices):
                    #if score_sum >= self.prune * vector.sum() or i > self.ent_width:
                    if score_sum >= self.prune * vector.sum():
                        break
                    pred_ents.append(id)
                    score_sum += vector[id]
                pred = ", ".join(map(str, pred_ents))
                pred_list.append(pred)
        for j in range(len(pred_list)):
            with open(output_file_list[j], "w") as prediction_file:
                print(pred_list[j], file=prediction_file)

    def answer_query(self, batch_query, query_type, batch_idx, output_path):
        output_file = os.path.join(f"{output_path}",f"{query_type}_%s_predicted_answer.txt")
        e1 = r1 = e2 = r2= e3 = r3 = None
        intermediate_variable = "intermediate_variable"

        if query_type=="1p":
            ers = [] 
            vectors = []
            questions = []

            for query in batch_query:
                (e1, (r1,)) = query
                ers.append((e1, r1))
            for e1, r1 in ers:
                vector = np.zeros((self.ent_num))
                vector[e1] = 1
                vectors.append(vector)
                question = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
                questions.append(question)
                ans_vecs = self.rel_proj(vectors, questions)
                self.ansVector_to_ansFile(ans_vecs, batch_idx, output_file)
                #del vector, ans_vec
