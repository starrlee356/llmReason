from prompt_list import *
from global_config import *
import numpy as np
import pickle as pkl
from collections import defaultdict
import os
#import ollama
import re
import time
from compute_scores import compute_mrr_score, clean_string
"""prompt, 让LLM输出更短"""

class infer_and_answer:
    def __init__(self, entity_triplets_file, id2ent_file, id2rel_file, stats_file, rel_width, ent_width,
                  fuzzy_rule, model_name, prune, score_rule, normalize_rule, LLM, args):
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
        self.LLM = LLM
        self.prompt_token_len, self.gen_token_len = self.LLM.get_token_length()
        self.args = args
        self.id2ent = pkl.load(open(os.path.join(args.data_path,"id2ent.pkl"), "rb"))
        self.id2rel = pkl.load(open(os.path.join(args.data_path, "id2rel.pkl"), "rb"))

    def clean_string(self, string):
        return clean_string(string)
    def compute_mrr(self, gt, pred):
        return compute_mrr_score(gt, pred)
    
    def search_KG(self, entities_with_scores): # {ent id: score}
        rel2answer = defaultdict(set) # dict. rel id: {(ent_id, ent_score, answer_id)}
        relations = set() # set of tuples, (rel id, rel name)
        for entity, score in entities_with_scores.items():
            for triple in self.entity_triplets[entity]:
                h, r, t = triple
                if entity != h: # 只看从ent出来的边
                    continue
                rel_name = self.id2rel[r]
                relations.add((r, rel_name))
                rel2answer[r].add((entity, score, t))
        return relations, rel2answer
     
        
    def get_prompt(self, relations, r):
        p = f"Please score the below {len(relations)} relations based on similarity to ({r}). Each score is in [0, 1], the sum of all scores is 1.\n"
        for rel_id, rel_name in relations:
            p += f"({rel_id}). ({rel_name})\n"
        p += "Answer only id and score of relation with no other text.\n"
        p += "Example input: '(1). (relation.name_1)\n(21). (relation.name_21)\n...'. Your example answer should be like: '(1). 0.8\n(21). 0.1\n...'"
        p += f"Your answer should contrain no more than {len(relations)} relations.\nYour answer is:"
        return p
    
    def format_llm_res(self, res:str):
        if res.startswith("Error"):
            print(res)

        pattern = r"\((\d+)\)\.\s*(\d*\.?\d*)"
        matches = re.findall(pattern, res) # [(rel id, rel score),...] e.g. [('1', '1.0'), ('2', '0.1'), ('3', '1')]

        relation_dict = {}
        #score_arr = np.array([score for _, score in matches]).astype(np.float32) #dtype: "<U3" -> np.float32
        
        score_list = []
        for _, score in matches:
            try:
                score_list.append(float(score))
            except ValueError:
                continue
        score_arr = np.array(score_list)
        
        if not matches or score_arr.sum()==0: 
            self.empty_cnt += 1
            return False, "No relations found"
        
        #score_arr = self.normalize(score_arr)
        for i in range(score_arr.size):
            if score_arr[i] > 0:
                relation_dict[int(matches[i][0])] = score_arr[i] # relation: normalized score
        
        return True, relation_dict
    
    def get_token_len(self):
        p, g = self.LLM.get_token_length()
        self.prompt_token_len = p - self.prompt_token_len
        self.gen_token_len = g - self.gen_token_len

    def rel_proj(self, vector, sub_question, relation): # 接受一个variable的fuzzy vec; 返回它经过rel proj 后的variable的fuzzy vec
        if vector is None:
            return None
        
        entities_with_scores = self.fuzzyVector_to_entities(vector) #{ent id: score}
        relations, rel2answer = self.search_KG(entities_with_scores) # dict. rel id: set of (ent_id, ent_score, answer_id)

        prompt = self.get_prompt(relations, relation)

        self.llm_cnt += 1

        start = time.time()
        result = self.LLM.run(prompt)
        end = time.time()
        self.llm_time += (end-start)
        
        flag, res = self.format_llm_res(result) # dict. rel id(int): rel_score(float)

        self.prompt = prompt
        self.result = result
        self.res = res

        if not flag:
            return None
        else:
            relations_with_scores = res
            ans_vec = np.zeros((self.ent_num))
            for rel_id in relations_with_scores:
                rel_score = relations_with_scores[rel_id]
                for answer in rel2answer[rel_id]:
                    _, ent_score, ans_id = answer
                    if self.score_rule == "sum":
                        ans_vec[ans_id] += ent_score * rel_score 
                    if self.score_rule == "max":
                        ans_vec[ans_id] = max(ans_vec[ans_id], ent_score * rel_score)    
            ans_vec = self.normalize(ans_vec)
            return ans_vec
        
    def union(self, v1, v2, v3 = None): 
        if v1 is None or v2 is None:
            #self.empty_cnt += 1
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
            #self.empty_cnt += 1
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
            #self.empty_cnt += 1
            return None
        return self.normalize(1-v)
    

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
                #return arr - arr.min()
                return arr 
        
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
            #return sigmoid(arr)
            return average_norm(min_max_norm(sigmoid(arr)))
        
        def softmax(arr):
            x = np.exp(arr - np.max(arr))
            if x.sum() > 0:
                return x / x.sum()
            return x
        
        if self.normalize_rule == "softmax":
            #return softmax(arr)
            return min_max_norm(softmax(arr))
        
        def l2_norm(arr):
            norm = np.linalg.norm(arr, ord=2)
            if norm == 0:
                return arr
            return arr / norm
        
        if self.normalize_rule == "l2_norm":
            return l2_norm(arr)
        
        
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

    def ansVector_to_ansFile(self, vector, output_file):
        if vector is not None:
            sorted_indices = np.argsort(vector)[::-1] #全体实体,分数从高到低
            preds = []
            score_sum = 0
            for i, id in enumerate(sorted_indices):
                #if score_sum >= self.prune * vector.sum() or i > self.ent_width:
                if score_sum >= self.prune * vector.sum():
                    break
                preds.append(id)
                score_sum += vector[id]
            
            if set(self.gt) != set(preds):
                mrr = self.compute_mrr(self.gt, preds)
                self.mrr = mrr
                self.pred = preds #list of id
                #self.pred_text = ", ".join([self.id2ent[x] for x in self.pred])
                if mrr < 1:
                    print("bk")
            
            pred = ", ".join(map(str, preds))
        else:
            self.ent_num += 1
            pred = ""

        with open(output_file, "w") as prediction_file:
                print(pred, file=prediction_file)


    def answer_query(self, logical_query, query_type, idx, output_path):
        e1 = r1 = e2 = r2= e3 = r3 = None
        self.q = logical_query 
        output_file = os.path.join(f"{output_path}",f"{query_type}_{idx}_predicted_answer.txt")

        gt_file = os.path.join(self.args.prefix_path, self.args.ground_truth_path, f"{query_type}_{idx}_answer.txt")
        with open(gt_file) as f:
            cleaned_gt = self.clean_string(f.read()).split(",")
            self.gt = [int(x) for x in cleaned_gt if x.isdigit()]
        #self.gt_text = ", ".join([self.id2ent[x] for x in self.gt])

        intermediate_variable = "intermediate_variable"

        if query_type=="1p": 
            (e1, (r1,)) = logical_query
            vector = np.zeros((self.ent_num))
            vector[e1] = 1
            question = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            ans_vec = self.rel_proj(vector=vector, sub_question=question, relation=self.id2rel[r1])
            self.ansVector_to_ansFile(vector=ans_vec, output_file=output_file)
            del vector, ans_vec

        if query_type=="2p": 
            (e1, (r1, r2)) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            v2 = self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            q2 = rel_proj_question % (intermediate_variable, self.id2rel[r2])
            va = self.rel_proj(vector=v2, sub_question=q2, relation=self.id2rel[r2]) # answer vector
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, va

        if query_type=="3p": 
            (e1, (r1, r2, r3)) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            v2 = self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            q2 = rel_proj_question % (intermediate_variable, self.id2rel[r2])
            v3 = self.rel_proj(vector=v2, sub_question=q2, relation=self.id2rel[r2]) # answer vector
            q3 = rel_proj_question % (intermediate_variable, self.id2rel[r3])
            va = self.rel_proj(vector=v3, sub_question=q3, relation=self.id2rel[r3])
            self.ansVector_to_ansFile(vector=va, output_file=output_file) 
            del v1, v2, v3, va            

        if query_type=="2i": 
            ((e1, (r1,)), (e2, (r2,))) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            q2 = rel_proj_question % (self.id2ent[e2], self.id2rel[r2])
            i1= self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            i2 = self.rel_proj(vector=v2, sub_question=q2, relation=self.id2rel[r2])
            va = self.intersection(i1, i2) # answer vector
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, i1, i2, va
            
        if query_type=="3i": 
            ((e1, (r1,)), (e2, (r2,)), (e3, (r3,))) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            v3 = np.zeros((self.ent_num))
            v3[e3] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            q2 = rel_proj_question % (self.id2ent[e2], self.id2rel[r2])
            q3 = rel_proj_question % (self.id2ent[e3], self.id2rel[r3])
            i1= self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            i2 = self.rel_proj(vector=v2, sub_question=q2, relation=self.id2rel[r2])
            i3 = self.rel_proj(vector=v3, sub_question=q3, relation=self.id2rel[r3])
            va = self.intersection(i1, i2, i3)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, v3, i1, i2, i3, va
        
        if query_type=="2in": 
            ((e1, (r1,)), (e2, (r2, n))) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            q2 = rel_proj_question % (self.id2ent[e2], self.id2rel[r2])            
            i1= self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            i2 = self.rel_proj(vector=v2, sub_question=q2, relation=self.id2rel[r2])
            i2 = self.negation(i2)
            va = self.intersection(i1, i2)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, i1, i2, va

        if query_type=="3in": 
            ((e1, (r1,)), (e2, (r2,)), (e3, (r3, n))) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            v3 = np.zeros((self.ent_num))
            v3[e3] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            q2 = rel_proj_question % (self.id2ent[e2], self.id2rel[r2])
            q3 = rel_proj_question % (self.id2ent[e3], self.id2rel[r3])
            i1= self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            i2 = self.rel_proj(vector=v2, sub_question=q2, relation=self.id2rel[r2])
            i3 = self.rel_proj(vector=v3, sub_question=q3, relation=self.id2rel[r3])
            i3 = self.negation(i3)
            va = self.intersection(i1, i2, i3)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, v3, i1, i2, i3, va

        if query_type=="inp":     
            (((e1, (r1,)), (e2, (r2, n))), (r3,)) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            q2 = rel_proj_question % (self.id2ent[e2], self.id2rel[r2])
            i1= self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            i2 = self.rel_proj(vector=v2, sub_question=q2, relation=self.id2rel[r2])
            i2 = self.negation(i2)
            v3 = self.intersection(i1, i2)
            q3 = rel_proj_question % (intermediate_variable, self.id2rel[r3])
            va = self.rel_proj(vector=v3, sub_question=q3, relation=self.id2rel[r3])
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, i1, i2, v3, va

        if query_type=="pin": 
            ((e1, (r1, r2)), (e2, (r3, n))) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            v11 = self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            q2 = rel_proj_question % (intermediate_variable, self.id2rel[r2])
            i1 = self.rel_proj(vector=v11, sub_question=q2, relation=self.id2rel[r2]) 
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            q3 = rel_proj_question % (self.id2ent[e2], self.id2rel[r3])
            i2 = self.rel_proj(vector=v2, sub_question=q3, relation=self.id2rel[r3])
            i2 = self.negation(i2)
            va = self.intersection(i1, i2)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v11, i1, v2, i2, va

        if query_type=="pni": 
            ((e1, (r1, r2, n)), (e2, (r3,))) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            v11 = self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            q2 = rel_proj_question % (intermediate_variable, self.id2rel[r2])
            i1 = self.rel_proj(vector=v11, sub_question=q2, relation=self.id2rel[r2]) 
            i1 = self.negation(i1)
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            q3 = rel_proj_question % (self.id2ent[e2], self.id2rel[r3])
            i2 = self.rel_proj(vector=v2, sub_question=q3, relation=self.id2rel[r3])
            va = self.intersection(i1, i2)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v11, i1, v2, i2, va

        if query_type=="ip": 
            (((e1, (r1,)), (e2, (r2,))), (r3,)) = logical_query
            v1 = np.zeros((self.ent_num))
            v2 = np.zeros((self.ent_num))
            v1[e1] = 1
            v2[e2] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            q2 = rel_proj_question % (self.id2ent[e2], self.id2rel[r2])
            i1 = self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            i2 = self.rel_proj(vector=v2, sub_question=q2, relation=self.id2rel[r2])
            v3 = self.intersection(i1, i2)
            q3 = rel_proj_question % (intermediate_variable, self.id2rel[r3])
            va = self.rel_proj(vector=v3, sub_question=q3, relation=self.id2rel[r3])
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, i1, i2, v3, va

        if query_type=="pi": 
            ((e1, (r1, r2)), (e2, (r3,))) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            v11 = self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            q2 = rel_proj_question % (intermediate_variable, self.id2rel[r2])
            i1 = self.rel_proj(vector=v11, sub_question=q2, relation=self.id2rel[r2]) 
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            q3 = rel_proj_question % (self.id2ent[e2], self.id2rel[r3])
            i2 = self.rel_proj(vector=v2, sub_question=q3, relation=self.id2rel[r3])
            va = self.intersection(i1, i2)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v11, i1, v2, i2, va

        if query_type=="2u": 
            ((e1, (r1,)), (e2, (r2,)), (u,)) = logical_query
            v1 = np.zeros((self.ent_num))
            v2 = np.zeros((self.ent_num))
            v1[e1] = 1
            v2[e2] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            q2 = rel_proj_question % (self.id2ent[e2], self.id2rel[r2])
            i1 = self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            i2 = self.rel_proj(vector=v2, sub_question=q2, relation=self.id2rel[r2])
            va = self.union(i1, i2)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, i1, i2, va

        if query_type=="up": 
            (((e1, (r1,)), (e2, (r2,)), (u,)), (r3,)) = logical_query
            v1 = np.zeros((self.ent_num))
            v2 = np.zeros((self.ent_num))
            v1[e1] = 1
            v2[e2] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            q2 = rel_proj_question % (self.id2ent[e2], self.id2rel[r2])
            i1 = self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            i2 = self.rel_proj(vector=v2, sub_question=q2, relation=self.id2rel[r2])
            v3 = self.union(i1, i2)
            q3 = rel_proj_question % (intermediate_variable, self.id2rel[r3])
            va = self.rel_proj(vector=v3, sub_question=q3, relation=self.id2rel[r3])
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, i1, i2, v3, va

        if query_type=="nin": 
            (((e1, (r1, n)), (e2, (r2, n))), (n,)) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            i1 = self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            i1 = self.negation(i1)
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            q2 = rel_proj_question % (self.id2ent[e2], self.id2rel[r2])
            i2 = self.rel_proj(vector=v2, sub_question=q2, relation=self.id2rel[r2])
            i2 = self.negation(i2)
            va = self.intersection(i1, i2)
            va = self.negation(va)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, i1, v2, i2, va

        if query_type=="nipn": 
            (((e1, (r1, n)), (e2, (r2, n))), (n, r3)) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            i1 = self.rel_proj(vector=v1, sub_question=q1, relation=self.id2rel[r1])
            i1 = self.negation(i1)
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            q2 = rel_proj_question % (self.id2ent[e2], self.id2rel[r2])
            i2 = self.rel_proj(vector=v2, sub_question=q2, relation=self.id2rel[r2])
            i2 = self.negation(i2)
            v3 = self.intersection(i1, i2)
            v3 = self.negation(v3)
            q3 = rel_proj_question % (intermediate_variable, self.id2rel[r3])
            va = self.rel_proj(vector=v3, sub_question=q3, relation=self.id2rel[r3])
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, i1, v2, i2, v3, va       




