from prompt_list import *
from global_config import *
import numpy as np
import pickle as pkl
from collections import defaultdict
import os
import ollama
import re

class infer_and_answer:
    def __init__(self, entity_triplets_file, id2ent_file, id2rel_file, stats_file, rel_width, ent_width, fuzzy_rule, model_name):
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
        self.llm = model_name

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
        content = ollama.generate(model=self.llm, prompt=prompt)
        response = content["response"]
        return response
    
    def format_llm_res(self, res:str):
        if res.startswith("Error"):
            print(res)
            # 其他错误处理代码

        pattern = r'\d+\.\s*(\S+)\s*\(Score:\s*([0-1](?:\.\d+)?)\)'
        matches = re.findall(pattern, res)
        relation_dict = {relation: float(score) for relation, score in matches}
        if not relation_dict:
            return False, "No relations found"
        return True, relation_dict
        
        
    def normalize(self, arr):
        total = np.sum(arr)
        if total == 0:
            return arr  
        return arr / total

    def rel_proj(self, vector, sub_question): # 接受一个variable的fuzzy vec; 返回它经过rel proj 后的variable的fuzzy vec
        entities_with_scores = self.fuzzyVector_to_entities(vector) #{ent id: score}
        relations, rel2answer = self.search_KG(entities_with_scores) # dict. rel name: {(ent_id, ent_score, answer_id)}
        prompt = extract_relation_prompt % (self.rel_width, self.rel_width, self.rel_width, sub_question, '; '.join(relations))
        result = self.run_llm(prompt) 
        flag, res = self.format_llm_res(result) # dict. rel name: rel_score
        if not flag:
            print(res)
        else:
            relations_with_scores = res
            ans_vec = np.zeros((self.ent_num))
            for rel in relations_with_scores:
                rel_score = relations_with_scores[rel]
                for answer in rel2answer[rel]:
                    _, ent_score, ans_id = answer
                    ans_vec[ans_id] += ent_score * rel_score # 原ent_score应该加起来==1表示概率; rel_score要求llm产生加和==1的分数。或者用其他的概率计算方法？
            ans_vec = self.normalize(ans_vec)
            return ans_vec

    def union(self, v1, v2, v3 = None): 
        if v3 == None:
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
        if v3 == None:
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
        return self.normalize(1-v)

    def fuzzyVector_to_entities(self, vector): # from fuzzy vec to answer entity
        try:
            indices = np.argpartition(vector, -self.ent_width)[-self.ent_width:]  # 获取前k大的元素的下标
        except ValueError:
            print("breakpoint")
        top_k_values = vector[indices]  # 获取前k大的元素

        # 对前k个元素进行归一化
        top_k_sum = np.sum(top_k_values)
        if top_k_sum != 0:
            normalized_values = top_k_values / top_k_sum
        else:
            normalized_values = np.zeros(self.ent_width)  # 如果总和为0，返回全0数组
        
        # 构建字典，key为下标，value为归一化后的值
        result = {indices[i]: normalized_values[i] for i in range(self.ent_width)}
        
        return result
    
    def ansVector_to_ansFile(self, vector, output_file):
        entities_with_scores = self.fuzzyVector_to_entities(vector)
        pred = ", ".join(map(str, entities_with_scores.keys()))
        with open(output_file,"w") as prediction_file:
                print(pred, file=prediction_file)

    def answer_query(self, logical_query, query_type, idx, output_path):
        e1 = r1 = e2 = r2= e3 = r3 = None
        output_file = os.path.join(f"{output_path}",f"{query_type}_{idx}_predicted_answer.txt")
        intermediate_variable = "intermediate_variable"

        # 更倾向于让1p~3p按照原llmReason的做法？ 
        if query_type=="1p": 
            (e1, (r1,)) = logical_query
            vector = np.zeros((self.ent_num))
            vector[e1] = 1
            question = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            ans_vec = self.rel_proj(vector=vector, sub_question=question)
            self.ansVector_to_ansFile(vector=ans_vec, output_file=output_file)
            del vector, ans_vec

        if query_type=="2p": 
            (e1, (r1, r2)) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            v2 = self.rel_proj(vector=v1, sub_question=q1)
            q2 = rel_proj_question % (intermediate_variable, self.id2rel[r2])
            va = self.rel_proj(vector=v2, sub_question=q2) # answer vector
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, va

        if query_type=="3p": 
            (e1, (r1, r2, r3)) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            v2 = self.rel_proj(vector=v1, sub_question=q1)
            q2 = rel_proj_question % (intermediate_variable, self.id2rel[r2])
            v3 = self.rel_proj(vector=v2, sub_question=q2) # answer vector
            q3 = rel_proj_question % (intermediate_variable, self.id2rel[r3])
            va = self.rel_proj(vector=v3, sub_question=q3)
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
            i1= self.rel_proj(vector=v1, sub_question=q1)
            i2 = self.rel_proj(vector=v2, sub_question=q2)
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
            i1= self.rel_proj(vector=v1, sub_question=q1)
            i2 = self.rel_proj(vector=v2, sub_question=q2)
            i3 = self.rel_proj(vector=v3, sub_question=q3)
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
            i1= self.rel_proj(vector=v1, sub_question=q1)
            i2 = self.rel_proj(vector=v2, sub_question=q2)
            i2 = self.negation(i2)
            va = self.intersection(i1, i2)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, i1, i2, va

        if query_type=="3in": 
            ((e1, (r1,)), (e2, (r2,)), (e3, (r3, n))) = logical_query
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
            i1= self.rel_proj(vector=v1, sub_question=q1)
            i2 = self.rel_proj(vector=v2, sub_question=q2)
            i3 = self.rel_proj(vector=v3, sub_question=q3)
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
            i1= self.rel_proj(vector=v1, sub_question=q1)
            i2 = self.rel_proj(vector=v2, sub_question=q2)
            i2 = self.negation(i2)
            v3 = self.intersection(i1, i2)
            q3 = rel_proj_question % (intermediate_variable, self.id2rel[r3])
            va = self.rel_proj(vector=v3, sub_question=q3)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, i1, i2, v3, va

        if query_type=="pin": 
            ((e1, (r1, r2)), (e2, (r3, n))) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            v11 = self.rel_proj(vector=v1, sub_question=q1)
            q2 = rel_proj_question % (intermediate_variable, self.id2rel[r2])
            i1 = self.rel_proj(vector=v11, sub_question=q2) 
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            q3 = rel_proj_question % (self.id2ent[e2], self.id2rel[r3])
            i2 = self.rel_proj(vector=v2, sub_question=q3)
            i2 = self.negation(i2)
            va = self.intersection(i1, i2)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v11, i1, v2, i2, va

        if query_type=="pni": 
            ((e1, (r1, r2, n)), (e2, (r3,))) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            v11 = self.rel_proj(vector=v1, sub_question=q1)
            q2 = rel_proj_question % (intermediate_variable, self.id2rel[r2])
            i1 = self.rel_proj(vector=v11, sub_question=q2) 
            i1 = self.negation(i1)
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            q3 = rel_proj_question % (self.id2ent[e2], self.id2rel[r3])
            i2 = self.rel_proj(vector=v2, sub_question=q3)
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
            i1 = self.rel_proj(vector=v1, sub_question=q1)
            i2 = self.rel_proj(vector=v2, sub_question=q2)
            v3 = self.intersection(i1, i2)
            q3 = rel_proj_question % (intermediate_variable, self.id2rel[r3])
            va = self.rel_proj(vector=v3, sub_question=q3)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, i1, i2, v3, va

        if query_type=="pi": 
            ((e1, (r1, r2)), (e2, (r3,))) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            v11 = self.rel_proj(vector=v1, sub_question=q1)
            q2 = rel_proj_question % (intermediate_variable, self.id2rel[r2])
            i1 = self.rel_proj(vector=v11, sub_question=q2) 
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            q3 = rel_proj_question % (self.id2ent[e2], self.id2rel[r3])
            i2 = self.rel_proj(vector=v2, sub_question=q3)
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
            i1 = self.rel_proj(vector=v1, sub_question=q1)
            i2 = self.rel_proj(vector=v2, sub_question=q2)
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
            i1 = self.rel_proj(vector=v1, sub_question=q1)
            i2 = self.rel_proj(vector=v2, sub_question=q2)
            v3 = self.union(i1, i2)
            q3 = rel_proj_question % (intermediate_variable, self.id2rel[r3])
            va = self.rel_proj(vector=v3, sub_question=q3)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, v2, i1, i2, v3, va

        if query_type=="nin": 
            (((e1, (r1, n)), (e2, (r2, n))), (n,)) = logical_query
            v1 = np.zeros((self.ent_num))
            v1[e1] = 1
            q1 = rel_proj_question % (self.id2ent[e1], self.id2rel[r1])
            i1 = self.rel_proj(vector=v1, sub_question=q1)
            i1 = self.negation(i1)
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            q2 = rel_proj_question % (self.id2ent[e2], self.id2rel[r2])
            i2 = self.rel_proj(vector=v2, sub_question=q2)
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
            i1 = self.rel_proj(vector=v1, sub_question=q1)
            i1 = self.negation(i1)
            v2 = np.zeros((self.ent_num))
            v2[e2] = 1
            q2 = rel_proj_question % (self.id2ent[e2], self.id2rel[r2])
            i2 = self.rel_proj(vector=v2, sub_question=q2)
            i2 = self.negation(i2)
            v3 = self.intersection(i1, i2)
            v3 = self.negation(v3)
            q3 = rel_proj_question % (intermediate_variable, self.id2rel[r3])
            va = self.rel_proj(vector=v3, sub_question=q3)
            self.ansVector_to_ansFile(vector=va, output_file=output_file)
            del v1, i1, v2, i2, v3, va       
