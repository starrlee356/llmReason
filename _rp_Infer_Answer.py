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
from _relproj import RelProj
from _rp_utils import *

"""prompt, 让LLM输出更短"""

class infer_and_answer:
    def __init__(self, entity_triplets_file, id2ent_file, id2rel_file, stats_file, rel_width, ent_width,
                  fuzzy_rule, model_name, prune, score_rule, normalize_rule, LLM, args, Retriever):
        self.entity_triplets = pkl.load(open(entity_triplets_file,"rb"))
        self.id2ent = pkl.load(open(id2ent_file,"rb"))
        self.id2rel = pkl.load(open(id2rel_file,"rb"))
        self.rel2path = pkl.load(open(os.path.join(args.prefix_path, "llmR", "rel2path.pkl"), "rb"))
        self.rel2id = pkl.load(open(os.path.join(args.data_path, "rel2id.pkl"), "rb"))

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

        self.rp_cnt = 0 
        self.llm_time = 0
        self.LLM = LLM

        self.prompt_token_len, self.gen_token_len = self.LLM.get_token_length()

        self.args = args
        self.Retriever = Retriever

        self.search_time = 0
        self.DR_time = 0

        #for db
        self.hit_rate = 0
        self.hit_rate_wo_prune = 0
        self.hit_rate_prune_tail = 0
        self.empty_inter_id = 0
        self.all_path_top_score = 0
        self.DR_path_top_score = 0
        self.DR_loader_len = 0

        self.mrr_wo_llm = 0
        self.mrr_w_llm = 0
        self.topK_path_hit_ans_rate = 0
        self.llm_res_hit_ans_rate = 0
        self.tails_size = 0

        self.fscore1 = 0
        self.fscore2 = 0
        self.fscore5 = 0


    def clean_string(self, string):
        return clean_string(string)
    def compute_mrr(self, gt, pred):
        return compute_mrr_score(gt, pred)

    def get_token_len(self):
        p, g = self.LLM.get_token_length()
        self.prompt_token_len = p - self.prompt_token_len
        self.gen_token_len = g - self.gen_token_len


        
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
        return normalize(arr, self.normalize_rule)
        
        
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
                arr_score = np.array(scores)
                #arr_score = normalize(arr_score, self.normalize_rule)
                result = {sorted_indices[i]: arr_score[i] for i in range(len(scores))} #ent id: norm score
            else:
                result = {} #empty
        
            return result


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
                #if mrr < 1:
                    #print("bk")
            
            pred = ", ".join(map(str, preds))
        else:
            self.ent_num += 1
            pred = ""

        with open(output_file, "w") as prediction_file:
                print(pred, file=prediction_file)

    def rel_proj(self, vector, sub_question, relation):#vector=vector, sub_question=question, relation=self.id2rel[r1]
        entities_with_score = self.fuzzyVector_to_entities(vector)
        rid = self.rel2id[relation]
        #self, args, LLM, entities_with_score, r_name, ent_num, ent_triplets, id2rel, DR; *ans only for 1p*
        RP = RelProj(self.args, self.LLM, entities_with_score, relation, self.ent_num, self.entity_triplets, self.id2rel, self.Retriever, self.gt, self.id2ent, sub_question)
        tail_vector = RP.run()
        self.rp_cnt += 1
        self.llm_time += RP.llm_time
        self.empty_cnt += RP.empty_cnt
        self.search_time += RP.search_time
        self.DR_time += RP.DR_time
        #for debug
        self.mrr_wo_llm += RP.mrr_wo_llm
        self.mrr_w_llm += RP.mrr_w_llm
        print(f"avg mrr_w_llm = {self.mrr_w_llm/self.rp_cnt}, avg mrr_wo_llm = {self.mrr_wo_llm/self.rp_cnt}")
        self.llm_res_hit_ans_rate += RP.llm_res_hit_ans_rate
        self.topK_path_hit_ans_rate += RP.topK_path_hit_ans_rate
        self.tails_size += RP.tails_size
        self.fscore1 += RP.fscore1
        self.fscore2 += RP.fscore1
        self.fscore5 += RP.fscore5

        self.hit_rate += RP.hit_rate
        self.hit_rate_wo_prune += RP.hit_rate_wo_prune
        self.hit_rate_prune_tail += RP.hit_rate_prune_tail
        self.empty_inter_id += RP.empty_inter_id
        self.all_path_top_score += RP.all_path_top_score
        self.DR_path_top_score += RP.DR_path_top_score
        self.DR_loader_len += RP.DR_loader_len

        return tail_vector

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




