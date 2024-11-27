import os
import csv
from tqdm import tqdm
import pickle as pkl
import argparse
import multiprocessing as mp
import random
import json
import time
import hydra
import logging
import torch
from pathlib import Path
import sys
root = str(Path.cwd().parent)
DPR_path = os.path.join(root, "DPR")
score_path = os.path.join(root, "path_score")
if root not in sys.path:
    sys.path.append(root)
if DPR_path not in sys.path:
    sys.path.append(DPR_path)
if score_path not in sys.path:
    sys.path.append(score_path)

from DPR.generate_dense_embeddings import DR
from path_score.path_scorer import PathScorer

from _rp_Infer_Answer import infer_and_answer
from compute_scores import compute_score_main
from llm import LLM_vllm, LLM_ollama, LLM_zhipu, LLM_gpt

logging.getLogger("httpx").setLevel(logging.WARNING)
#logging.basicConfig(level=logging.INFO)

def mergeDictionary(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = value.union(dict_1[key])
   return dict_3

from pathlib import Path
from omegaconf import DictConfig
config_path = os.path.join(str(Path.cwd().parent), "DPR", "conf")
config_name = "gen_embs"
@hydra.main(config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):    
    start = time.time()
    #Retriever = DR(cfg)
    Retriever = PathScorer(args.device, args.encoder_model)
    ckpt_file = args.ckpt_file % args.ckpt_ep
    ckpt = torch.load(ckpt_file)
    Retriever.load_state_dict(ckpt)
    Retriever.eval()

    entity_triplets_file = os.path.join(args.prefix_path, args.triplets_file)
    id2ent_file = os.path.join(args.data_path, "id2ent.pkl")
    id2rel_file = os.path.join(args.data_path, "id2rel.pkl")
    stats_file = os.path.join(args.data_path, "stats.txt")
    #queries = pkl.load(open(os.path.join(args.data_path, args.queries_path), "rb"))

    pred_path = os.path.join(args.prefix_path, args.prediction_path)
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    
    # llm = LLM(model_name=args.llm_name)
    if args.api == "ollama":
        llm_obj = LLM_ollama(model=args.model_name)
    if args.api == "vllm":
        llm_obj = LLM_vllm(model=args.model_name)
    if args.api == "zhipu":
        llm_obj = LLM_zhipu(model=args.model_name)
    if args.api == "gpt":
        llm_obj = LLM_gpt(model=args.model_name)
    if args.api == "none":
        llm_obj = None

    model = infer_and_answer(entity_triplets_file, id2ent_file, id2rel_file, stats_file, args.rel_width, 
                             args.ent_width, args.fuzzy_rule, args.model_name, args.prune, args.score_rule, 
                             args.normalize_rule, llm_obj, args, Retriever)
    idx_to_queries = pkl.load(open(os.path.join(args.prefix_path, args.id2q_file), "rb"))
    qtype2cnt = pkl.load(open(os.path.join(args.prefix_path, args.qtype2cnt_file), "rb"))
    
    def answer(qtype, qpattern):
        idx_list = random.sample(range(qtype2cnt[qtype]), args.random_num)
        random_path = os.path.join(args.prefix_path, args.random_path)
        if not os.path.exists(random_path):
            os.makedirs(random_path)
        with open(os.path.join(random_path, f"{qtype}.json"), "w") as f:
            json.dump(idx_list, f)
        
        for idx in tqdm(idx_list, desc=f"{qtype}"):
            query = idx_to_queries[qtype][idx]
            model.answer_query(logical_query=query, query_type=qtype, idx=idx, output_path=pred_path)
        
        model.get_token_len()

    if args.qtype == "all":
        for qtype, qpattern in model.q_structs.items():
            answer(qtype=qtype, qpattern=qpattern)
    else:
        for qtype, qpattern in model.q_structs.items():
            if qtype == args.qtype:
                answer(qtype=qtype, qpattern=qpattern)

    end = time.time()
    info = f"random {args.random_num} q\n"
    info += f"avg prompt token len = {model.prompt_token_len/model.rp_cnt}\n"
    info += f"avg gen token len = {model.gen_token_len/model.rp_cnt}\n"
    info += f"llm response no match cnt = {model.empty_cnt}\n"
    info += f"llm time = {model.llm_time} s\n"
    info += f"search G time = {model.search_time} s\n"
    info += f"DR time = {model.DR_time} s\n"
    info += f"total time = {end-start} s\n"
    
    #for db
    info += f"mrr wo llm = {model.mrr_wo_llm / model.rp_cnt}\n"
    info += f"topK_path_hit_ans_rate = {model.topK_path_hit_ans_rate / model.rp_cnt}\n"
    info += f"mrr_w_llm = {model.mrr_w_llm / model.rp_cnt}\n"
    info += f"llm_res_hit_ans_rate = {model.llm_res_hit_ans_rate / model.rp_cnt}\n"
    info += f"tails_size = {model.tails_size / model.rp_cnt}\n"
    info += f"fuse score 1 = {model.fscore1 / model.rp_cnt}\n"
    info += f"fuse score 2 = {model.fscore2 / model.rp_cnt}\n"
    info += f"fuse score 5 = {model.fscore5 / model.rp_cnt}\n"

    info += "==============================\n"
    info += f"all path top fuse score = {model.all_path_top_score / model.rp_cnt}\n"
    info += f"DR path top fuse score = {model.DR_path_top_score / model.rp_cnt}\n"
    info += f"DR loader len = {model.DR_loader_len / model.rp_cnt}, bsz = {args.DR_bsz}\n"
    info += f"hit rate = {model.hit_rate / model.rp_cnt}\n"
    info += f"hit rate wo prune = {model.hit_rate_wo_prune / model.rp_cnt}\n"
    info += f"hit rate prune tail = {model.hit_rate_prune_tail / model.rp_cnt}\n"
    info += f"empty inter id = {model.empty_inter_id / model.rp_cnt}\n"
    
    
    
    print(info)
    compute_score_main(args, info)

    


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="../data/NELL-betae", help="Path to raw data.")
    parser.add_argument('--prefix_path', type=str, default="../data/NELL-betae/processed", help="Path to output the processed files.")

    parser.add_argument('--rel_width', type=int, default=50, help="Ask LLM to retrieve top rel_width relations each time.")
    parser.add_argument('--ent_width', type=int, default=25, help="Retrieve top ent_width entities from fuzzy vector.")
    parser.add_argument('--tail_width', type=int, default=5, help="LLM prompt, for each path, only keep tail_width tails.")
    parser.add_argument('--prune', type=float, default=0.9, help="if entities score add up to {prune} then omit the rest.")
    parser.add_argument('--fuzzy_rule', type=str, default="min_max", help="min_max/ prod/ lukas.")
    parser.add_argument('--score_rule', type=str, default="max", help="the rule when rel proj. max/sum.") #origin: sum
    parser.add_argument('--normalize_rule', type=str, default="standard_norm", help="choose a normalize function. min_max_norm/standard_norm/l2_norm/sigmoid/softmax.")
    parser.add_argument("--random_size", type=int, default=50, help="randomly select k queries for each qtype. if size=0 then use the whole dataset.")
    parser.add_argument("--whole_size", type=int, default=0, help="randomly select k queries for each qtype. if size=0 then use the whole dataset.")
    
    parser.add_argument('--ground_truth_path', type=str, default="sorted_answers", help="Path to ground truth data.")
    parser.add_argument('--prediction_path', type=str, default="preds", help="Path to the prediction files.")
    parser.add_argument('--log_score_path', type=str, default="scores", help="Path to log scores")
    parser.add_argument('--random_path', type=str, default="llmR/random_list")
    parser.add_argument('--random_num', type=int, default=1000)
    
    parser.add_argument('--quries_file', type=str, default="test-queries.pkl")
    parser.add_argument('--triplets_file', type=str, default="llmR/train-entity-triplets.pkl")
    parser.add_argument('--qtype2cnt_file', type=str, default="test-qtype2cnt.pkl")
    parser.add_argument('--id2q_file', type=str, default="test-id2q.pkl")

    parser.add_argument('--qtype', type=str, default="2p")
    parser.add_argument('--score_file', type=str, default="2p.txt", help="file name to log scores")
    parser.add_argument('--api', type=str, default="vllm", help="ollama/vllm/zhipu")
    parser.add_argument('--model_name', type=str, default="llama3:8b", help="llama3:8b/glm-4;glm-4-flash")
    
    parser.add_argument('--depth', type=int, default=3, help="search k-hop sub graph for LLM prompt.")
    parser.add_argument('--path_width', type=int, default=50, help="select top k paths to prompt LLM")
    parser.add_argument('--prompt_width', type=int, default=20, help="tell LLM to score no more than k paths.")

    parser.add_argument('--device', type=str, default="cuda:1", help="scorer run(infer) on this device")
    parser.add_argument('--encoder_model', type=str, default="FacebookAI/roberta-base")
    parser.add_argument('--ckpt_file', type=str, default="../path_score/output/2024-11-18-14-43/encoder_ep%s.ckpt")
    parser.add_argument('--ckpt_ep', type=int, default=15)
    parser.add_argument('--DR_bsz', type=int, default=100, help="retriever infer batch size")

    

    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown

    for dir_name in [args.prediction_path, args.log_score_path, args.random_path]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    main()

