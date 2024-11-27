import numpy as np


def normalize(arr, rule):

    def average_norm(arr):
        total = np.sum(arr)
        if total == 0:
            return arr  
        return arr / total
    
    if rule == "average_norm":
        return average_norm(arr)
    
    def min_max_norm(arr):
        if arr.max() - arr.min() > 0:
            return (arr - arr.min()) / (arr.max() - arr.min())
        else:
            #return arr - arr.min()
            return arr
    
    if rule == "min_max_norm":
        return min_max_norm(arr)
    
    def standard_norm(arr):
        if np.std(arr) == 0:
                return arr - np.mean(arr)
        return (arr - np.mean(arr)) / np.std(arr)
    
    if rule == "standard_norm":
        return standard_norm(arr)
    
    def sigmoid(arr):
        return 1 / (1 + np.exp(-arr))
    
    if rule == "sigmoid":
        #return sigmoid(arr)
        return average_norm(min_max_norm(sigmoid(arr)))
    
    def softmax(arr):
        x = np.exp(arr - np.max(arr))
        if x.sum() > 0:
            return x / x.sum()
        return x
    
    if rule == "softmax":
        #return softmax(arr)
        return min_max_norm(softmax(arr))
    
    def l2_norm(arr):
        norm = np.linalg.norm(arr, ord=2)
        if norm == 0:
            return arr
        return arr / norm
    
    if rule == "l2_norm":
        return l2_norm(arr)
    

def _rp_prompt(path_width, r_name, cands_str):
    prompt = f"""
You'll be given some candidate paths and a true path.
Each path is represented as a sequence (r1, r2, ..., rn), which indicates a series of relation projections r1~rn in order.
Starting from the same head entity, a candidate path and the true path will yield to tail entity set S_cand and S_true respectively. 
The higher the overlap between S_cand and S_true, the higher the score you should assign to that candidate path.

Here is an example of input and answer: 
score the following 4 candidate paths based on the true path (concept:agentcompeteswithagent). 
path1: (concept:agentcompeteswithagent,);
path2: (concept:agentcompeteswithagent_reverse, concept:agentcompeteswithagent, concept:agentcompeteswithagent);
path3: (concept:agentparticipatedinevent, concept:agentparticipatedinevent_reverse, concept:teamplaysincity);
path4: (concept:agentcompeteswithagent_reverse, concept:agentcompeteswithagent_reverse, concept:organizationdissolvedatdate);
The answer is: "[1.0, 0.8, 0.2, 0.1]"

Please score the following {path_width} candidate paths based on the true path ({r_name}) on a scale from 0 to 1, and return a score list of length {path_width}. 
The score index and path index should be aligned. Return only the score list with no other text.
{cands_str}
Your answer is:
"""
    return prompt


def path_tail_prompt(question, cand_str, width):
    p = f"""
    You'll be given a question and some candidate paths with its tails. You need to score those tails based on their likelihood to be the answer.

Here is an example of input and answer: 
question: which entities are connected to concept_visualartist_caravaggio by relation concept:visualartistartform?
candidate paths with tails:
path0: (concept:visualartistartform,); tail0: (concept_visualartform_art); tail1: (concept_visualizablething_painting); tail2: (concept_visualartform_collection);
path1: (concept:visualartistartform, concept:visualartistartform_reverse, concept:visualartistartform); tail0: (concept_visualartform_paintings); tail1: (concept_hobby_colors);
path2: (concept:visualartistartform, concept:atdate, concept:atdate_reverse); tail0: (concept_attraction_lords); tail2: (concept_charactertrait_opening);

First, We can see that path0=(concept:visualartistartform,) and path1=(concept:visualartistartform, concept:visualartistartform_reverse, concept:visualartistartform) are likely to represent relation concept:visualartistartform.
Then, path0-tail0=(concept_visualartform_art), path0-tail1=(concept_visualizablething_painting), path0-tail2=(concept_visualartform_collection) and path1-tail0=(concept_visualartform_paintings) are likely to be the answers to the question. 
Thus, your answer is: "path0-tail0: 0.9; path0-tail1: 0.9; path0-tail2: 0.8; path1-tail0: 0.8"

question: {question}
candidate paths with tails:
{cand_str}
Your answer should contain no more than {width} items, and each item is like "pathID-tailID: score". Each score is on a scale from 0 to 1.  
Return only those items with no other text. Your answer is:
"""
    return p


