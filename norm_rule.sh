cd ~/liuxy/llmReason/code
"""
python Infer_ans_main.py --sub_path norm_std --qtype ip --score_rule max --normalize_rule standard_norm
python compute_scores.py --prediction_path ../data/NELL-betae/processed/norm_std --score_file norm_std_ip.txt --qtype ip

python Infer_ans_main.py --sub_path norm_l2 --qtype ip --score_rule max --normalize_rule l2_norm
python compute_scores.py --prediction_path ../data/NELL-betae/processed/norm_l2 --score_file norm_l2_ip.txt --qtype ip

python Infer_ans_main.py --sub_path norm_sigmoid --qtype ip --score_rule max --normalize_rule sigmoid
python compute_scores.py --prediction_path ../data/NELL-betae/processed/norm_sigmoid --score_file norm_sigmoid_ip.txt --qtype ip
"""
python Infer_ans_main.py --sub_path norm_std --qtype 2u --score_rule max --normalize_rule standard_norm
python compute_scores.py --prediction_path ../data/NELL-betae/processed/norm_std --score_file norm_std_2u.txt --qtype 2u

python Infer_ans_main.py --sub_path norm_std --qtype up --score_rule max --normalize_rule standard_norm
python compute_scores.py --prediction_path ../data/NELL-betae/processed/norm_std --score_file norm_std_up.txt --qtype up
