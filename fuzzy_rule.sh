cd ~/liuxy/llmReason/code

python Infer_ans_main.py --sub_path lukas --fuzzy_rule lukas --qtype up
python compute_scores.py --prediction_path ../data/NELL-betae/processed/lukas --score_file up_lukas.txt --qtype up

python Infer_ans_main.py --sub_path prod --fuzzy_rule prod --qtype up
python compute_scores.py --prediction_path ../data/NELL-betae/processed/prod --score_file up_prod.txt --qtype up

python Infer_ans_main.py --sub_path lukas --fuzzy_rule lukas --qtype ip
python compute_scores.py --prediction_path ../data/NELL-betae/processed/lukas --score_file ip_lukas.txt --qtype ip

python Infer_ans_main.py --sub_path prod --fuzzy_rule prod --qtype ip
python compute_scores.py --prediction_path ../data/NELL-betae/processed/lukas --score_file ip_prod.txt --qtype ip
