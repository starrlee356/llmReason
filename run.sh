cd ~/liuxy/llmReason/code

python Infer_ans_main.py --qtype 3p
python compute_scores.py --score_file 3p.txt --qtype 3p

python Infer_ans_main.py --qtype pi
python compute_scores.py --score_file pi.txt --qtype pi

python Infer_ans_main.py --qtype pni
python compute_scores.py --score_file pni.txt --qtype pni

python Infer_ans_main.py --qtype 2in
python compute_scores.py --score_file 2in.txt --qtype 2in

python Infer_ans_main.py --qtype 3i
python compute_scores.py --score_file 3i.txt --qtype 3i

python Infer_ans_main.py --qtype 3in
python compute_scores.py --score_file 3in.txt --qtype 3in

python Infer_ans_main.py --qtype inp
python compute_scores.py --score_file inp.txt --qtype inp

python Infer_ans_main.py --qtype pni
python compute_scores.py --score_file pni.txt --qtype pni

python Infer_ans_main.py --qtype nipn
python compute_scores.py --score_file nipn.txt --qtype nipn
