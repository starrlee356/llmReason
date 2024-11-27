python -m pdb _RAG_prepare_dataset.py \
 --prefix_path ../data/NELL0/processed \
 --data_path ../data/NELL0 \
 --trainG_file llmR/train-headent-triplets.pkl \
 --id2q_file test-id2q.pkl \
 --pos_num 5 \
 --neg_num 5

