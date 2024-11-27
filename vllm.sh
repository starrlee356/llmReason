export VLLM_USE_MODELSCOPE=True


export CUDA_VISIBLE_DEVICES=4,5,6,7

"""
python -m vllm.entrypoints.openai.api_server \
    --model ~/.cache/modelscope/hub/llama/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 \
    --dtype half \
    --max-model-len 12256 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --served-model-name llama3:8b \
"""

python -m vllm.entrypoints.openai.api_server \
    --model ~/.cache/modelscope/hub/llama/Meta-Llama-3-8B-Instruct \
    --dtype half \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --served-model-name llama3:8b \
   
