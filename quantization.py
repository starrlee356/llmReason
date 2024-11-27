from transformers import AutoTokenizer
from GPTQModel import GPTQModel as AutoGPTQForCausalLM
from GPTQModel import QuantizeConfig as BaseQuantizeConfig


model_path = "/raid/home/specter/.cache/modelscope/hub/llama/Meta-Llama-3-8B-Instruct"
quantized_path = "/raid/home/specter/.cache/modelscope/hub/llama/Llama3-8B-4bitGPTQ"

"""
from transformers import AutoTokenizer
from auto_gptq import GPTQQuantizer
model = AutoModelForCausalLM.from_pretrained(model_path)
quantizer = GPTQQuantizer(bits=4)  # 选择 4-bit 量化;
quantized_model = quantizer.quantize(model, dataloader=None, device='cuda')
quantized_model.save_pretrained(quantized_path)
"""
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=4,  # 将模型量化为 4-bit 数值类型
    group_size=128,  # 一般推荐将此参数的值设置为 128
    desc_act=False,  # 设为 False 可以显著提升推理速度，但是 ppl 可能会轻微地变差
)

# 加载未量化的模型，默认情况下，模型总是会被加载到 CPU 内存中
model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)

# 量化模型, 样本的数据类型应该为 List[Dict]，其中字典的键有且仅有 input_ids 和 attention_mask
model.quantize(examples)
# 保存量化好的模型
model.save_quantized(quantized_path)

# 使用 safetensors 保存量化好的模型
model.save_quantized(quantized_path, use_safetensors=True)


# 加载量化好的模型到能被识别到的第一块显卡中
#model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")


