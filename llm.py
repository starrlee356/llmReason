from openai import OpenAI
import openai
import re
from collections import defaultdict
import requests
import json
#from zhipuai import ZhipuAI
import concurrent.futures
#import ollama

class LLM_zhipu:#zhipu api
    def __init__(self, model):#glm-4/glm-4-flash
        self.client = ZhipuAI(api_key="5718d3448fed2423a234c27a4c8b04fe.GsPy9ZPU2IFfUPSS")
        self.model = model
    
    def run(self, prompt):
        response =self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    
class LLM_ollama:#ollama
    def __init__(self, model):
        self.model = model
        self.url = "http://localhost:11434/api/generate"
        self.headers = {"Content-Type": "application/json"}
        
    def process_single_prompt(self, prompt):
        return ollama.generate(model=self.model, prompt=prompt)["response"]
    
    def run(self, prompt):
        if isinstance(prompt, str):
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url=self.url, headers=self.headers, data=json.dumps(data)).text
            res = json.loads(response)["response"]
            return res
        else: #list of prompt, batch
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompt)) as executor:
                res = list(executor.map(self.process_single_prompt, prompt))
                return res


class LLM_vllm: #vllm
    def __init__(self, model):
        self.model = model #model name
        self.client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    
    def run(self, prompt, mode="generation"):
        messages = [{'role': 'system', 'content': 'You are a question-answering assistant.'},
                    {'role': 'user', 'content': prompt}]
        if mode == "generation":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                max_tokens=2048,
                temperature=0,
                presence_penalty=1.1,
                top_p=0.8)
            
            if response:
                return response.choices[0].message.content
            else:
                return f"Error: {response.status_code}"
            
        if mode == "score":
            # raise NotImplementedError

            api_url = "http://localhost:8000/v1/completions"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 5,
                "logprobs": 10, 
            }

            res = requests.post(api_url, json=payload)
            response_data = res.json()
            res_dict = {}
            for d in response_data['choices'][0]['logprobs']['top_logprobs']:
                if len(res_dict) >= 2:
                    break
                for token, prob in d.items():
                    if len(res_dict) >= 2:
                        break
                    if "yes" in token:
                        res_dict["yes"] = prob
                    if "no" in token:
                        res_dict["no"] = prob
            return res_dict
        

                    
    def get_token_length(self):
        # Fetch metrics from the vLLM server
        response = requests.get("http://localhost:8000/metrics")
        metrics = response.text

        # Parse the metrics to find the relevant token counts
        prompt_tokens_total = None
        generation_tokens_total = None

        for line in metrics.splitlines():
            if line.startswith("vllm:prompt_tokens_total"):
                prompt_tokens_total = float(line.split()[-1])
            elif line.startswith("vllm:generation_tokens_total"):
                generation_tokens_total = float(line.split()[-1])

        #print(f"Prompt Tokens Total: {prompt_tokens_total}")
        #print(f"Generation Tokens Total: {generation_tokens_total}")
        return prompt_tokens_total, generation_tokens_total

class LLM_gpt:
    def __init__(self, model="gpt-3.5-turbo"):
        self.base_url = "https://api.openai-proxy.org/v1"
        self.api_key = "sk-zxI3AxBGdFFzK1HSrPgQb8131xI5JGn3HAL2apzc7tmOknEH"
        # self.base_url = "https://api.chatanywhere.tech/v1"
        # self.api_key = "sk-vRuhhaC3IxbknjzUfrTkI8bKZoZ2c8MMfSHk2VGBDfYWgCzJ"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.model = model
    
    def run(self, prompt):
        messages = [{'role': 'system', 'content': 'You are a question-answering assistant.'},
                    {'role': 'user', 'content': prompt}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            max_tokens=2000,
            )
        
        if response:
            return response.choices[0].message.content
        else:
            return f"Error: {response.status_code}"
        
    def get_token_length(self):
        return 0, 0
