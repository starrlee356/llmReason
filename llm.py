from openai import OpenAI
import re
from collections import defaultdict

class LLM:
    def __init__(self, model_name="Qwen2-7B-Instruct"):
        self.model = model_name
        self.base_url = "http://localhost:8000/v1"
        self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)


    def run_llm(self, prompt):
        '''
        对话函数, 提供和LLM对话的功能, 这里提供了2个字符串类型参数
        prompt和text, 方便使用同一个prompt处理不同文本
        Input:
        - client: OpenAI client, 访问接口的客户端
        - prompt: prompt字符串
        - text: prompt中使用可替换的字符串
        - model: 字符串，指代需要调用的模型名字
        Output: 字符串，模型返回的回复
        '''
        messages = [{'role': 'system', 'content': "You're an assistant for answering questions, and when responding, only provide the answer."},
                    {'role': 'user', 'content':prompt}]
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
        

    def format_llm_res(self, res:str):
        if res.startswith("Error"):
            print(res)
            # 其他错误处理代码
        
        pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
        relations_with_scores = defaultdict()
        for match in re.finditer(pattern, res):
            relation = match.group("relation").strip()
            if ';' in relation:
                continue
            score = match.group("score")
            if not relation or not score:
                return False, "output uncompleted.."
            try:
                score = float(score)
            except ValueError:
                return False, "Invalid score"
            relations_with_scores[relation] = score

        if not relations_with_scores:
            return False, "No relations found"
        
        return True, relations_with_scores
