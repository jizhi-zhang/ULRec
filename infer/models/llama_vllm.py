import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

class KeywordsStoppingCriteria:
    def __init__(self, keywords, tokenizer):
        self.keywords = keywords
        self.tokenizer = tokenizer

    def should_stop(self, generated_text):
        # 检查关键词是否出现在生成的文本中
        for keyword in self.keywords:
            if keyword in generated_text:
                return True
        return False

class LlamaInterface:
    def __init__(self, modelpath):
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.model = LLM(modelpath, gpu_memory_utilization=0.8)

    def llama(self, prompt, temperature=1.0, max_tokens=1000, stop=None):
        # encoded_prompt = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        
        # 使用 vllm 生成文本
        outputs = self.model.generate(prompt, sampling_params=sampling_params)
        
        # generated_ids = outputs[0]['token_ids'][0]
        # generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text = outputs[0].outputs[0].text

        if stop:
            for ending in stop:
                if ending in generated_text:
                    generated_text = generated_text.split(ending)[0]
                    break

        return generated_text

    def generate_responses_from_llama(self, prompts, temperature=1.0, max_tokens=1000, n=1, stop=None):
        responses = []
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

        outputs = self.model.generate(prompts, sampling_params=sampling_params)
        for i in range(len(prompts)):
            generated_text = outputs[i].outputs[0].text
            if stop:
                for ending in stop:
                    if ending in generated_text:
                        generated_text = generated_text.split(ending)[0]
                        break
            responses += [generated_text]

        return responses
