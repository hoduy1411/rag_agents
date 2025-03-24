import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

CACHE_DIR = "/home/duyhv/.cache/huggingface/hub"


class LLM():
    def __init__(self, model_id: str = "google/gemma-2b-it", device="cuda"):
        self.model_id = model_id
        ACCESS_TOKEN = os.getenv(
            "ACCESS_TOKEN"
        )  # reads .env file with ACCESS_TOKEN=<your hugging face access token>

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=CACHE_DIR, token=ACCESS_TOKEN
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir=CACHE_DIR,
            token=ACCESS_TOKEN,
        )
        self.model.eval()
        
        self.reload_pipeline()
        
    def reload_pipeline(self, max_new_tokens = 512, temperature = 1):
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens, 
            temperature=temperature,
            return_full_text=False,
            # device=0 if device == "cuda" else -1
        )
        

        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        
        # self.llm = HuggingFacePipeline.from_model_id(
        #     model_id=self.model_id,
        #     task="text-generation",
        #     pipeline_kwargs={
        #         "max_new_tokens": max_new_tokens,
        #         # "top_k": 50,
        #         "temperature": temperature,
        #         # "device_map": "cuda"
        #     },
        #     device_map="auto"
            
        # )
        
class LLM1(Runnable):
    def __init__(self, model_id: str = "google/gemma-2b-it", device="cuda"):

        ACCESS_TOKEN = os.getenv(
            "ACCESS_TOKEN"
        )  # reads .env file with ACCESS_TOKEN=<your hugging face access token>

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=CACHE_DIR, token=ACCESS_TOKEN
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir=CACHE_DIR,
            token=ACCESS_TOKEN,
        )
        self.model.eval()
        self.chat = []
        self.device = device

    def reload_pipeline(self, max_new_tokens = 512, temperature = 0.7):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def invoke(self, prompt, config=None):
        print(prompt)
        converted_prompt = []
        for message in prompt.messages:
            role = 'system' if isinstance(message, SystemMessage) else 'user'
            content = message.content
            converted_prompt.append({"role": role, "content": content})
            
        formatted_prompt = self.tokenizer.apply_chat_template(
            converted_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(formatted_prompt)
        inputs = self.tokenizer.encode(
            formatted_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature = self.temperature,
            )
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=False)
        # response = response[len(formatted_prompt):]  # remove input prompt from reponse
        # print(response)
        response = response.replace(self.tokenizer.eos_token, "")  # remove eos token
        print(response)
        return response
    
class OpenAILLM():
    def __init__(self, model_id):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
        
    def reload_pipeline(self, max_new_tokens = 512, temperature = 1):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature