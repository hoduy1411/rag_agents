import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = "/home/duyhv/.cache/huggingface/hub"
# CACHE_DIR = os.path.normpath(
#     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
# )


class ChatModel:
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

    def generate(self, question: str, context: str = None, max_new_tokens: int = 250):

#         if context == None or context == "":
#             prompt = f"""Give a detailed answer to the following question. Question: {question}"""
#         else:
#             prompt = f"""Using the information contained in the context, give a detailed answer to the question.
# Context: {context}.
# Question: {question}"""

#         chat = [{"role": "user", "content": prompt}]
        
        system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể."
        template = f'''Chú ý các yêu cầu sau:
        - Câu trả lời phải chính xác và đầy đủ nếu ngữ cảnh có câu trả lời. 
        - Chỉ sử dụng các thông tin có trong ngữ cảnh được cung cấp.
        - Chỉ cần từ chối trả lời và không suy luận gì thêm nếu ngữ cảnh không có câu trả lời.
        Hãy trả lời câu hỏi dựa trên ngữ cảnh:
        ### Ngữ cảnh :
        {context}

        ### Câu hỏi :
        {question}

        ### Trả lời :'''
        
        chat = [{"role": "system", "content": system_prompt }]
        chat.append({"role": "user", "content": template})
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat,
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
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature = 1,
            )
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=False)
        # response = response[len(formatted_prompt):]  # remove input prompt from reponse
        print(response)
        response = response.replace(self.tokenizer.eos_token, "")  # remove eos token
        print(response)
        return response
