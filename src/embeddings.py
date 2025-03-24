import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
CACHE_DIR = "/home/duyhv/.cache/huggingface/hub"
ACCESS_TOKEN = os.getenv(
        "ACCESS_TOKEN"
    )  # reads .env file with ACCESS_TOKEN=<your hugging face access token>

class Embeddings():
    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2", device="cpu"
    ):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=CACHE_DIR,
            model_kwargs={"device": device, "token": ACCESS_TOKEN},
        )