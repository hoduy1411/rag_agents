import os
import streamlit as st
from model import ChatModel
import rag_util


FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)


st.title("Chatbot RAG Assistant")


@st.cache_resource
def load_model():
    # model = ChatModel(model_id="google/gemma-2b-it", device="cuda")
    # model = ChatModel(model_id="microsoft/Phi-3-mini-4k-instruct", device="cuda")
    model = ChatModel(model_id="AITeamVN/Vi-Qwen2-3B-RAG", device="cuda")
    
    return model


@st.cache_resource
def load_encoder():
    encoder = rag_util.Encoder(
        # model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu"
        model_name="BAAI/bge-small-en-v1.5", device="cpu"
    )
    return encoder


model = load_model()  # load our models once and then cache it
encoder = load_encoder()