import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)

def save_file(uploaded_file):
    """helper function to save documents to disk"""
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Post-processing
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

class SplitDocs():
    def __init__(self, model_name, chunk_size: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=AutoTokenizer.from_pretrained(
                model_name
            ),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            strip_whitespace=True,
        )
        
    def load_and_split_pdfs(self, file_paths: list):
        loaders = [PyPDFLoader(file_path) for file_path in file_paths]
        pages = []
        for loader in loaders:
            pages.extend(loader.load())

        docs = self.text_splitter.split_documents(pages)
        return docs

def load_and_split_pdfs(model_name, file_paths: list, chunk_size: int = 256):
    loaders = [PyPDFLoader(file_path) for file_path in file_paths]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(
            model_name
        ),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        strip_whitespace=True,
    )
    docs = text_splitter.split_documents(pages)
    return docs