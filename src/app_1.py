import os
import streamlit as st
from chat import RAGChat
from llm import LLM, LLM1, OpenAILLM
from embeddings import Embeddings
from retriever import Retriever
from utils import SplitDocs, save_file

FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = ''
os.environ['LANGSMITH_PROJECT']="hands-on-llm"
os.environ['OPENAI_API_KEY']=""


llm_model = "AITeamVN/Vi-Qwen2-3B-RAG"
embeddings_model = "BAAI/bge-small-en-v1.5"


st.title("Chatbot RAG Assistant")


@st.cache_resource
def load_model(llm_model):
    # model = ChatModel(model_id="google/gemma-2b-it", device="cuda")
    # model = ChatModel(model_id="microsoft/Phi-3-mini-4k-instruct", device="cuda")
    # model = LLM(model_id=llm_model, device="cuda")
    model = LLM1(model_id=llm_model, device="cuda")
    
    return model


@st.cache_resource
def load_embeddings(embeddings_model):
    embeddings = Embeddings(
        # model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu"
        model_name=embeddings_model, device="cpu"
    )
    return embeddings


model = load_model(llm_model)  # load our models once and then cache it
# model = OpenAILLM("gpt-3.5-turbo")
embeddings = load_embeddings(embeddings_model)
split_docs = SplitDocs(embeddings_model)

rag_chat = RAGChat(model, embeddings)


with st.sidebar:
    max_new_tokens = st.number_input("max_new_tokens", 128, 4096, 512)
    k = st.number_input("k", 1, 10, 5)
    rag_chat.llm.reload_pipeline(max_new_tokens)
    uploaded_files = st.file_uploader(
        "Upload PDFs for context", type=["PDF", "pdf"], accept_multiple_files=True
    )
    file_paths = []
    for uploaded_file in uploaded_files:
        file_paths.append(save_file(uploaded_file))
    if uploaded_files != []:
        docs = split_docs.load_and_split_pdfs(file_paths)
        rag_chat.retriever = Retriever(docs=docs, embedding_function=rag_chat.embedding.embedding_function, k = k)
    rag_chat.reload_chain()
        
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        user_prompt = st.session_state.messages[-1]["content"]
        question_cons = rag_chat.rag_history_chain.invoke({
            # "history": [tuple(item.values()) for item in st.session_state.messages[:-1]],
            "history": "\n".join([": ".join(list(item.values())) for item in st.session_state.messages[:-1]]),
            "question": user_prompt
        })
        print(question_cons)
        # print([tuple(item.values()) for item in st.session_state.messages[:-1]])
        # answer = rag_chat.generate(
        #     user_prompt
        # )
        answer = "Dựa trên ngữ cảnh được cung cấp, vật thể bỏ quên là các vật thể như ba lô, túi xách, thùng, v.v. bị bỏ quên trong các khu vực công cộng như sân bay, ga tàu, và các vật thể khác. Hệ thống Phát hiện vật thể bỏ quên sử dụng trí tuệ nhân tạo (AI) và thị giác máy tính để giám sát và nhận diện các vật thể này, xác định chúng có bị bỏ quên khi không có người đi kèm trong khoảng thời gian xác định (ví dụ: 1 phút). Nếu phát hiện vật thể bỏ quên, hệ thống sẽ tự động tạo cảnh báo và hiển thị trên giao diện phần mềm, đồng thời gửi thông tin chi tiết về vật thể bỏ quên đến người giám sát để xử lý kịp thời."
        response = st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})