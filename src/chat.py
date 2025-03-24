from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import Runnable

from retriever import Retriever
from utils import format_docs

class RAGChat():
    def __init__(self, llm, embedding):
        self.llm = llm
        self.embedding = embedding
        self.retriever = Retriever(docs=[Document(page_content="")], embedding_function=self.embedding.embedding_function)
        self.init_prompt()
        self.init_prompt_query_cons()
        
        self.reload_chain()
        
        self.rag_history_chain = (
            self.prompt_history
            # | self.llm.llm
            | self.llm
            | StrOutputParser()
        )
    
    def reload_chain(self):
        self.rag_chain = (
            {"context": self.retriever.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            # | self.llm.llm
            | self.llm
            | StrOutputParser()
        )
        
    def init_prompt(self,):
        system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể."
        template = '''Chú ý các yêu cầu sau:
        - Câu trả lời phải chính xác và đầy đủ nếu ngữ cảnh có câu trả lời. 
        - Chỉ sử dụng các thông tin có trong ngữ cảnh được cung cấp.
        - Chỉ cần từ chối trả lời và không suy luận gì thêm nếu ngữ cảnh không có câu trả lời.
        Hãy trả lời câu hỏi dựa trên ngữ cảnh:
        ### Ngữ cảnh :
        {context}

        ### Câu hỏi :
        {question}

        ### Trả lời :'''
        
        self.prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", template)])
        
    def init_prompt_query_cons(self,):
        # system_prompt = """Given a chat history and the latest user question \
        # which might reference context in the chat history, formulate a standalone question \
        # which can be understood without the chat history. Do NOT answer the question, \
        # just reformulate it if needed and otherwise return it as is."""
        # system_prompt = "Dựa trên lịch sử trò chuyện và câu hỏi mới nhất của người dùng, có thể tham chiếu đến ngữ cảnh trong lịch sử trò chuyện, hãy tạo ra một câu hỏi độc lập mà có thể hiểu được mà không cần đến lịch sử trò chuyện. ĐỪNG trả lời câu hỏi, chỉ cần diễn đạt lại nếu cần thiết và nếu không cần, hãy trả lại câu hỏi như ban đầu. Không có thông tin cụ thể về câu hỏi trong lịch sử trò chuyện, Vui lòng trả lại {question}."
        # system_prompt = "Dựa trên lịch sử trò chuyện và câu hỏi mới nhất của người dùng, có thể tham chiếu đến ngữ cảnh trong lịch sử trò chuyện, hãy tạo ra một câu hỏi độc lập mà có thể hiểu được mà không cần đến lịch sử trò chuyện. ĐỪNG trả lời câu hỏi, chỉ cần diễn đạt lại nếu cần thiết và nếu không cần, hãy trả lại câu hỏi như ban đầu. Không có thông tin cụ thể về câu hỏi trong lịch sử trò chuyện, Vui lòng trả lại {question}."
#         system_prompt = "Dựa vào toàn bộ lịch sử trò chuyện và câu hỏi mới đưa vào. Hãy trả lời cho tôi một trong ý sau: Trả về nội dung giống câu hỏi đưa vào nhưng mang ý nghĩa cụ thể nếu không thì trả lời giống hệt câu hỏi đưa vào. "
#         history_chat = """
#         ### Lịch sử :
#         {history}

#         ### Câu hỏi :
#         {question}

#         ### Trả lời :"""
        
#         self.prompt_history = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         # MessagesPlaceholder("history"),
#         # ("human", "{question}")
#         ("human", history_chat)
#     ]
# )
        # template = """Given a chat history and the latest user question \
        # which might reference context in the chat history, formulate a standalone question \
        # which can be understood without the chat history. Do NOT answer the question, \
        # just reformulate it if needed and otherwise return it as is. The output is in the same language as the Latest user question.
        
        # ## Chat history:
        # {history}
        
        # ## Latest user quesion:
        # {question}

        # ## Output (1 question):"""
        template = """Dựa trên lịch sử trò chuyện và câu hỏi hiện tại của người dùng, tạo một câu hỏi ngắn gọn và độc lập mà không cần tham chiếu đến lịch sử trò chuyện. Nếu câu hỏi hiện tại không liên quan đến lịch sử, trả lại câu hỏi nguyên bản mà không thay đổi.
## Lịch sử trò chuyện:
{history}

## Câu hỏi hiện tải
{question}"""
        self.prompt_history = ChatPromptTemplate.from_template(template)
        
    def generate(self, question: str):
        return self.rag_chain.invoke(question)