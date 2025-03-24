from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

class Retriever:
    def __init__(self, docs, embedding_function, k=5):
        self.db = FAISS.from_documents(
            docs, embedding_function, distance_strategy=DistanceStrategy.COSINE
        )
        self.retriever = self.db.as_retriever(search_kwargs={"k": k})
        
    def get_relevant_documents(self, query, threshold = 0.6):
        results = self.retriever.get_relevant_documents(query)
        print(results)
        
        # Lọc kết quả theo ngưỡng
        filtered_results = [
            doc for doc in results if doc.metadata.get("score", 0) >= threshold
        ]
        
        return filtered_results if filtered_results else []  # Trả về danh sách rỗng nếu không có kết quả phù hợp