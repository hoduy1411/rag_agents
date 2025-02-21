

class RAGChat():
    def __init__(self, llm, embedding):
        self.llm = llm
        self.embedding = embedding