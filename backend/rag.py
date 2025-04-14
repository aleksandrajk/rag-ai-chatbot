from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI  # Or Mistral
import os

class RAGEngine:
    def __init__(self, data_dir: str):
        self.llm = OpenAI(model="gpt-3.5-turbo")  # Or Mistral
        self.documents = SimpleDirectoryReader(data_dir).load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = self.index.as_query_engine()

    def query(self, question: str) -> str:
        response = self.query_engine.query(question)
        return str(response)
