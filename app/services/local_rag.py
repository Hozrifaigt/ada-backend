from typing import IO
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
import qdrant_client


from app.core.rag_system import RAGSystem
from app.config.settings import settings


class LocalRAGSystem(RAGSystem):
    pass