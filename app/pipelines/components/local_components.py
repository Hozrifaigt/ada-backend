import os
import asyncio

from typing import List, Generator, Dict, Any
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from concurrent.futures import ThreadPoolExecutor
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

from app.config.settings import settings
from app.pipelines.base import BaseLoader, BaseEmbedder, BaseVectorStore, BaseLLM

from app.prompts.manager import PromptManager

class LocalDocxLoader(BaseLoader):
    def load(self, data_path: str) -> Generator[Document, None, None]:
        for filename in os.listdir(data_path):
            if filename.endswith(".docx"):
                file_path = os.path.join(data_path, filename)
                print(f"Loading document: {file_path}")
                loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
                yield from loader.load()

class LocalEmbedder(BaseEmbedder):
    def __init__(self):
        self.client = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_HOST
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_documents(texts)

    @property
    def vector_size(self) -> int:
        return settings.VECTOR_SIZE

class LocalVectorStore(BaseVectorStore):
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)

    def setup(self, collection_name: str, vector_size: int):
        try:
            self.client.get_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' already exists.")
        except Exception:
            print(f"Collection '{collection_name}' not found. Creating...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
            print("Collection created.")

    def upsert(self, documents: List[Document], collection_name: str):
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=i,  # In a real system, use a better ID (e.g., hash of content)
                    vector=doc.metadata["embedding"],
                    payload={k: v for k, v in doc.metadata.items() if k != "embedding"}
                ) for i, doc in enumerate(documents)
            ],
            wait=True
        )

class LocalLLM(BaseLLM):
    def __init__(self, prompt_manager: PromptManager, max_workers: int = 4):
        self.client = ChatOllama(model=settings.LLM_MODEL, base_url=settings.OLLAMA_HOST)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None

        self.summary_chain = prompt_manager.get_prompt("generate_summary") | self.client
        self.keyword_chain = prompt_manager.get_prompt("generate_keywords") | self.client


    def _generate_for_one_text(self, text: str) -> Dict[str, Any]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)        
        try:
            summary = self.summary_chain.invoke({"text": text}).content
            keywords_str = self.keyword_chain.invoke({"text": text}).content
            keywords = [k.strip() for k in keywords_str.split(",")]
            
            return {"summary": summary, "keywords": keywords}
        finally:
            loop.close()



    async def ainvoke_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Implements the async contract by submitting sync jobs to a thread pool.
        """
        if self.main_loop is None:
            self.main_loop = asyncio.get_running_loop()

        futures = [
            self.main_loop.run_in_executor(self.executor, self._generate_for_one_text, text)
            for text in texts
        ]
        results = await asyncio.gather(*futures)
        return [res for res in results if res is not None]



